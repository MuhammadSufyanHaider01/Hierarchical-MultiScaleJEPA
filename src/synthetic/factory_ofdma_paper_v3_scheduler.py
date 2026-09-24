"""Published factory-OFDMA allocation semantics for the v3 stress-test adapter."""

from __future__ import annotations

import numpy as np

FFT_SIZE = 2048
ACTIVE_SUBCARRIERS = 1320
SYMBOLS = 70


def resource_grids(
    rng: np.random.Generator,
    *,
    transmitter_count: int,
    utilization_rates: list[float],
    pilot_slots: list[int] | tuple[int, ...],
    slot_symbols: int = 14,
    resource_blocks: int = 110,
    subcarriers_per_rb: int = 12,
    out_of_band_suppression_db: float = -20.0,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Generate feasible, non-overlapping RB segments with randomized guard gaps.

    Feasible segment widths are sampled uniformly while reserving at least one
    RB for each not-yet-assigned transmitter. Independent per-slot Bernoulli
    activity uses the supplied U(0.3,1) utilization rates. Pilot slots are
    forced active and carry a known pilot across their allocated RBs.
    """
    if transmitter_count < 1 or len(utilization_rates) != transmitter_count:
        raise ValueError("one utilization rate is required per transmitter")
    if resource_blocks < transmitter_count or SYMBOLS != 5 * 14:
        raise ValueError("invalid transmitter count or five-slot grid")
    if any(not 0.3 <= float(rate) <= 1.0 for rate in utilization_rates):
        raise ValueError("utilization rates must be between 0.3 and 1.0")
    if any(not 0 <= int(slot) < 5 for slot in pilot_slots):
        raise ValueError("pilot slot must be in [0,4]")
    if out_of_band_suppression_db > 0:
        raise ValueError("out-of-band suppression must be non-positive")

    order = rng.permutation(transmitter_count).tolist()
    widths = [0] * transmitter_count
    left = resource_blocks
    for offset, tx in enumerate(order):
        maximum = left - (transmitter_count - offset - 1)
        widths[tx] = int(rng.integers(1, maximum + 1))
        left -= widths[tx]
    gaps = rng.multinomial(left, np.full(transmitter_count + 1, 1.0 / (transmitter_count + 1)))
    starts = [0] * transmitter_count
    cursor = int(gaps[0])
    for offset, tx in enumerate(order):
        starts[tx] = cursor
        cursor += widths[tx] + int(gaps[offset + 1])
    if cursor != resource_blocks:
        raise AssertionError("RB segments and guard gaps do not fill the channel")

    negative = np.arange(FFT_SIZE - ACTIVE_SUBCARRIERS // 2, FFT_SIZE)
    positive = np.arange(1, ACTIVE_SUBCARRIERS // 2 + 1)
    active = np.concatenate((negative, positive))
    if active.size != resource_blocks * subcarriers_per_rb:
        raise ValueError("resource grid parameters do not match 1320 active subcarriers")
    oob = np.setdiff1d(np.arange(FFT_SIZE), active, assume_unique=True)
    oob_scale = 10.0 ** (float(out_of_band_suppression_db) / 20.0)
    grids: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for tx in range(transmitter_count):
        rate = float(utilization_rates[tx])
        active_slots = rng.random(5) < rate
        for pilot_slot in pilot_slots:
            active_slots[int(pilot_slot)] = True
        if not active_slots.any():
            active_slots[int(rng.integers(0, 5))] = True
        first, width = starts[tx], widths[tx]
        carriers = active[first * subcarriers_per_rb:(first + width) * subcarriers_per_rb]
        grid = np.zeros((SYMBOLS, FFT_SIZE), dtype=np.complex64)
        for slot in range(5):
            if not active_slots[slot]:
                continue
            rows = slice(slot * 14, (slot + 1) * 14)
            i = 2 * rng.integers(0, 2, (14, carriers.size)) - 1
            q = 2 * rng.integers(0, 2, (14, carriers.size)) - 1
            grid[rows, carriers] = ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex64)
            if slot in pilot_slots:
                grid[rows, carriers] = 1.0 + 0.0j
            oi = 2 * rng.integers(0, 2, (14, oob.size)) - 1
            oq = 2 * rng.integers(0, 2, (14, oob.size)) - 1
            grid[rows, oob] = ((oi + 1j * oq) / np.sqrt(2.0) * oob_scale).astype(np.complex64)
        grids.append(grid)
        records.append({
            "transmitter_index": tx,
            "first_resource_block": first,
            "allocated_resource_blocks": width,
            "utilization_rate": rate,
            "active_slots": [i for i, active in enumerate(active_slots.tolist()) if active],
            "pilot_slots": [int(x) for x in pilot_slots],
            "out_of_band_suppression_db": float(out_of_band_suppression_db),
        })
    assigned: set[int] = set()
    for record in records:
        block = set(range(int(record["first_resource_block"]),
                          int(record["first_resource_block"]) + int(record["allocated_resource_blocks"])))
        if assigned & block:
            raise AssertionError("centralized OFDMA allocation contains an RB collision")
        assigned |= block
    return grids, records
