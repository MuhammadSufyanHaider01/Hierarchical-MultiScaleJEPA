"""Factory/OFDMA realization generator using the pinned Sionna RT interface.

The room and six obstacle meshes are vendored from the Apache-2.0 reference
repository identified in ``configs/synthetic_trained_heads/factory_ofdma_v1.yml``.
Published geometry/radio values are retained; receiver array, moving receiver,
multi-task labels, feature raster, and waveform adapters are study extensions.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np


JAMMER_LABELS = {
    0: "no_jammer",
    1: "barrage",
    2: "deceptive",
    3: "pilot",
    4: "random_hop",
    5: "sweep",
}
JAMMER_TYPES = {value: key for key, value in JAMMER_LABELS.items()}
NFFT = 2048
CP_LEN = 144
N_RB = 110
SC_PER_RB = 12
N_SYMBOLS = 70
N_RX_ANT = 16
N_FREQUENCY_FEATURES = 224
N_TIME_FEATURES = 896
STFT_HOP = 168
PATCH_SIZE = 16


def balanced_jammer_labels(count: int, rng: np.random.Generator) -> np.ndarray:
    """Build the reference 50%-jammed mixture with equal jammer-type support."""

    if count < 6:
        raise ValueError("at least six realizations are required to include every jammer class")
    no_jammer = count // 2
    jammed = count - no_jammer
    counts = np.full(5, jammed // 5, dtype=np.int64)
    counts[: jammed % 5] += 1
    labels = np.concatenate(
        [np.zeros(no_jammer, dtype=np.int64)]
        + [np.full(int(n), mode, dtype=np.int64) for mode, n in zip(range(1, 6), counts)]
    )
    return labels[rng.permutation(count)]


def _inside_obstacle(x: float, y: float) -> bool:
    x_ranges = ((7.0, 11.0), (18.0, 22.0), (29.0, 33.0))
    y_ranges = ((4.0, 18.0), (22.0, 36.0))
    return any(a <= x <= b for a, b in x_ranges) and any(c <= y <= d for c, d in y_ranges)


def _free_xy(rng: np.random.Generator, *, margin: float = 0.75) -> tuple[float, float]:
    for _ in range(10000):
        x, y = rng.uniform(margin, 40.0 - margin, size=2)
        if not _inside_obstacle(float(x), float(y)):
            return float(x), float(y)
    raise RuntimeError("could not sample a free factory-floor coordinate")


def _split_resource_blocks(n_tx: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Partition the 110 RBs into non-overlapping contiguous user allocations."""

    if not 1 <= n_tx <= N_RB:
        raise ValueError(f"invalid number of transmitters: {n_tx}")
    weights = rng.dirichlet(np.ones(n_tx))
    raw = weights * N_RB
    lengths = np.maximum(1, np.floor(raw).astype(int))
    while lengths.sum() > N_RB:
        candidates = np.flatnonzero(lengths > 1)
        lengths[candidates[np.argmin(raw[candidates] - lengths[candidates])]] -= 1
    while lengths.sum() < N_RB:
        lengths[np.argmax(raw - lengths)] += 1
    bounds = []
    start = 0
    for length in lengths:
        bounds.append((start, start + int(length)))
        start += int(length)
    return bounds


def _qpsk(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    bits = rng.integers(0, 2, size=shape + (2,), dtype=np.int8)
    return ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(2.0)


def _occupied_fft_bins() -> np.ndarray:
    # 1320 active LTE/NR-like subcarriers centered around DC with the DC tone
    # omitted, as in the published 110-RB allocation.
    negative = np.arange(NFFT - 660, NFFT)
    positive = np.arange(1, 661)
    return np.concatenate((negative, positive)).astype(np.int64)


def _make_legitimate_grids(
    *, n_tx: int, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
    """Return [transmitter, OFDM-symbol, FFT-bin] QPSK resource grids."""

    grids = np.zeros((n_tx, N_SYMBOLS, NFFT), dtype=np.complex64)
    allocations = _split_resource_blocks(n_tx, rng)
    utilization = []
    bins = _occupied_fft_bins()
    for tx_idx, (rb0, rb1) in enumerate(allocations):
        use_probability = float(rng.uniform(*config["scenario"]["legitimate_transmitters"]["resource_utilization"]))
        active_slots = rng.random(5) < use_probability
        if not active_slots.any():
            active_slots[int(rng.integers(0, 5))] = True
        utilization.append(use_probability)
        first = rb0 * SC_PER_RB
        last = rb1 * SC_PER_RB
        user_bins = bins[first:last]
        for slot in range(5):
            if not active_slots[slot]:
                continue
            symbol_start = slot * 14
            symbol_stop = symbol_start + 14
            grids[tx_idx, symbol_start:symbol_stop][:, user_bins] = _qpsk(
                rng, (14, user_bins.size)
            )
            if slot in config["scenario"]["pilot_slots"]:
                # Known unit-power pilot sequence, one phase per user.
                phase = np.exp(2j * np.pi * tx_idx / max(1, n_tx))
                grids[tx_idx, symbol_start:symbol_stop][:, user_bins] = phase
    return grids, allocations, utilization


def _make_jammer_grid(
    *,
    mode: int,
    power_dbm: float,
    legitimate_grids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if mode == 0:
        return None
    grid = np.zeros((N_SYMBOLS, NFFT), dtype=np.complex64)
    bins = _occupied_fft_bins()
    if mode == 1:  # barrage: independent QPSK over the full occupied band
        grid[:, bins] = _qpsk(rng, (N_SYMBOLS, bins.size))
    elif mode == 2:  # deceptive: replay one user's resource/pilot pattern with delay/phase
        user = int(rng.integers(legitimate_grids.shape[0]))
        symbols = legitimate_grids[user].copy()
        delay = int(rng.integers(1, 8))
        grid = np.roll(symbols, shift=delay, axis=0) * np.exp(1j * rng.uniform(-np.pi, np.pi))
    elif mode == 3:  # pilot: known sequence aligned with the declared pilot slot
        slot = 2
        symbol_start = slot * 14
        grid[symbol_start : symbol_start + 14][:, bins] = 1.0 + 0.0j
    elif mode == 4:  # random-hop: 20% contiguous RB band, new center each slot
        hop_rbs = max(1, N_RB // 5)
        for slot in range(5):
            rb0 = int(rng.integers(0, N_RB - hop_rbs + 1))
            slot_bins = bins[rb0 * SC_PER_RB : (rb0 + hop_rbs) * SC_PER_RB]
            start, stop = slot * 14, (slot + 1) * 14
            grid[start:stop][:, slot_bins] = _qpsk(rng, (14, slot_bins.size))
    elif mode == 5:  # sweep: a 1/6-band contiguous allocation moves each symbol
        width = max(1, N_RB // 6) * SC_PER_RB
        for symbol in range(N_SYMBOLS):
            rb0 = int(round(symbol * (N_RB - width // SC_PER_RB) / (N_SYMBOLS - 1)))
            start = rb0 * SC_PER_RB
            slot_bins = bins[start : start + width]
            grid[symbol, slot_bins] = _qpsk(rng, (slot_bins.size,))
    else:
        raise ValueError(f"unsupported jammer mode index {mode}")
    return grid


def _power_normalize_grid(grid: np.ndarray, target_dbm: float) -> np.ndarray:
    """Scale each active OFDM symbol to the requested average transmit power."""

    time_signal = np.fft.ifft(grid, axis=-1)
    mean_power = np.mean(np.abs(time_signal) ** 2, axis=-1, keepdims=True)
    target_watt = 1.0e-3 * 10.0 ** (float(target_dbm) / 10.0)
    scale = np.sqrt(target_watt / np.maximum(mean_power, 1.0e-20))
    scale = np.where(mean_power > 0, scale, 0.0)
    return (grid * scale).astype(np.complex64)


def make_scene(scene_path: Path, config: dict[str, Any], *, tx_pattern: str = "iso"):
    """Load the pinned room and apply its published ITU material mapping."""

    import sionna.rt as rt

    scene = rt.load_scene(str(scene_path), merge_shapes=False)
    scene.frequency = float(config["scenario"]["carrier_frequency_hz"])
    for object_name in scene.objects:
        if "ceiling" in object_name:
            material = rt.ITURadioMaterial(
                name=f"study_{object_name}", itu_type="concrete", thickness=0.2
            )
        elif "room" in object_name:
            material = rt.ITURadioMaterial(
                name=f"study_{object_name}", itu_type="concrete", thickness=0.2
            )
        else:
            material = rt.ITURadioMaterial(
                name=f"study_{object_name}", itu_type="metal", thickness=0.02
            )
        scene.objects[object_name].radio_material = material

    scenario = config["scenario"]
    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=tx_pattern,
        polarization="V",
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=N_RX_ANT,
        vertical_spacing=0.5,
        horizontal_spacing=float(scenario["receiver"]["spacing_fraction_wavelength"]),
        pattern="iso",
        polarization="V",
    )
    return scene


def _trace_cfr(
    scene,
    *,
    names: list[str],
    receiver_name: str,
    seed: int,
    config: dict[str, Any],
    ray_count_per_source: int | None = None,
) -> np.ndarray:
    import sionna.rt as rt

    solver = rt.PathSolver(deterministic=True)
    propagation = config["scenario"]["propagation"]
    paths = solver(
        scene=scene,
        max_depth=int(propagation["reflection_depth"]),
        samples_per_src=int(
            propagation["ray_count_per_source"]
            if ray_count_per_source is None
            else ray_count_per_source
        ),
        max_num_paths_per_src=100000,
        synthetic_array=False,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=bool(propagation["refraction"]),
        diffraction=bool(propagation["diffraction"]),
        seed=int(seed),
    )
    nfft = int(config["scenario"]["fft_size"])
    subcarrier_spacing = float(config["scenario"]["subcarrier_spacing_hz"])
    frequency = float(config["scenario"]["carrier_frequency_hz"])
    frequencies = rt.utils.subcarrier_frequencies(nfft, subcarrier_spacing) + frequency
    cfr = paths.cfr(
        frequencies,
        sampling_frequency=float(config["scenario"]["sample_rate_hz"]),
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )
    # Sionna RT returns [rx, rx_ant, tx, tx_ant, time, frequency]. Convert
    # from increasing absolute frequency order to NumPy FFT-bin order.
    result = np.fft.ifftshift(cfr[0, :, :, 0, 0, :], axes=-1)
    for name in names:
        scene.remove(name)
    del paths, cfr
    return np.asarray(result, dtype=np.complex64)


def received_capture(
    *,
    scene,
    receiver_xyz: np.ndarray,
    jammer_mode: int,
    rng: np.random.Generator,
    ray_seed: int,
    config: dict[str, Any],
    ray_count_per_source: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a 16-antenna time capture through Sionna RT and OFDMA waveforms."""

    import sionna.rt as rt

    tx_profile = config["scenario"]["legitimate_transmitters"]
    n_tx = int(rng.integers(tx_profile["count_inclusive"][0], tx_profile["count_inclusive"][1] + 1))
    receiver_name = "study_rx"
    receiver_xyz = np.asarray(receiver_xyz, dtype=np.float64)
    scene.add(rt.Receiver(name=receiver_name, position=receiver_xyz.tolist()))

    tx_positions = np.asarray([_free_xy(rng) + (float(tx_profile["height_m"]),) for _ in range(n_tx)])
    tx_names = [f"study_tx_{idx}" for idx in range(n_tx)]
    for name, position in zip(tx_names, tx_positions, strict=True):
        scene.add(
            rt.Transmitter(
                name=name,
                position=position.tolist(),
                power_dbm=float(tx_profile["power_dbm"]),
            )
        )

    channel_legit = _trace_cfr(
        scene,
        names=tx_names,
        receiver_name=receiver_name,
        seed=ray_seed,
        config=config,
        ray_count_per_source=ray_count_per_source,
    )
    legit_grid, allocations, utilizations = _make_legitimate_grids(
        n_tx=n_tx, rng=rng, config=config
    )
    legit_grid = _power_normalize_grid(legit_grid, float(tx_profile["power_dbm"]))
    received_freq = np.einsum(
        "atf,tsf->asf", channel_legit, legit_grid, optimize=True
    )

    jammer_metadata: dict[str, Any] = {"present": False, "position_m": None, "power_dbm": None}
    if jammer_mode:
        jammer_profile = config["scenario"]["jammer"]
        jammer_position = np.asarray(_free_xy(rng) + (float(tx_profile["height_m"]),))
        jammer_power = float(rng.choice(jammer_profile["power_dbm_values"]))
        jammer_orientation = float(rng.uniform(0.0, 2.0 * np.pi))
        jammer_name = "study_jammer"
        scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )
        scene.add(
            rt.Transmitter(
                name=jammer_name,
                position=jammer_position.tolist(),
                orientation=[jammer_orientation, 0.0, 0.0],
                power_dbm=jammer_power,
            )
        )
        channel_jammer = _trace_cfr(
            scene,
            names=[jammer_name],
            receiver_name=receiver_name,
            seed=ray_seed + 1,
            config=config,
            ray_count_per_source=ray_count_per_source,
        )
        jammer_grid = _make_jammer_grid(
            mode=jammer_mode,
            power_dbm=jammer_power,
            legitimate_grids=legit_grid,
            rng=rng,
        )
        assert jammer_grid is not None
        jammer_grid = _power_normalize_grid(jammer_grid[None, ...], jammer_power)[0]
        received_freq += np.einsum(
            "af,sf->asf", channel_jammer[:, 0, :], jammer_grid, optimize=True
        )
        jammer_metadata = {
            "present": True,
            "position_m": jammer_position.tolist(),
            "orientation_rad": jammer_orientation,
            "power_dbm": jammer_power,
            "mode": JAMMER_LABELS[jammer_mode],
        }
        del channel_jammer, jammer_grid
        # The next realization uses the isotropic legitimate pattern.
        scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

    # Convert each frequency-domain OFDM symbol to time domain and add CP.
    time_symbols = np.fft.ifft(received_freq, axis=-1).astype(np.complex64)
    cp = time_symbols[..., -CP_LEN:]
    capture = np.concatenate((cp, time_symbols), axis=-1).reshape(N_RX_ANT, -1)

    # The reference adds thermal noise, receiver noise figure, and an additional
    # impairment margin. Noise is complex circular Gaussian with total power
    # defined over the configured monitored bandwidth.
    noise_dbm = (
        -174.0
        + 10.0 * np.log10(float(config["scenario"]["bandwidth_hz"]))
        + float(config["scenario"]["propagation"]["sensing_unit_noise_figure_db"])
        + float(config["scenario"]["propagation"]["additional_impairments_db"])
    )
    noise_watt = 1.0e-3 * 10.0 ** (noise_dbm / 10.0)
    noise = np.sqrt(noise_watt / 2.0) * (
        rng.standard_normal(capture.shape) + 1j * rng.standard_normal(capture.shape)
    )
    capture = (capture + noise.astype(np.complex64)).astype(np.complex64)
    metadata = {
        "transmitter_count": n_tx,
        "transmitter_positions_m": tx_positions.tolist(),
        "resource_block_ranges": [list(value) for value in allocations],
        "resource_utilization_draws": utilizations,
        "jammer": jammer_metadata,
        "noise_power_dbm": float(noise_dbm),
    }
    scene.remove(receiver_name)
    del channel_legit, legit_grid, received_freq, time_symbols, cp
    gc.collect()
    return capture, metadata


def capture_to_feature(capture: np.ndarray) -> np.ndarray:
    """Convert complex baseband captures to [16,224,896] dBm spectrograms.

    Frequency channels are selected, not interpolated. Time columns are the
    first 896 non-padded Hann-window STFT frames from the 70-symbol capture.
    """

    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(capture, dtype=np.complex64)
    if x.ndim != 2 or x.shape[0] != N_RX_ANT:
        raise ValueError(f"expected capture shape [{N_RX_ANT}, samples], got {x.shape}")
    window = np.hanning(NFFT).astype(np.float32)
    active = _occupied_fft_bins()
    shifted_active = np.sort((active + NFFT // 2) % NFFT)
    selected_offsets = np.rint(np.linspace(0, shifted_active.size - 1, N_FREQUENCY_FEATURES)).astype(int)
    selected_bins = shifted_active[selected_offsets]
    output = np.empty((N_RX_ANT, N_FREQUENCY_FEATURES, N_TIME_FEATURES), dtype=np.float32)
    for antenna in range(N_RX_ANT):
        frames = sliding_window_view(x[antenna], NFFT)[::STFT_HOP][:N_TIME_FEATURES]
        if frames.shape[0] != N_TIME_FEATURES:
            raise ValueError(
                f"capture has {frames.shape[0]} STFT frames; {N_TIME_FEATURES} are required"
            )
        spectrum = np.fft.fftshift(np.fft.fft(frames * window, axis=-1), axes=-1)
        amplitude = np.abs(spectrum[:, selected_bins]) / float(window.sum())
        power_watt = amplitude**2
        power_dbm = 10.0 * np.log10(np.maximum(power_watt, 1.0e-18)) + 30.0
        output[antenna] = power_dbm.T.astype(np.float32)
    if output.shape != (16, 224, 896) or not np.isfinite(output).all():
        raise ValueError("feature transform returned an invalid shape or non-finite values")
    return output


def generate_realization(
    *,
    scene,
    jammer_mode: int,
    seed: int,
    config: dict[str, Any],
    ray_count_per_source: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate a single paired-task master observation and task metadata."""

    rng = np.random.default_rng(seed)
    x, y = _free_xy(rng)
    receiver_xyz = np.asarray([x, y, float(config["scenario"]["receiver"]["height_m"])], dtype=np.float32)
    capture, metadata = received_capture(
        scene=scene,
        receiver_xyz=receiver_xyz,
        jammer_mode=jammer_mode,
        rng=rng,
        ray_seed=seed,
        config=config,
        ray_count_per_source=ray_count_per_source,
    )
    features = capture_to_feature(capture)
    metadata.update({"receiver_xyz_m": receiver_xyz.tolist(), "jammer_mode_index": int(jammer_mode)})
    return features, receiver_xyz, metadata
