"""Paper-config factory OFDMA adapter with a distinct mobile-array extension."""

from __future__ import annotations

from typing import Any

import numpy as np

from .factory_ofdma_paper_v3_scheduler import resource_grids
from .factory_ofdma_paper_v3_rt import sample_receiver_position, sample_transmitter_positions, trace_factory_cfr
from .factory_ofdma_paper_v3_signals import apply_paper_channels, paper_iq_to_master, paper_jammer_grid
from .factory_ofdma_signals import FFT_SIZE, SAMPLE_RATE_HZ


JAMMER_POWER_LEVELS_DBM = (0, 5, 10, 15, 20, 25, 30)


def factory_ofdma_paper_v3_realization(
    *, realization_id: str, seed: int, task_profile: Any, task_config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one signal realization and the declared synthetic-domain label."""
    task = str(task_profile.name)
    if task not in {"nrpos-outdoor", "interference", "jamming"}:
        raise ValueError(f"unsupported task {task}")
    rng = np.random.default_rng(int(seed))
    paper = task_config["paper_radio_parameters"]
    tx_min, tx_max = map(int, paper["legitimate_transmitter_count"])
    legit_count = int(rng.integers(tx_min, tx_max + 1))
    tx_height = float(paper["transmitter_height_m"])
    legitimate_positions = sample_transmitter_positions(seed, legit_count, height_m=tx_height)
    receiver = sample_receiver_position(seed ^ 0x5A17)

    if task == "interference":
        label_id = int(rng.integers(0, 2))
        jammer_present = bool(label_id)
    elif task == "jamming":
        label_id = int(rng.integers(0, task_profile.num_outputs))
        jammer_present = label_id != 2
    else:
        label_id = -1
        jammer_present = bool(rng.random() < float(paper["jammer_probability"]))

    jammer_position = (sample_transmitter_positions(seed ^ 0xA913, 1, height_m=tx_height)
                       if jammer_present else np.empty((0, 3), np.float32))
    frequencies = 3.75e9 + np.fft.fftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE_HZ)
    rt = task_config["sionna"]
    ray_count = int(rt.get("ray_count_per_source", paper["ray_count_per_source"]))
    reflection_depth = int(rt.get("reflection_depth", paper["reflection_depth"]))
    cfr, rt_metadata = trace_factory_cfr(
        receiver, frequencies, transmitter_positions_m=legitimate_positions, seed=seed,
        ray_count=ray_count, reflection_depth=reflection_depth,
        refraction=bool(rt.get("refraction", False)),
        diffraction=bool(rt.get("diffraction", paper["diffraction"])), tx_pattern="iso",
    )
    jammer_orientation = None
    if jammer_present:
        jammer_orientation = [0.0, 0.0, float(rng.uniform(-np.pi, np.pi))]
        jammer_cfr, jammer_rt_metadata = trace_factory_cfr(
            receiver, frequencies, transmitter_positions_m=jammer_position, seed=seed ^ 0xB117,
            ray_count=ray_count, reflection_depth=reflection_depth,
            refraction=bool(rt.get("refraction", False)),
            diffraction=bool(rt.get("diffraction", paper["diffraction"])),
            tx_pattern="tr38901", tx_orientations=[jammer_orientation],
        )
        cfr = np.concatenate((cfr, jammer_cfr), axis=0)
        rt_metadata["jammer_ray_tracing"] = jammer_rt_metadata

    utilization_rates = [float(rng.uniform(*map(float, paper["legitimate_resource_utilization"])))
                         for _ in range(legit_count)]
    grids, allocations = resource_grids(
        rng, transmitter_count=legit_count, utilization_rates=utilization_rates,
        pilot_slots=paper["pilot_slots"],
        slot_symbols=int(paper["symbols_per_slot"]),
        subcarriers_per_rb=int(paper["subcarriers_per_resource_block"]),
        resource_blocks=int(paper["resource_blocks"]),
        out_of_band_suppression_db=float(paper["out_of_band_suppression_db"]),
    )
    powers = [float(paper["legitimate_transmitter_power_dbm"])] * legit_count
    jammer_power: float | None = None
    if jammer_present:
        jammer_power = float(rng.choice(JAMMER_POWER_LEVELS_DBM))
        if task == "interference":
            extra_grid, extra_allocation = resource_grids(
                rng, transmitter_count=1, utilization_rates=[1.0],
                pilot_slots=paper["pilot_slots"],
                slot_symbols=int(paper["symbols_per_slot"]),
                subcarriers_per_rb=int(paper["subcarriers_per_resource_block"]),
                resource_blocks=int(paper["resource_blocks"]),
                out_of_band_suppression_db=float(paper["out_of_band_suppression_db"]),
            )
            grids.append(extra_grid[0])
            allocations.append({"role": "uncoordinated_added_cochannel_source", **extra_allocation[0]})
        elif task == "jamming":
            grids.append(paper_jammer_grid(label_id, rng))
        else:
            synthetic_jammer_id = int(rng.choice((0, 1, 3, 4, 5)))
            grids.append(paper_jammer_grid(synthetic_jammer_id, rng))
        powers.append(jammer_power)

    received, noise_metadata = apply_paper_channels(
        cfr, grids, powers, rng=rng,
        noise_figure_db=float(paper["sensing_unit_noise_figure_db"]),
        additional_impairments_db=float(paper["additional_impairments_db"]),
        bandwidth_hz=float(paper["bandwidth_hz"]),
    )
    feature_config = task_config.get("features", {})
    features = paper_iq_to_master(
        received,
        log_power_mean_db=float(feature_config.get("log_power_mean_db", -100.0)),
        log_power_std_db=float(feature_config.get("log_power_std_db", 20.0)),
    )
    if task == "nrpos-outdoor":
        lo, hi = np.asarray(task_profile.coord_min, np.float32), np.asarray(task_profile.coord_max, np.float32)
        label = 2.0 * (receiver - lo) / (hi - lo) - 1.0
    else:
        label = np.asarray(label_id, dtype=np.int64)
    return features, label, {
        "simulator_parameters": {
            **rt_metadata,
            "task": task, "realization_id": str(realization_id), "seed": int(seed),
            "receiver_position_m": receiver.astype(float).tolist(),
            "legitimate_transmitter_count": legit_count,
            "legitimate_transmitter_resource_allocations": allocations,
            "legitimate_transmitter_power_dbm": float(paper["legitimate_transmitter_power_dbm"]),
            "jammer_present": jammer_present, "jammer_label_id": label_id,
            "jammer_position_m": jammer_position.astype(float).tolist() if jammer_present else None,
            "jammer_orientation_rpy_rad": jammer_orientation,
            "jammer_power_dbm": jammer_power,
            "paper_jammer_pattern": paper["jammer_pattern"],
            "implemented_jammer_pattern": "tr38901_directional_pattern_in_separate_jammer_trace",
            "noise": noise_metadata,
            "waveform": {
                "carrier_frequency_hz": int(paper["carrier_frequency_hz"]),
                "bandwidth_hz": float(paper["bandwidth_hz"]),
                "subcarrier_spacing_hz": float(paper["subcarrier_spacing_hz"]),
                "resource_blocks": int(paper["resource_blocks"]),
                "subcarriers_per_resource_block": int(paper["subcarriers_per_resource_block"]),
                "slots": int(paper["slots_per_observation"]),
                "symbols_per_slot": int(paper["symbols_per_slot"]),
                "fft_size": FFT_SIZE, "cyclic_prefix_samples": int(paper["cyclic_prefix_samples"]),
                "bits_per_symbol": int(paper["qam_bits_per_symbol"]),
                "pilot_slots": list(paper["pilot_slots"]),
                "out_of_band_suppression_db": float(paper["out_of_band_suppression_db"]),
            },
            "label_map": {str(k): v for k, v in task_profile.label_map.items()},
            "declared_synthetic_domain": "factory_ofdma_mobile_array_paper_v3",
            "semantic_status": "assumption_based_synthetic_stress_test",
        }
    }
