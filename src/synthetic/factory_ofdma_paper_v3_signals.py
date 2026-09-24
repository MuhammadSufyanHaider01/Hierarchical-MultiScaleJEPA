"""Paper-configured OFDMA grids, impairment noise, and encoder feature view."""

from __future__ import annotations

import numpy as np

from .factory_ofdma_signals import (
    ACTIVE_SUBCARRIERS,
    CYCLIC_PREFIX,
    FFT_SIZE,
    FEATURE_BINS,
    FEATURE_COLUMNS,
    SAMPLE_RATE_HZ,
    SYMBOLS_PER_OBSERVATION,
    jammer_grid,
)


def paper_ofdma_grid(
    rng: np.random.Generator,
    *,
    utilization: float,
    pilot_slot: int = 2,
    slot_symbols: int = 14,
    subcarriers_per_rb: int = 12,
    resource_blocks: int = 110,
) -> tuple[np.ndarray, dict[str, object]]:
    """Draw contiguous RB allocation and QPSK data with one known pilot slot."""

    if not 0.3 <= utilization <= 1.0:
        raise ValueError("resource utilization must be within the paper range [0.3, 1]")
    if pilot_slot < 0 or (pilot_slot + 1) * slot_symbols > SYMBOLS_PER_OBSERVATION:
        raise ValueError("pilot slot lies outside the observation window")
    active = np.concatenate((
        np.arange(1, ACTIVE_SUBCARRIERS // 2 + 1),
        np.arange(FFT_SIZE - ACTIVE_SUBCARRIERS // 2, FFT_SIZE),
    ))
    max_rb = min(resource_blocks, ACTIVE_SUBCARRIERS // subcarriers_per_rb)
    used_rb = min(max_rb, max(1, int(round(max_rb * utilization))))
    first_rb = int(rng.integers(0, max_rb - used_rb + 1))
    carriers = active[first_rb * subcarriers_per_rb:(first_rb + used_rb) * subcarriers_per_rb]
    grid = np.zeros((SYMBOLS_PER_OBSERVATION, FFT_SIZE), dtype=np.complex64)
    signs = 2 * rng.integers(0, 2, size=(SYMBOLS_PER_OBSERVATION, len(carriers))) - 1
    quadrature = 2 * rng.integers(0, 2, size=(SYMBOLS_PER_OBSERVATION, len(carriers))) - 1
    grid[:, carriers] = ((signs + 1j * quadrature) / np.sqrt(2.0)).astype(np.complex64)
    pilot_slice = slice(pilot_slot * slot_symbols, (pilot_slot + 1) * slot_symbols)
    grid[pilot_slice, carriers] = np.complex64(1.0 + 0.0j)
    return grid, {
        "utilization": float(utilization),
        "first_resource_block": first_rb,
        "allocated_resource_blocks": used_rb,
        "pilot_slot": int(pilot_slot),
    }


def _cp(symbol: np.ndarray) -> np.ndarray:
    return np.concatenate((symbol[..., -CYCLIC_PREFIX:], symbol), axis=-1)


def apply_paper_channels(
    cfr: np.ndarray,
    grids: list[np.ndarray],
    powers_dbm: list[float],
    *,
    rng: np.random.Generator,
    noise_figure_db: float = 10.0,
    additional_impairments_db: float = 15.0,
    bandwidth_hz: float = 20e6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply CFRs with absolute dBm scaling and the paper's thermal-noise budget."""

    channel = np.asarray(cfr)
    if channel.ndim != 3 or not np.iscomplexobj(channel) or channel.shape[1:] != (16, FFT_SIZE):
        raise ValueError("cfr must be complex [transmitter,16,2048]")
    if len(grids) != channel.shape[0] or len(powers_dbm) != channel.shape[0]:
        raise ValueError("transmit waveforms and powers must match CFR transmitter count")
    if any(np.asarray(grid).shape != (SYMBOLS_PER_OBSERVATION, FFT_SIZE) for grid in grids):
        raise ValueError("each waveform must have shape [70,2048]")
    if noise_figure_db < 0 or additional_impairments_db < 0 or bandwidth_hz <= 0:
        raise ValueError("invalid receiver noise parameters")

    symbols = np.zeros((16, SYMBOLS_PER_OBSERVATION, FFT_SIZE + CYCLIC_PREFIX), np.complex64)
    for tx, grid in enumerate(grids):
        occupied = np.count_nonzero(np.max(np.abs(np.asarray(grid)), axis=0) > 0.5)
        if occupied < 1:
            continue
        # Normalize the useful OFDM-symbol energy to one, then apply dBm power.
        scale = FFT_SIZE / np.sqrt(float(occupied))
        amplitude = 10.0 ** ((float(powers_dbm[tx]) - 30.0) / 20.0)
        for symbol_index in range(SYMBOLS_PER_OBSERVATION):
            useful = np.fft.ifft(channel[tx] * grid[symbol_index][None, :], axis=-1)
            symbols[:, symbol_index] += (_cp(useful) * scale * amplitude).astype(np.complex64)
    received = symbols.reshape(16, -1)
    noise_dbm = -174.0 + 10.0 * np.log10(float(bandwidth_hz)) + noise_figure_db + additional_impairments_db
    noise_watts = 10.0 ** ((noise_dbm - 30.0) / 10.0)
    noise = np.sqrt(noise_watts / 2.0) * (
        rng.standard_normal(received.shape) + 1j * rng.standard_normal(received.shape)
    )
    return (received + noise.astype(np.complex64)).astype(np.complex64), {
        "thermal_noise_floor_dbm": float(noise_dbm),
        "noise_figure_db": float(noise_figure_db),
        "additional_impairments_db": float(additional_impairments_db),
        "bandwidth_hz": float(bandwidth_hz),
    }


def paper_iq_to_master(
    received_iq: np.ndarray, *, log_power_mean_db: float, log_power_std_db: float
) -> np.ndarray:
    """Encode full radio time support as `[16,224,896]`; interpolate frequency only."""

    from scipy.signal import stft
    import torch
    import torch.nn.functional as functional

    iq = np.asarray(received_iq)
    expected_samples = SYMBOLS_PER_OBSERVATION * (FFT_SIZE + CYCLIC_PREFIX)
    if iq.shape != (16, expected_samples) or not np.isfinite(iq).all():
        raise ValueError(f"received IQ must be finite [16,{expected_samples}]")
    if not np.isfinite([log_power_mean_db, log_power_std_db]).all() or log_power_std_db <= 0:
        raise ValueError("invalid fixed log-magnitude affine")
    features = np.empty((16, FEATURE_BINS, FEATURE_COLUMNS), np.float32)
    for antenna in range(16):
        _, _, spectrum = stft(
            iq[antenna], fs=SAMPLE_RATE_HZ, nperseg=256, noverlap=85,
            return_onesided=False, boundary="zeros", padded=True,
        )
        spectrum = np.fft.fftshift(spectrum, axes=0)
        if spectrum.shape[1] < FEATURE_COLUMNS:
            raise ValueError(f"STFT produced only {spectrum.shape[1]} columns")
        log_magnitude = 20.0 * np.log10(np.maximum(np.abs(spectrum[:, :FEATURE_COLUMNS]), 1e-12))
        tensor = torch.as_tensor(log_magnitude, dtype=torch.float32)[None, None]
        mapped = functional.interpolate(
            tensor, size=(FEATURE_BINS, FEATURE_COLUMNS), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        features[antenna] = ((mapped - log_power_mean_db) / log_power_std_db).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("paper-adapted features contain non-finite values")
    return features


def paper_jammer_grid(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Use the local six-class emission family; not the paper's five attack labels."""

    return jammer_grid(class_id, rng)
