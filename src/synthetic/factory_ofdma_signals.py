"""Deterministic OFDMA/emitter and fixed log-STFT adapters for the factory study."""

from __future__ import annotations

from typing import Any

import numpy as np


FFT_SIZE = 2048
ACTIVE_SUBCARRIERS = 1320
SYMBOLS_PER_OBSERVATION = 70  # five slots x fourteen OFDM symbols
SAMPLE_RATE_HZ = 30_720_000.0
CYCLIC_PREFIX = 144
FEATURE_BINS = 224
FEATURE_COLUMNS = 896
PATCH_SIZE = 16


def ofdma_grid(rng: np.random.Generator) -> np.ndarray:
    """Generate five slots of QPSK OFDMA symbols on 1,320 active subcarriers."""

    grid = np.zeros((SYMBOLS_PER_OBSERVATION, FFT_SIZE), dtype=np.complex64)
    positive = np.arange(1, ACTIVE_SUBCARRIERS // 2 + 1)
    negative = np.arange(FFT_SIZE - ACTIVE_SUBCARRIERS // 2, FFT_SIZE)
    active = np.concatenate((positive, negative))
    if active.size != ACTIVE_SUBCARRIERS:
        raise AssertionError("active OFDMA carrier count is inconsistent")
    i = 2 * rng.integers(0, 2, size=(SYMBOLS_PER_OBSERVATION, active.size)) - 1
    q = 2 * rng.integers(0, 2, size=(SYMBOLS_PER_OBSERVATION, active.size)) - 1
    grid[:, active] = ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex64)
    return grid


def jammer_grid(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Create the six explicitly defined jammer/no-jammer classes.

    Class IDs are aligned to the frozen local head: DME, NB, NoJam, SingleAM,
    SingleChirp, SingleFM. DME pulses are a simplified DME-like pulse-pair
    emitter, not a protocol-conformant aircraft transponder implementation.
    """

    if not 0 <= int(class_id) < 6:
        raise ValueError("jamming class_id must be in [0,5]")
    if class_id == 2:
        return np.zeros((SYMBOLS_PER_OBSERVATION, FFT_SIZE), dtype=np.complex64)
    t = np.arange(FFT_SIZE, dtype=np.float64) / SAMPLE_RATE_HZ
    grid = np.empty((SYMBOLS_PER_OBSERVATION, FFT_SIZE), dtype=np.complex64)
    phase = float(rng.uniform(-np.pi, np.pi))
    for symbol in range(SYMBOLS_PER_OBSERVATION):
        if class_id == 0:  # DME-like pair: 3.5 us pulses, 12 us separation
            sigma = 0.50e-6
            pulse = np.exp(-0.5 * ((t - 1.75e-6) / sigma) ** 2)
            pulse += np.exp(-0.5 * ((t - 13.75e-6) / sigma) ** 2)
            value = pulse.astype(np.complex128) * np.exp(1j * phase)
        elif class_id == 1:  # narrow-band carrier, randomized small frequency offset
            frequency = float(rng.uniform(-250_000.0, 250_000.0))
            value = np.exp(1j * (2.0 * np.pi * frequency * t + phase))
        elif class_id == 3:  # single-tone carrier with sinusoidal AM
            carrier = float(rng.uniform(1_000_000.0, 4_000_000.0))
            modulation = float(rng.uniform(2_000.0, 30_000.0))
            value = (1.0 + 0.7 * np.sin(2.0 * np.pi * modulation * t)) * np.exp(
                1j * (2.0 * np.pi * carrier * t + phase)
            )
        elif class_id == 4:  # single linear chirp across most of the occupied band
            start = float(rng.uniform(-9_000_000.0, -5_000_000.0))
            stop = float(rng.uniform(5_000_000.0, 9_000_000.0))
            rate = (stop - start) / max(float(t[-1]), 1e-12)
            value = np.exp(1j * (2.0 * np.pi * (start * t + 0.5 * rate * t**2) + phase))
        else:  # single-carrier FM with a low-frequency sinusoidal modulator
            carrier = float(rng.uniform(1_000_000.0, 4_000_000.0))
            modulation = float(rng.uniform(5_000.0, 80_000.0))
            deviation = float(rng.uniform(100_000.0, 800_000.0))
            instantaneous_phase = 2.0 * np.pi * carrier * t - deviation / modulation * np.cos(
                2.0 * np.pi * modulation * t
            )
            value = np.exp(1j * (instantaneous_phase + phase))
        value = value * np.exp(1j * float(rng.uniform(-np.pi, np.pi)))
        grid[symbol] = np.fft.fft(value).astype(np.complex64)
    return grid


def _with_cyclic_prefix(symbol: np.ndarray) -> np.ndarray:
    return np.concatenate((symbol[..., -CYCLIC_PREFIX:], symbol), axis=-1)


def apply_factory_channels(
    cfr: np.ndarray,
    tx_grids: list[np.ndarray],
    *,
    tx_relative_dbm: list[float],
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one static ray-traced CFR per transmitter to the OFDMA symbols."""

    channel = np.asarray(cfr)
    if channel.ndim != 3 or not np.iscomplexobj(channel):
        raise ValueError("cfr must have shape [transmitter,rx_element,frequency]")
    if channel.shape[1:] != (16, FFT_SIZE):
        raise ValueError(f"expected CFR trailing shape [16,{FFT_SIZE}], got {channel.shape}")
    if len(tx_grids) != channel.shape[0] or len(tx_relative_dbm) != channel.shape[0]:
        raise ValueError("transmitter grids/powers must align with CFR transmitters")
    symbols = tx_grids[0].shape[0]
    if any(grid.shape != (symbols, FFT_SIZE) for grid in tx_grids):
        raise ValueError("all transmitter grids must share [symbols,2048] shape")

    output = np.zeros((16, symbols, FFT_SIZE + CYCLIC_PREFIX), dtype=np.complex64)
    for tx_index, grid in enumerate(tx_grids):
        amplitude = 10.0 ** (float(tx_relative_dbm[tx_index]) / 20.0)
        for symbol_index in range(symbols):
            received_spectrum = channel[tx_index] * grid[symbol_index][None, :]
            useful = np.fft.ifft(received_spectrum, axis=-1).astype(np.complex64)
            output[:, symbol_index] += _with_cyclic_prefix(useful) * amplitude
    output = output.reshape(16, -1)
    signal_power = float(np.mean(np.abs(output) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(output.shape) + 1j * rng.standard_normal(output.shape)
    )
    return (output + noise.astype(np.complex64)).astype(np.complex64)


def iq_to_factory_master(
    received_iq: np.ndarray,
    *,
    log_power_mean: float,
    log_power_std: float,
) -> np.ndarray:
    """Transform a complete IQ window to a single fixed `[16,224,896]` master.

    STFT time columns are never resized. The final columns beyond the registered
    896-column window are discarded; frequency alone is bilinearly mapped to
    the encoder's 224-bin input. One fixed affine is used for every sample and
    all availability conditions.
    """

    from scipy.signal import stft
    import torch
    import torch.nn.functional as functional

    iq = np.asarray(received_iq)
    if iq.shape != (16, SYMBOLS_PER_OBSERVATION * (FFT_SIZE + CYCLIC_PREFIX)):
        raise ValueError(f"unexpected received IQ shape: {iq.shape}")
    if not np.isfinite(iq).all() or not np.isfinite([log_power_mean, log_power_std]).all():
        raise ValueError("received IQ and feature affine must be finite")
    if log_power_std <= 0:
        raise ValueError("log_power_std must be positive")

    features = np.empty((16, FEATURE_BINS, FEATURE_COLUMNS), dtype=np.float32)
    for element in range(16):
        _, _, spectrum = stft(
            iq[element], fs=SAMPLE_RATE_HZ, nperseg=256, noverlap=85,
            return_onesided=False, boundary="zeros", padded=True,
        )
        spectrum = np.fft.fftshift(spectrum, axes=0)
        if spectrum.shape[1] < FEATURE_COLUMNS:
            raise ValueError(f"STFT produced only {spectrum.shape[1]} columns")
        log_power = 20.0 * np.log10(np.abs(spectrum[:, :FEATURE_COLUMNS]) + 1e-12)
        tensor = torch.as_tensor(log_power, dtype=torch.float32)[None, None]
        mapped = functional.interpolate(
            tensor, size=(FEATURE_BINS, FEATURE_COLUMNS),
            mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        features[element] = ((mapped - log_power_mean) / log_power_std).astype(
            np.float32, copy=False
        )
    if not np.isfinite(features).all():
        raise ValueError("factory-OFDMA features contain non-finite values")
    return features


def metadata_signal_summary(
    *, task: str, label: int, snr_db: float, jammer_power_dbm: float | None
) -> dict[str, Any]:
    return {
        "task": task,
        "label_id": int(label),
        "snr_db": float(snr_db),
        "jammer_power_dbm": None if jammer_power_dbm is None else float(jammer_power_dbm),
        "waveform_grid": {
            "subcarriers": ACTIVE_SUBCARRIERS,
            "fft_size": FFT_SIZE,
            "symbols": SYMBOLS_PER_OBSERVATION,
            "slots": 5,
            "ofdm_symbols_per_slot": 14,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "feature_stft_nperseg": 256,
            "feature_stft_hop": 171,
        },
    }
