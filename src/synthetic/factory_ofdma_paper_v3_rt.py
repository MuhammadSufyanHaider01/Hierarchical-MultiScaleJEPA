"""Sionna RT factory-OFDMA scene used by the synthetic availability study.

Published factory parameters are transcribed in
``configs/sionna/factory_ofdma_paper_v3.yml``. The mobile 16-element
receiver and three-dimensional route are explicit study extensions.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


HALL_M = (40.0, 40.0, 5.0)
PRODUCTION_LINE_CENTERS_M = tuple(
    (x, y, 1.5) for y in (10.0, 30.0) for x in (8.0, 20.0, 32.0)
)
DEFAULT_TX_POSITIONS_M = (
    (8.0, 7.0, 4.5),
    (32.0, 7.0, 4.5),
    (20.0, 34.0, 4.5),
    (20.0, 20.0, 4.5),  # co-channel source / jammer; signal may be zero
)
C0_M_PER_S = 299_792_458.0
FC_HZ = 3_750_000_000.0
RX_ELEMENTS = 16
RX_SPACING_M = C0_M_PER_S / FC_HZ / 2.0


def _box(name: str, scale: Sequence[float], center: Sequence[float], material: str) -> str:
    sx, sy, sz = (float(v) for v in scale)
    x, y, z = (float(v) for v in center)
    return (
        f'<shape type="cube" id="{name}">'
        '<transform name="to_world">'
        f'<scale x="{sx}" y="{sy}" z="{sz}"/>'
        f'<translate x="{x}" y="{y}" z="{z}"/>'
        '</transform>'
        f'<ref id="{material}" name="bsdf"/></shape>'
    )


def factory_scene_xml() -> str:
    """Return a deterministic 40 x 40 x 5 m simplified factory scene.

    It implements the published six production-line cuboids (4 x 14 x 3 m,
    rotated in plan to fit the hall), concrete boundary surfaces, and metal
    production lines. This is a compact scene definition, not a recovered
    Blender mesh from the reference paper.
    """

    lx, ly, lz = HALL_M
    wall = 0.2
    geometry = [
        _box("factory-floor", (lx, ly, wall), (lx / 2, ly / 2, -wall / 2), "concrete"),
        _box("factory-ceiling", (lx, ly, wall), (lx / 2, ly / 2, lz + wall / 2), "concrete"),
        _box("factory-wall-x0", (wall, ly, lz), (-wall / 2, ly / 2, lz / 2), "concrete"),
        _box("factory-wall-x1", (wall, ly, lz), (lx + wall / 2, ly / 2, lz / 2), "concrete"),
        _box("factory-wall-y0", (lx, wall, lz), (lx / 2, -wall / 2, lz / 2), "concrete"),
        _box("factory-wall-y1", (lx, wall, lz), (lx / 2, ly + wall / 2, lz / 2), "concrete"),
    ]
    for index, center in enumerate(PRODUCTION_LINE_CENTERS_M):
        geometry.append(
            _box(f"production-line-{index:02d}", (4.0, 14.0, 3.0), center, "metal")
        )
    return (
        '<scene version="2.1.0">'
        '<bsdf type="itu-radio-material" id="concrete">'
        '<string name="type" value="concrete"/>'
        '</bsdf>'
        '<bsdf type="itu-radio-material" id="metal">'
        '<string name="type" value="metal"/>'
        '</bsdf>'
        + "".join(geometry)
        + "</scene>"
    )


def sample_receiver_position(seed: int, *, margin_m: float = 1.0) -> np.ndarray:
    """Draw a valid fixed-height receiver point outside the six obstacles."""

    if margin_m < 0 or margin_m >= 8.0:
        raise ValueError("margin_m must be in [0,8)")
    rng = np.random.default_rng(int(seed))
    for _ in range(10_000):
        x = float(rng.uniform(margin_m, HALL_M[0] - margin_m))
        y = float(rng.uniform(margin_m, HALL_M[1] - margin_m))
        inside_line = any(abs(x - cx) < 2.0 + margin_m * 0.1 and abs(y - cy) < 7.0 + margin_m * 0.1
                          for cx, cy, _ in PRODUCTION_LINE_CENTERS_M)
        if not inside_line:
            return np.asarray([x, y, 1.5], dtype=np.float32)
    raise RuntimeError("could not sample a receiver point outside factory obstacles")

def sample_transmitter_positions(
    seed: int, count: int, *, height_m: float = 1.5, margin_m: float = 1.0
) -> np.ndarray:
    """Draw transmitters from free floor space, avoiding metal line meshes."""

    if count < 1 or height_m <= 0 or margin_m < 0 or margin_m >= 8:
        raise ValueError("invalid transmitter placement parameters")
    rng = np.random.default_rng(int(seed))
    positions: list[list[float]] = []
    for _ in range(100_000):
        x = float(rng.uniform(margin_m, HALL_M[0] - margin_m))
        y = float(rng.uniform(margin_m, HALL_M[1] - margin_m))
        inside_line = any(
            abs(x - cx) < 2.0 + margin_m * 0.1
            and abs(y - cy) < 7.0 + margin_m * 0.1
            for cx, cy, _ in PRODUCTION_LINE_CENTERS_M
        )
        if inside_line:
            continue
        if all(np.linalg.norm(np.asarray([x, y]) - np.asarray(p[:2])) >= 1.0 for p in positions):
            positions.append([x, y, float(height_m)])
        if len(positions) == count:
            return np.asarray(positions, dtype=np.float32)
    raise RuntimeError(f"could not place {count} transmitters outside production lines")



def trace_factory_cfr(
    receiver_position_m: Sequence[float],
    frequencies_hz: np.ndarray,
    *,
    transmitter_positions_m: Sequence[Sequence[float]] = DEFAULT_TX_POSITIONS_M,
    seed: int = 0,
    ray_count: int = 100_000,
    reflection_depth: int = 5,
    diffraction: bool = False,
    refraction: bool = False,
    tx_pattern: str = "iso",
    tx_orientations: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Trace CFRs for fixed transmitters to one 16-element mobile UE.

    Returns complex `H[transmitter,rx_element,frequency]` and immutable
    simulation metadata. No fallback propagation model is used.
    """

    position = np.asarray(receiver_position_m, dtype=np.float32).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=np.float32).reshape(-1)
    tx_positions = np.asarray(transmitter_positions_m, dtype=np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("receiver_position_m must be a finite [x,y,z] vector")
    if tx_positions.ndim != 2 or tx_positions.shape[1] != 3 or not len(tx_positions):
        raise ValueError("transmitter_positions_m must have shape [N,3]")
    if not np.isfinite(tx_positions).all():
        raise ValueError("transmitter positions must be finite")
    if tx_pattern not in {"iso", "tr38901"}:
        raise ValueError("tx_pattern must be iso or tr38901")
    if tx_orientations is None:
        tx_orientations = np.zeros((len(tx_positions), 3), dtype=np.float32)
    tx_orientations = np.asarray(tx_orientations, dtype=np.float32)
    if tx_orientations.shape != (len(tx_positions), 3) or not np.isfinite(tx_orientations).all():
        raise ValueError("tx_orientations must be finite [N,3] Euler angles")
    if frequencies.size < 2 or not np.isfinite(frequencies).all():
        raise ValueError("at least two finite CFR frequencies are required")
    if ray_count < 1 or reflection_depth < 0:
        raise ValueError("ray_count must be positive and reflection_depth non-negative")

    try:
        import sionna
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene_from_string
    except ImportError as error:  # pragma: no cover - isolated environment only
        raise RuntimeError("Sionna RT 2.1.x is required for factory-OFDMA generation") from error
    if not str(sionna.__version__).startswith("2.1."):
        raise RuntimeError(f"expected Sionna 2.1.x, found {sionna.__version__}")

    scene = load_scene_from_string(factory_scene_xml(), merge_shapes=True)
    scene.frequency = FC_HZ
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern=tx_pattern, polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=RX_ELEMENTS,
        horizontal_spacing=float(RX_SPACING_M), pattern="iso", polarization="V",
    )
    for index, tx_position in enumerate(tx_positions):
        scene.add(Transmitter(name=f"factory_tx_{index:02d}", position=tx_position.tolist(),
                              orientation=tx_orientations[index].tolist()))
    scene.add(Receiver(name="mobile_ue", position=position.tolist()))
    paths = PathSolver()(
        scene=scene,
        max_depth=int(reflection_depth),
        samples_per_src=int(ray_count),
        max_num_paths_per_src=int(ray_count),
        synthetic_array=False,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=bool(refraction),
        diffraction=bool(diffraction),
        seed=int(seed),
    )
    native = np.asarray(paths.cfr(
        frequencies=frequencies,
        sampling_frequency=float(np.ptp(frequencies)),
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    ))
    # Sionna RT 2.1 `Paths.cfr`: [rx,rx_ant,tx,tx_ant,tx,freq].
    if native.ndim != 6 or native.shape[0] != 1 or native.shape[1] != RX_ELEMENTS or native.shape[3:5] != (1, 1):
        raise ValueError(f"unexpected Sionna multi-transmitter CFR axes: {native.shape}")
    response = native[0, :, :, 0, 0, :].transpose(1, 0, 2)
    expected = (len(tx_positions), RX_ELEMENTS, len(frequencies))
    if response.shape != expected or not np.isfinite(response).all():
        raise ValueError(f"invalid factory CFR shape/values: {response.shape}")
    metadata = {
        "simulator": "Sionna RT",
        "sionna_version": str(sionna.__version__),
        "carrier_frequency_hz": FC_HZ,
        "receiver_position_m": position.astype(float).tolist(),
        "transmitter_positions_m": tx_positions.astype(float).tolist(),
        "factory_hall_m": list(HALL_M),
        "production_line_count": 6,
        "production_line_size_m": [14.0, 4.0, 3.0],
        "production_line_material": "metal",
        "wall_material": "concrete",
        "receiver_array": {
            "geometry": "1x16 ULA",
            "spacing_rule": "half_wavelength",
            "spacing_m": RX_SPACING_M,
            "aperture_m": (RX_ELEMENTS - 1) * RX_SPACING_M,
        },
        "ray_tracing": {
            "transmitter_pattern": tx_pattern,
            "transmitter_orientations": tx_orientations.astype(float).tolist(),
            "max_depth": int(reflection_depth),
            "ray_count_per_source": int(ray_count),
            "refraction": bool(refraction),
            "diffraction": bool(diffraction),
            "seed": int(seed),
        },
    }
    return response.astype(np.complex64, copy=False), metadata
