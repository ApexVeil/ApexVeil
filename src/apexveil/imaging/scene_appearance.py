"""Bounded lighting guidance; artwork pixels always belong to the renderer."""

from dataclasses import asdict, dataclass

import numpy as np

from .appearance import AppearanceEstimate
from .color import luma


@dataclass
class SceneAppearanceGuidance:
    exposure_gain_linear: float | None = None
    additive_rgb_linear: list[float] | None = None
    flare_control_points: list[dict] | None = None
    shading_control_points: list[dict] | None = None
    diffuse_tint_rgb: list[float] | None = None
    provenance: str = "external_guidance"

    @classmethod
    def from_value(cls, value):
        return value if isinstance(value, cls) else cls(**value)


def _field(points, shape, key, channels):
    if not points or len(points) > 16:
        raise ValueError("Lighting guidance requires 1..16 control points")
    coordinates = np.asarray([p["xy"] for p in points], np.float32)
    values = np.asarray([p[key] for p in points], np.float32).reshape(
        len(points), channels
    )
    if (
        coordinates.shape != (len(points), 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Lighting control points must be finite canonical xy/value pairs"
        )
    if np.any((coordinates < 0) | (coordinates > 1)) or np.any(values < 0):
        raise ValueError(
            "Lighting control points require normalized xy and nonnegative values"
        )
    h, w = shape
    yy, xx = np.mgrid[0 : 1 : complex(h), 0 : 1 : complex(w)]
    grid = np.stack([xx, yy], -1).astype(np.float32)
    if len(points) == 1:
        return np.broadcast_to(values[0], (h, w, channels)).copy()
    if len(points) == 2:
        direction = coordinates[1] - coordinates[0]
        t = np.clip(
            np.sum((grid - coordinates[0]) * direction, -1)
            / max(float(direction @ direction), 1e-8),
            0,
            1,
        )
        return values[0] + t[:, :, None] * (values[1] - values[0])
    distance = np.sum((grid[:, :, None, :] - coordinates[None, None, :, :]) ** 2, -1)
    weights = 1 / np.maximum(distance, 1e-6)
    weights /= weights.sum(-1, keepdims=True)
    return np.sum(weights[:, :, :, None] * values[None, None, :, :], axis=2)


def apply_scene_guidance(estimate, guidance):
    if guidance is None:
        return estimate
    guidance = SceneAppearanceGuidance.from_value(guidance)
    gain = (
        estimate.gain
        if guidance.exposure_gain_linear is None
        else float(guidance.exposure_gain_linear)
    )
    if not np.isfinite(gain) or not 0 <= gain <= 4:
        raise ValueError("Guided exposure gain must be finite within 0..4")
    veil = estimate.black.copy()
    if guidance.additive_rgb_linear is not None:
        value = np.asarray(guidance.additive_rgb_linear, np.float32)
        if (
            value.shape != (3,)
            or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1))
        ):
            raise ValueError(
                "additive_rgb_linear must be 3 finite linear values within0..1"
            )
        veil[:] = value
    if guidance.flare_control_points is not None:
        veil = _field(guidance.flare_control_points, veil.shape[:2], "rgb_linear", 3)
        if veil.max() > 1:
            raise ValueError("Guided flare must lie within0..1 linear RGB")
    shading = estimate.spatial_field.copy()
    if guidance.shading_control_points is not None:
        shading = _field(guidance.shading_control_points, shading.shape, "value", 1)[
            :, :, 0
        ]
        if shading.max() > 4:
            raise ValueError("Guided shading must lie within0..4")
    tint = (
        estimate.tint.copy()
        if guidance.diffuse_tint_rgb is None
        else np.asarray(guidance.diffuse_tint_rgb, np.float32)
    )
    if (
        tint.shape != (3,)
        or not np.isfinite(tint).all()
        or np.any((tint < 0) | (tint > 4))
    ):
        raise ValueError("Guided diffuse tint must be 3 finite values within0..4")
    diagnostics = dict(estimate.diagnostics)
    diagnostics["scene_guidance"] = asdict(guidance)
    diagnostics["exposure_gain_linear"] = gain
    diagnostics["black_point_linear"] = float(np.median(luma(veil)))
    diagnostics["black_field_rgb_p50_linear"] = np.median(
        veil.reshape(-1, 3), axis=0
    ).tolist()
    diagnostics["tint_rgb"] = tint.tolist()
    diagnostics["light_field_range"] = [float(shading.min()), float(shading.max())]
    diagnostics["illumination_model"] = (
        "guided_spatial_rgb_veil_and_multiplicative_shading"
    )
    diagnostics["additive_veil_rgb_p50_linear"] = np.median(
        veil.reshape(-1, 3), axis=0
    ).tolist()
    return AppearanceEstimate(gain, veil, tint, shading, diagnostics)
