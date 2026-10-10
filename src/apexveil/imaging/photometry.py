"""Temporal material parameters, separate from current-frame spatial shadows."""

import math

import numpy as np


class TemporalPhotometry:
    """Damp fitted gain/color spikes while responding to observed light changes.

    Exposure is filtered in log space. A source-scene change requires agreement
    between robust middle and upper luminance percentiles, rather than a glint
    at the brightest tail. The spatial illumination field is never averaged by
    this class, so moving shadow geometry remains the current frame's estimate.
    """

    def __init__(self, fps=30.0):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.dt = 1.0 / fps
        self.reset()

    def reset(self):
        self.log_gain = None
        self.black = None
        self.black_shape = None
        self.tint = None
        self.scene = None
        self.fast_seconds = 0.0

    def filter(self, gain, black, tint, scene_levels):
        gain = float(gain)
        black = np.asarray(black, np.float32)
        tint = np.asarray(tint, np.float32)
        scene = np.maximum(np.asarray(scene_levels, np.float32), 1e-6)
        if (
            not np.isfinite(gain)
            or gain < 0
            or not all(np.isfinite(x).all() for x in (black, tint, scene))
        ):
            raise ValueError("Non-finite or negative temporal appearance parameters")
        target = math.log(max(gain, 1e-6))
        raw_level = np.median(black.reshape(-1, 3), axis=0)
        raw_black = raw_level.tolist()
        current_residual = black - raw_level
        if self.black_shape is not None and self.black_shape != black.shape:
            self.reset()
        scene_step = np.zeros(2, np.float32)
        if self.scene is not None:
            scene_step = np.log(scene / self.scene)
        observed_light_change = bool(
            np.all(scene_step > 0.10) or np.all(scene_step < -0.10)
        )
        if observed_light_change:
            self.fast_seconds = 0.10
        fast = self.fast_seconds > 0
        weight = 1 - math.exp(-self.dt / (0.022 if fast else 0.12))
        if self.log_gain is None:
            self.log_gain = target
            self.black = raw_level.copy()
            self.black_shape = black.shape
            self.tint = tint.copy()
            applied_step = 0.0
        else:
            # Bound unsupported fitting corrections, not actual source motion.
            applied_step = (target - self.log_gain) * weight
            if not fast:
                applied_step = float(np.clip(applied_step, -2 * self.dt, 2 * self.dt))
            self.log_gain += applied_step
            self.black += (raw_level - self.black) * weight
            self.tint += (tint - self.tint) * weight
        self.scene = scene.copy()
        self.fast_seconds = max(0.0, self.fast_seconds - self.dt)
        return (
            float(math.exp(self.log_gain)),
            np.maximum(current_residual + self.black, 0).astype(np.float32),
            self.tint.copy(),
            {
                "method": "scene_gated_log_gain_and_global_rgb_veil_level",
                "filtered_veil_rgb_level_linear": self.black.tolist(),
                "spatial_veil_residual_filtered": False,
                "raw_exposure_gain_linear": gain,
                "raw_black_rgb_p50_linear": raw_black,
                "observed_scene_log_change": scene_step.tolist(),
                "source_scene_levels": scene.tolist(),
                "fast_light_response": fast,
                "parameter_weight_current": weight,
                "applied_gain_log_step": float(applied_step),
                "spatial_shadow_field_filtered": False,
            },
        )
