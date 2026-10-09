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
        raw_black = np.median(black.reshape(-1, 3), axis=0).tolist()
        if self.black is not None and self.black.shape != black.shape:
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
            self.black = black.copy()
            self.tint = tint.copy()
            applied_step = 0.0
        else:
            # Bound unsupported fitting corrections, not actual source motion.
            applied_step = (target - self.log_gain) * weight
            if not fast:
                applied_step = float(np.clip(applied_step, -2 * self.dt, 2 * self.dt))
            self.log_gain += applied_step
            self.black += (black - self.black) * weight
            self.tint += (tint - self.tint) * weight
        self.scene = scene.copy()
        self.fast_seconds = max(0.0, self.fast_seconds - self.dt)
        return (
            float(math.exp(self.log_gain)),
            self.black.copy(),
            self.tint.copy(),
            {
                "method": "scene_gated_log_gain_and_material_color",
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
