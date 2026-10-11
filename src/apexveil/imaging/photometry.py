"""Temporal material parameters, separate from current-frame spatial shadows."""

import math
from dataclasses import replace

import cv2
import numpy as np

from .color import linear, luma


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


class TemporalVideoAppearance:
    """Stabilize uncertain capture/material fits using current source evidence.

    Capture softness is filtered *before* it determines whether source ink is
    resolved. Light changes are measured on the same visible canonical samples,
    so a crop changing the population of bright/dark pixels is not exposure.
    Unsupported gain and veil changes have bounded log rates; observed lighting
    changes are applied immediately. Only RGB veil amplitudes are filtered: the
    current veil locations and multiplicative shadows retain their current shape.

    Use one instance per continuous target shot, and reset on a shot/identity
    change. Optional frame indices make gaps use elapsed time rather than one
    nominal frame. This is a conservative temporal heuristic, not relighting or
    recovery of a directional motion-blur kernel.
    """

    def __init__(self, fps=30.0):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.dt = 1.0 / float(fps)
        self.reset()

    def reset(self):
        self._sigma = None
        self._noise = None
        self._capture_index = None
        self._reset_appearance()

    def _reset_appearance(self):
        self._shape = None
        self._gain = None
        self._veil = None
        self._tint = None
        self._source = None
        self._source_valid = None
        self._scene = None
        self._visible_fraction = None
        self._appearance_index = None

    def _elapsed(self, frame_index, previous_index):
        if frame_index is None:
            return self.dt
        if not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if previous_index is not None and frame_index <= previous_index:
            raise ValueError("Temporal estimates require increasing frame indices")
        return self.dt * (
            frame_index - previous_index if previous_index is not None else 1
        )

    def filter_degradation(self, estimate, *, visible_fraction=1.0, frame_index=None):
        """Return a capture estimate, retaining history when edge evidence fails.

        ``visible_fraction`` is the fraction of the inferred plane visible in
        this image. Reflected samples beyond a viewport edge must not suddenly
        make a partially visible surface appear sharply focused.
        """
        if not np.isfinite(visible_fraction) or not 0 <= visible_fraction <= 1:
            raise ValueError("visible_fraction must be finite within 0..1")
        dt = self._elapsed(frame_index, self._capture_index)
        raw_sigma = float(estimate.sigma_px)
        raw_noise = np.asarray(
            [estimate.noise_intercept, estimate.noise_slope], np.float64
        )
        if (
            not np.isfinite(raw_sigma)
            or not 0 <= raw_sigma <= 20
            or (not np.isfinite(raw_noise).all() or np.any(raw_noise < 0))
        ):
            raise ValueError("Invalid capture degradation parameters")
        focus = estimate.diagnostics.get("focus", {})
        reliable = bool(focus.get("reliable", True))
        # Reliability is evidence about estimation, not a request to sharpen a
        # previously blurry object using a numerical fallback such as 0.35 px.
        confidence = math.sqrt(float(visible_fraction)) if reliable else 0.0
        weight = (1 - math.exp(-dt / 0.10)) * confidence
        if self._sigma is None:
            self._sigma = raw_sigma
            self._noise = raw_noise.copy()
        else:
            # The offset permits a true zero-sigma estimate without log(0).
            current = math.log(self._sigma + 0.25)
            step = (math.log(raw_sigma + 0.25) - current) * weight
            self._sigma = max(
                0.0, math.exp(current + np.clip(step, -6 * dt, 6 * dt)) - 0.25
            )
            if estimate.diagnostics.get("noise_reliable", True):
                self._noise += (raw_noise - self._noise) * (1 - math.exp(-dt / 0.12))
        self._capture_index = frame_index
        diagnostics = dict(estimate.diagnostics)
        diagnostics["focus"] = {**focus, "sigma_px": float(self._sigma)}
        diagnostics["noise_intercept_code_variance"] = float(self._noise[0])
        diagnostics["noise_slope_code_variance"] = float(self._noise[1])
        diagnostics["temporal_capture"] = {
            "method": "confidence_weighted_bounded_log_softness",
            "raw_sigma_px": raw_sigma,
            "filtered_sigma_px": float(self._sigma),
            "source_focus_reliable": reliable,
            "source_visible_fraction": float(visible_fraction),
            "parameter_weight_current": weight,
            "unsupported_log_rate_per_second": 6.0,
            "elapsed_seconds": dt,
        }
        return replace(
            estimate,
            sigma_px=float(self._sigma),
            noise_intercept=float(self._noise[0]),
            noise_slope=float(self._noise[1]),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _reference_samples(reference, valid):
        if reference.shape != (*valid.shape, 3) or reference.dtype != np.uint8:
            raise ValueError("source_reference must be canonical RGB uint8")
        h, w = valid.shape
        size = (min(w, 96), min(h, 24))
        support = cv2.resize(
            valid.astype(np.float32), size, interpolation=cv2.INTER_AREA
        )
        # Mask-aware downsampling excludes inpainted or off-image pixels even
        # where only part of a low-resolution sample is visible.
        samples = cv2.resize(
            linear(reference) * valid[:, :, None], size, interpolation=cv2.INTER_AREA
        ) / np.maximum(support[:, :, None], 1e-6)
        return samples, support >= 0.8

    def _light_change(self, source, source_valid, scene, visible_fraction):
        light_step, veil_step = 0.0, np.zeros(3, np.float64)
        quantiles = np.zeros(2, np.float64)
        overlap_fraction, overlap_count = 0.0, 0
        method = "initial_observation"
        if source is not None and self._source is not None:
            overlap = source_valid & self._source_valid
            overlap_count = int(overlap.sum())
            overlap_fraction = overlap_count / max(1, int(source_valid.sum()))
            method = "insufficient_matching_source_support"
            if overlap_count >= 16 and overlap_fraction >= 0.5:
                before, after = self._source[overlap], source[overlap]
                old_luma, new_luma = luma(before), luma(after)
                # A small floor avoids interpreting quantization near zero as
                # unbounded exposure changes; four stops per pair is the limit.
                changes = np.log((new_luma + 0.0003) / (old_luma + 0.0003))
                quantiles = np.percentile(changes, [20, 80])
                light_step = float(
                    np.clip(np.median(changes), -math.log(16), math.log(16))
                )
                # Veil evidence follows matching dark source samples, not the
                # old plate's full-face color or a changing crop percentile.
                # Symmetric selection avoids regression-to-the-mean bias from
                # repeatedly selecting only the previous frame's darkest pixels.
                middle_luma = (old_luma + new_luma) * 0.5
                dark = middle_luma <= np.percentile(middle_luma, 25)
                if int(dark.sum()) >= 8:
                    veil_step = np.clip(
                        np.median(
                            np.log((after[dark] + 0.0003) / (before[dark] + 0.0003)),
                            axis=0,
                        ),
                        -math.log(4),
                        math.log(4),
                    )
                else:
                    veil_step[:] = light_step
                method = "matching_visible_canonical_source_samples"
        elif source is None and self._scene is not None:
            method = "crop_guarded_source_percentiles"
            if abs(visible_fraction - self._visible_fraction) <= 0.05:
                quantiles = np.log(scene / self._scene)
                if np.all(quantiles >= 0) or np.all(quantiles <= 0):
                    light_step = float(
                        np.clip(np.median(quantiles), -math.log(16), math.log(16))
                    )
                    veil_step[:] = light_step
        fast = bool(np.all(quantiles > 0.10) or np.all(quantiles < -0.10))
        return (
            light_step,
            veil_step,
            {
                "light_evidence_method": method,
                "observed_scene_log_change": quantiles.tolist(),
                "source_overlap_fraction": overlap_fraction,
                "source_overlap_samples": overlap_count,
                "source_supported_gain_log_step": light_step,
                "source_supported_veil_rgb_log_step": veil_step.tolist(),
                "fast_light_response": fast,
            },
        )

    def filter_appearance(
        self,
        estimate,
        source_levels,
        *,
        valid_mask=None,
        source_reference=None,
        context_black_level=None,
        frame_index=None,
    ):
        """Filter the full material estimate after any explicit scene guidance.

        Pass the *current* canonical source crop and its visible mask for crop-
        invariant light evidence. The context level is recorded for provenance;
        surrounding car/road changes alone cannot prove a plate's veil changed.
        """
        black = np.asarray(estimate.black, np.float32)
        tint = np.asarray(estimate.tint, np.float32)
        field = np.asarray(estimate.spatial_field)
        gain = float(estimate.gain)
        scene = np.maximum(np.asarray(source_levels, np.float64), 1e-6)
        if (
            black.ndim != 3
            or black.shape[2] != 3
            or tint.shape != (3,)
            or (field.shape != black.shape[:2] or scene.shape != (2,))
        ):
            raise ValueError("Invalid temporal appearance parameter shapes")
        if not all(np.isfinite(x).all() for x in (gain, black, tint, field, scene)) or (
            gain < 0 or np.any(black < 0) or np.any(tint < 0) or np.any(field < 0)
        ):
            raise ValueError(
                "Temporal appearance parameters must be finite and nonnegative"
            )
        if context_black_level is not None and (
            not np.isfinite(context_black_level) or context_black_level < 0
        ):
            raise ValueError("context_black_level must be finite and nonnegative")
        valid = (
            np.ones(black.shape[:2], bool)
            if valid_mask is None
            else np.asarray(valid_mask) > 0
        )
        if valid.shape != black.shape[:2] or not valid.any():
            raise ValueError(
                "Temporal appearance requires matching visible reference pixels"
            )
        if self._shape is not None and self._shape != black.shape:
            self._reset_appearance()
        dt = self._elapsed(frame_index, self._appearance_index)
        source, source_valid = (
            (None, None)
            if source_reference is None
            else self._reference_samples(source_reference, valid)
        )
        coverage = float(valid.mean())
        light_step, veil_step, evidence = self._light_change(
            source, source_valid, scene, coverage
        )
        # A full envelope sees spatial glare peaks without retaining their old
        # positions. Unlike median-only filtering, it also bounds fitted slopes
        # and curvature that can brighten an entire entering/exiting edge.
        raw_veil = black[valid].max(axis=0)
        raw_level = np.median(black[valid], axis=0)
        weight = 1 - math.exp(
            -dt / (0.022 if evidence["fast_light_response"] else 0.12)
        )
        target_gain = math.log(max(gain, 1e-6))
        offset = 0.0003
        correction = 0.0
        supported_gain = 0.0
        supported_veil = np.zeros(3)
        if self._gain is None:
            self._gain = target_gain
            self._veil = raw_veil.copy()
            self._tint = tint.copy()
        else:
            # Evidence can accelerate movement toward the current observation,
            # never extrapolate away from it. This also prevents small matching
            # biases or low-signal RGB ratios from accumulating without bound.
            gain_delta = target_gain - self._gain
            supported_gain = float(
                np.clip(light_step, min(0, gain_delta), max(0, gain_delta))
            )
            predicted = self._gain + supported_gain
            correction = float(
                np.clip((target_gain - predicted) * weight, -2 * dt, 2 * dt)
            )
            self._gain = predicted + correction
            current_veil = np.log(self._veil + offset)
            target_veil = np.log(raw_veil + offset)
            veil_delta = target_veil - current_veil
            supported_veil = np.clip(
                veil_step, np.minimum(0, veil_delta), np.maximum(0, veil_delta)
            )
            predicted_veil = current_veil + supported_veil
            veil_correction = np.clip(
                (target_veil - predicted_veil) * weight,
                -2.5 * dt,
                2.5 * dt,
            )
            self._veil = np.maximum(
                np.exp(predicted_veil + veil_correction) - offset, 0
            )
            log_tint = np.log(self._tint + 1e-6)
            self._tint = np.maximum(
                np.exp(
                    log_tint
                    + np.clip(
                        (np.log(tint + 1e-6) - log_tint) * weight, -1.5 * dt, 1.5 * dt
                    )
                )
                - 1e-6,
                0,
            ).astype(np.float32)
        # Current shape only: no past shadow or reflection field is added.
        veil = black * np.divide(
            self._veil, raw_veil, out=np.ones(3), where=raw_veil > 1e-9
        )
        applied_gain = float(math.exp(self._gain)) if gain > 0 else 0.0
        veil = np.maximum(veil, 0).astype(np.float32)
        self._veil = veil[valid].max(axis=0)
        self._source, self._source_valid = source, source_valid
        self._scene, self._visible_fraction = scene.copy(), coverage
        self._shape, self._appearance_index = black.shape, frame_index
        diagnostics = dict(estimate.diagnostics)
        diagnostics.update(
            exposure_gain_linear=applied_gain,
            black_point_linear=float(np.median(luma(veil)[valid])),
            black_field_rgb_p50_linear=np.median(veil[valid], axis=0).tolist(),
            tint_rgb=self._tint.tolist(),
            temporal_parameters={
                "method": "matching_source_light_and_bounded_current_veil_envelope",
                **evidence,
                "raw_exposure_gain_linear": gain,
                "raw_black_rgb_p50_linear": raw_level.tolist(),
                "raw_veil_rgb_peak_linear": raw_veil.tolist(),
                "filtered_veil_rgb_peak_linear": self._veil.tolist(),
                "raw_veil_rgb_p99_linear": np.percentile(
                    black[valid], 99, axis=0
                ).tolist(),
                "filtered_veil_rgb_p99_linear": np.percentile(
                    veil[valid], 99, axis=0
                ).tolist(),
                "filtered_veil_rgb_level_linear": np.median(
                    veil[valid], axis=0
                ).tolist(),
                "source_scene_levels": scene.tolist(),
                "source_visible_fraction": coverage,
                "context_black_level_linear": context_black_level,
                "parameter_weight_current": weight,
                "unsupported_gain_log_correction": correction,
                "applied_source_gain_log_step": supported_gain,
                "applied_source_veil_rgb_log_step": supported_veil.tolist(),
                "unsupported_gain_log_rate_per_second": 2.0,
                "unsupported_veil_log_rate_per_second": 2.5,
                "spatial_veil_residual_filtered": False,
                "spatial_veil_envelope_bounded": True,
                "spatial_shadow_field_filtered": False,
                "elapsed_seconds": dt,
            },
        )
        return replace(
            estimate,
            gain=applied_gain,
            black=veil,
            tint=self._tint.copy(),
            diagnostics=diagnostics,
        )
