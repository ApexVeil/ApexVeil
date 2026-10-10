"""Source-derived stochastic capture degradation with explicit confidence.

Noise is estimated from flat patches, after rejecting glyph/edge structure.
A linear variance versus encoded signal approximation is not a camera model.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .focus import estimate_edge_softness


@dataclass
class DegradationEstimate:
    sigma_px: float
    noise_intercept: float
    noise_slope: float
    diagnostics: dict

    def __post_init__(self):
        values = [self.sigma_px, self.noise_intercept, self.noise_slope]
        if not np.isfinite(values).all() or min(values) < 0 or self.sigma_px > 20:
            raise ValueError(
                "Capture degradation must be finite, nonnegative and sigma <= 20"
            )


def estimate_degradation(rgb, mask, quad):
    focus = estimate_edge_softness(rgb, quad)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Visible mask is empty")
    q = np.asarray(quad)
    short = min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    pad = max(12, round(short * 0.5))
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = (
        min(rgb.shape[1], int(xs.max()) + pad + 1),
        min(rgb.shape[0], int(ys.max()) + pad + 1),
    )
    patch = rgb[y0:y1, x0:x1].astype(np.float32)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1)
    gx, gy = np.gradient(smooth)
    gradient = np.hypot(gx, gy)
    # A high-pass kernel has norm 6; robust MAD ignores outlying JPEG edges.
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    residual = cv2.filter2D(gray, -1, kernel) / 6
    levels, variances = [], []
    cell = max(8, min(24, round(short * 0.15)))
    for y in range(2, gray.shape[0] - cell - 2, cell):
        for x in range(2, gray.shape[1] - cell - 2, cell):
            region = np.s_[y : y + cell, x : x + cell]
            g = gradient[region]
            if np.percentile(g, 80) > max(2, float(np.percentile(gradient, 35))):
                continue
            signal = gray[region]
            if np.percentile(signal, 10) <= 3 or np.percentile(signal, 90) >= 252:
                continue
            r = residual[region]
            sigma = float(np.median(np.abs(r - np.median(r))) / 0.67449)
            levels.append(float(np.median(signal) / 255))
            variances.append(sigma**2)
    reliable = len(levels) >= 8
    intercept, slope = 0.0, 0.0
    if reliable:
        lv, vv = np.asarray(levels), np.asarray(variances)
        # Robust winsorization bounds texture leakage; uncertainty is recorded.
        vv = np.minimum(vv, np.percentile(vv, 75))
        slope = max(0.0, float(np.polyfit(lv, vv, 1)[0])) if np.ptp(lv) > 0.15 else 0.0
        intercept = max(0.0, float(np.median(vv - slope * lv)))
        # Suppress meaningless sub-quantization noise, cap gross texture errors.
        if intercept + slope * np.median(lv) < 0.35**2:
            intercept, slope = 0.0, 0.0
        bound = max(1.0, float(np.percentile(vv, 75)))
        intercept = min(intercept, bound, 36.0)
        slope = min(slope, bound * 2, 64.0)
    return DegradationEstimate(
        float(focus["sigma_px"]),
        intercept,
        slope,
        {
            "focus": focus,
            "noise_method": "flat_patch_mad_signal_variance",
            "noise_reliable": reliable,
            "noise_samples": len(levels),
            "noise_intercept_code_variance": intercept,
            "noise_slope_code_variance": slope,
            "noise_sigma_upper_bound_code": 10.0,
            "physically_calibrated": False,
        },
    )


def apply_capture_noise(rgb, estimate, seed=0):
    if estimate.noise_intercept + estimate.noise_slope <= 0:
        return rgb
    signal = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY) / 255
    sigma = np.sqrt(
        np.clip(estimate.noise_intercept + estimate.noise_slope * signal, 0, 100)
    )
    rng = np.random.default_rng(seed)
    # Shared luminance noise avoids inventing independent RGB colored speckle.
    noise = rng.standard_normal(signal.shape).astype(np.float32) * sigma
    return np.clip(np.rint(rgb.astype(np.float32) + noise[:, :, None]), 0, 255).astype(
        np.uint8
    )
