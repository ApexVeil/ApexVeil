"""Estimate isotropic softness from edge spread, with explicit uncertainty.

This is a still-image heuristic. It does not recover a directional motion kernel
or separate defocus from translucent foreground edges.
"""

import cv2
import numpy as np


def estimate_edge_softness(rgb, quad):
    q = np.asarray(quad, np.float32)
    short = min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    band = min(70.0, float(short) * 0.35)
    if band < 8:
        return {
            "sigma_px": 0.35,
            "samples": 0,
            "reliable": False,
            "method": "small_surface_fallback",
        }
    x0 = max(0, int(q[:, 0].min() - band - 12))
    y0 = max(0, int(q[:, 1].min() - band - 12))
    x1 = min(rgb.shape[1], int(q[:, 0].max() + band + 13))
    y1 = min(rgb.shape[0], int(q[:, 1].max() + band + 13))
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    if not gray.size:
        return {
            "sigma_px": 0.35,
            "samples": 0,
            "reliable": False,
            "method": "outside_image_fallback",
        }
    # Suppress JPEG/sensor noise before measuring the edge-spread function.
    pre_sigma = 0.7
    gray = cv2.GaussianBlur(gray, (0, 0), pre_sigma)
    offsets = np.arange(-band, band, 0.5)
    estimates = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        for t in np.linspace(0.2, 0.8, 12):
            points = a + (b - a) * t + offsets[:, None] * normal
            values = cv2.remap(
                gray,
                (points[:, 0] - x0).astype(np.float32).reshape(1, -1),
                (points[:, 1] - y0).astype(np.float32).reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )[0]
            gradient = np.abs(np.gradient(values, 0.5))
            middle = np.where(np.abs(offsets) < band * 0.65)[0]
            peak = middle[np.argmax(gradient[middle])]
            if (
                gradient[peak] < 0.003
                or np.percentile(values, 90) - np.percentile(values, 10) < 0.035
            ):
                continue
            left = right = peak
            while left > 0 and gradient[left - 1] > gradient[peak] * 0.5:
                left -= 1
            while (
                right < len(offsets) - 1 and gradient[right + 1] > gradient[peak] * 0.5
            ):
                right += 1
            if left == 0 or right == len(offsets) - 1:
                continue
            measured = ((right - left + 1) * 0.5) / 2.355
            estimates.append(np.sqrt(max(0.0, measured**2 - pre_sigma**2)))
    if len(estimates) < 12:
        return {
            "sigma_px": 0.35,
            "samples": len(estimates),
            "reliable": False,
            "method": "insufficient_edge_contrast_fallback",
        }
    sigma = float(np.clip(np.percentile(estimates, 60), 0.35, 20.0))
    return {
        "sigma_px": sigma,
        "samples": len(estimates),
        "reliable": True,
        "spread_p20_p80": np.percentile(estimates, [20, 80]).tolist(),
        "method": "robust_low_presmoothing_edge_spread",
        "calibrated": False,
    }
