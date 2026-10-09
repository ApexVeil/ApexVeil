"""Conservative local illumination transfer; not physical relighting."""

import cv2
import numpy as np
from .color import linear, srgb, luma


def adapt_asset(asset, original_rect, reference_percentile=99.5, match_black=True):
    """Transfer broad luminance variation, retaining a neutral black/white design.

    Do not transfer green-plate albedo directly. Neutral bright pixels provide
    an optional tint estimate. This is a heuristic, not recovered physical light.
    """
    src = linear(original_rect)
    a = linear(asset)
    lum = luma(src)
    height, width = lum.shape
    # Large closing fills dark glyph strokes; smoothing removes small hardware.
    k = max(9, int(height * 0.13) | 1)
    dark_face = float(np.median(lum)) < float(np.percentile(lum, 97)) * 0.25
    operation = cv2.MORPH_OPEN if dark_face else cv2.MORPH_CLOSE
    filled = cv2.morphologyEx(
        lum, operation, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    broad = cv2.GaussianBlur(filled, (0, 0), max(3, height * 0.08))
    if dark_face:
        # Existing branded plates have white glyphs on black. Remove glyph
        # structure from the illumination estimate rather than transferring it.
        yy, xx = np.mgrid[-1 : 1 : complex(height), -1 : 1 : complex(width)]
        terms = np.stack([np.ones_like(xx), xx, yy, xx * yy], axis=-1)
        samples = terms[::6, ::6].reshape(-1, 4)
        coefficients = np.linalg.lstsq(samples, broad[::6, ::6].ravel(), rcond=None)[0]
        broad = np.maximum(terms @ coefficients, 0.0001).astype(np.float32)
    median = float(np.median(broad))
    field = np.clip(broad / max(median, 0.001), 0.7, 1.35)
    brightness = float(np.percentile(lum, 97))
    vals = src.reshape(-1, 3)
    ls = luma(vals)
    chroma = (vals.max(1) - vals.min(1)) / np.maximum(vals.max(1), 0.005)
    neutral = vals[(chroma < 0.3) & (ls > np.percentile(ls, 85))]
    tint = np.ones(3, np.float32)
    if len(neutral) > 30:
        t = np.median(neutral, axis=0)
        tint = np.clip(t / max(float(luma(t)), 0.001), 0.7, 1.3)
    ref_white = float(np.percentile(luma(a), reference_percentile))
    black_point = float(np.percentile(lum, 3)) * 0.65 if match_black else 0.0
    gain = float(
        np.clip((brightness - black_point) / max(ref_white, 0.001), 0.008, 1.15)
    )
    out = a * gain * field[:, :, None] * tint + black_point
    return srgb(out), {
        "exposure_gain_linear": gain,
        "black_point_linear": black_point,
        "tint_rgb": tint.tolist(),
        "luminance_p97_linear": float(np.percentile(lum, 97)),
        "light_field_range": [float(field.min()), float(field.max())],
    }
