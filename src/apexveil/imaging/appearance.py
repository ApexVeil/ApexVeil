"""Local illumination and additive-veil transfer, not physical relighting."""

import cv2
import numpy as np
from .color import linear, srgb, luma


def estimate_black_field(src, valid_mask=None):
    """Fit a smooth RGB floor to local dark ink samples.

    A bright license plate contains dark glyphs whose observed color is useful
    evidence for veiling light. A lower-envelope fit discourages green face
    pixels in empty cells from becoming the new material's black point. This is
    a bounded heuristic: unknown ink reflectance and flare cannot be uniquely
    separated from a single photograph.
    """
    h, w = src.shape[:2]
    lum = luma(src)
    valid = np.ones(lum.shape, bool) if valid_mask is None else valid_mask > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(9, int(h * 0.13) | 1),) * 2
    )
    closed = cv2.morphologyEx(lum, cv2.MORPH_CLOSE, kernel)
    ink_support = (closed - lum) > np.maximum(0.00005, closed * 0.12)
    # Dark branded surfaces need no dark-on-light glyph detector.
    if np.median(lum[valid]) < np.percentile(lum[valid], 97) * 0.25:
        ink_support |= lum < np.percentile(lum[valid], 35)
    yy, xx = np.mgrid[-1 : 1 : complex(h), -1 : 1 : complex(w)]
    basis = np.stack([np.ones_like(xx), xx, yy, xx * yy, xx * xx, yy * yy], -1)
    samples, colors, contrast = [], [], []
    for iy in range(3):
        for ix in range(8):
            y0, y1 = round(iy * h / 3), round((iy + 1) * h / 3)
            x0, x1 = round(ix * w / 8), round((ix + 1) * w / 8)
            patch = lum[y0:y1, x0:x1]
            good = valid[y0:y1, x0:x1]
            support = ink_support[y0:y1, x0:x1] & good
            if support.sum() < 5:
                continue
            low = np.percentile(patch[support], 35)
            high = np.percentile(patch[good], 85)
            ink = support & (patch <= low)
            rgb = src[y0:y1, x0:x1][ink]
            colors.append(np.median(rgb, axis=0))
            samples.append(np.median(basis[y0:y1, x0:x1][ink], axis=0))
            contrast.append(max(0.02, float((high - low) / max(high, 1e-5))))
    if len(samples) < 6:
        return np.broadcast_to(np.percentile(src[valid], 3, axis=0), src.shape).copy()
    x, y = np.asarray(samples), np.asarray(colors)
    confidence = np.asarray(contrast)
    coef = np.zeros((6, 3))
    for c in range(3):
        weight = confidence.copy()
        for _ in range(8):
            wx = x * np.sqrt(weight[:, None])
            wy = y[:, c] * np.sqrt(weight)
            # Mild curvature regularization reduces extrapolation at corners.
            reg = np.diag([0, 0.01, 0.01, 0.04, 0.04, 0.04])
            coef[:, c] = np.linalg.lstsq(
                np.vstack([wx, reg]), np.r_[wy, np.zeros(6)], rcond=None
            )[0]
            residual = y[:, c] - x @ coef[:, c]
            scale = max(float(np.median(np.abs(residual))), 1e-5)
            weight = (
                confidence
                * np.where(residual < 0, 0.85, 0.15)
                / np.maximum(1, np.abs(residual) / (scale * 2))
            )
    field = np.maximum(basis @ coef, 0).astype(np.float32)
    # Ink is not perfectly black; conservative removal of its reflectance.
    field *= 0.85
    ceiling = np.percentile(src[valid], 45, axis=0)
    return np.minimum(field, ceiling).astype(np.float32)


def estimate_context_tint(rgb, mask, quad):
    """Estimate a dark-pixel color prior from the immediate surrounding scene.

    This limits transfer of the old plate's green/blue albedo into a black asset.
    The surrounding car/road can still be colored, so this remains a heuristic.
    """
    q = np.asarray(quad, np.float32)
    height = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    pad = max(8, int(height * 0.8))
    x0 = max(0, int(q[:, 0].min()) - pad)
    x1 = min(rgb.shape[1], int(q[:, 0].max()) + pad + 1)
    y0 = max(0, int(q[:, 1].min()) - pad)
    y1 = min(rgb.shape[0], int(q[:, 1].max()) + pad + 1)
    patch = rgb[y0:y1, x0:x1]
    if not patch.size:
        return np.ones(3, np.float32), {"method": "neutral_fallback", "samples": 0}
    exclude = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    cv2.fillConvexPoly(exclude, np.rint(q - [x0, y0]).astype(np.int32), 1)
    exclude = cv2.dilate(exclude, np.ones((5, 5), np.uint8))
    samples = linear(cv2.GaussianBlur(patch, (0, 0), 1.0))[exclude == 0]
    if len(samples) < 32:
        return np.ones(3, np.float32), {
            "method": "neutral_fallback",
            "samples": len(samples),
        }
    light = luma(samples)
    lo, hi = np.percentile(light, [8, 25])
    selected = samples[(light >= lo) & (light <= hi)]
    color = np.median(selected, axis=0)
    if float(luma(color)) < 0.0003:
        tint = np.ones(3, np.float32)
    else:
        tint = np.clip(color / max(float(luma(color)), 1e-6), 0.05, 4)
        tint = tint / max(float(luma(tint)), 1e-6)
    return tint.astype(np.float32), {
        "method": "surrounding_dark_pixels",
        "samples": len(selected),
        "tint_rgb": tint.tolist(),
    }


def adapt_asset(
    asset,
    original_rect,
    reference_percentile=99.5,
    match_black=True,
    valid_mask=None,
    context_tint=None,
    ink_resolved=True,
):
    """Transfer broad exposure, shadow structure and an RGB additive floor.

    Green face color and old lettering are not directly copied into the asset.
    Matte/retroreflective materials still differ; this is not recovered lighting.
    """
    src, a = linear(original_rect), linear(asset)
    lum = luma(src)
    height, width = lum.shape
    valid = np.ones(lum.shape, bool) if valid_mask is None else valid_mask > 0
    if not valid.any():
        raise ValueError("Appearance requires visible reference pixels")
    k = max(9, int(height * 0.13) | 1)
    dark_face = (
        float(np.median(lum[valid])) < float(np.percentile(lum[valid], 97)) * 0.25
    )
    operation = cv2.MORPH_OPEN if dark_face else cv2.MORPH_CLOSE
    filled = cv2.morphologyEx(
        lum, operation, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    if dark_face:
        yy, xx = np.mgrid[-1 : 1 : complex(height), -1 : 1 : complex(width)]
        terms = np.stack([np.ones_like(xx), xx, yy, xx * yy], axis=-1)
        coefficients = np.linalg.lstsq(
            terms[::6, ::6].reshape(-1, 4), filled[::6, ::6].ravel(), rcond=None
        )[0]
        broad = np.maximum(terms @ coefficients, 0.000001).astype(np.float32)
    else:
        # A small edge-preserving smoother retains broad shadow boundaries.
        sigma = max(1.0, height * 0.025)
        broad = cv2.bilateralFilter(
            filled, 0, max(0.0002, float(np.percentile(filled, 80)) * 0.2), sigma
        )
    reference_coverage = float(valid.mean())
    color_floor_reliable = reference_coverage >= 0.65 and ink_resolved
    if not match_black:
        black = np.zeros_like(src)
    elif color_floor_reliable:
        black = estimate_black_field(src, valid)
    else:
        # Too little visible ink constrains a spatial color fit. A neutral
        # lower bound is safer than extrapolating the green face into a black asset.
        floor = float(np.percentile(lum[valid], 3)) * 0.65
        black = np.full_like(src, floor)
    if context_tint is not None:
        black = luma(black)[:, :, None] * np.asarray(context_tint, np.float32)
    # Illumination is estimated after removing the additive veil.
    diffuse = np.maximum(broad - luma(black), 1e-6)
    field = np.clip(diffuse / max(float(np.median(diffuse[valid])), 1e-6), 0.25, 2.0)
    brightness = float(np.percentile(lum[valid], 97))
    vals = src[valid]
    ls = luma(vals)
    chroma = (vals.max(1) - vals.min(1)) / np.maximum(vals.max(1), 0.005)
    neutral = vals[(chroma < 0.3) & (ls > np.percentile(ls, 85))]
    tint = np.ones(3, np.float32)
    if len(neutral) > 30:
        t = np.median(neutral, axis=0)
        tint = np.clip(t / max(float(luma(t)), 1e-6), 0.7, 1.3)
    ref_white = float(np.percentile(luma(a), reference_percentile))
    black_point = float(np.median(luma(black)[valid]))
    # A dark image should remain dark. No artificial minimum white intensity.
    gain = float(np.clip((brightness - black_point) / max(ref_white, 1e-6), 0.0, 1.15))
    out = a * gain * field[:, :, None] * tint + black
    return srgb(out), {
        "exposure_gain_linear": gain,
        "black_point_linear": black_point,
        "black_field_rgb_p50_linear": np.median(black.reshape(-1, 3), axis=0).tolist(),
        "black_field_method": "local_dark_ink_lower_envelope_rgb"
        if color_floor_reliable
        else "uncertain_ink_neutral_fallback",
        "reference_visible_fraction": reference_coverage,
        "black_chroma_source": "scene_context"
        if context_tint is not None
        else "plate_ink",
        "ink_resolved": ink_resolved,
        "color_floor_reliable": color_floor_reliable,
        "tint_rgb": tint.tolist(),
        "luminance_p97_linear": brightness,
        "light_field_range": [float(field.min()), float(field.max())],
        "illumination_model": "edge_preserving_diffuse_plus_smooth_additive_veil",
        "physically_calibrated": False,
    }
