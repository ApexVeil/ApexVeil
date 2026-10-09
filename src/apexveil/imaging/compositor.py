"""Visibility-aware replacement and redaction, shared by still/video frames.

All edits are restricted to the supplied visible mask. The inferred full plane
never doubles as a compositing mask; holes may represent foreground occluders.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import cv2
import numpy as np
from PIL import Image
from .color import linear, srgb
from .appearance import adapt_asset


@dataclass
class Template:
    rgb: np.ndarray
    alpha: np.ndarray
    metadata: dict


def load_template(path: Path, opaque_threshold: int = 248) -> Template:
    """Crop to the supplied silhouette, preserving its aspect and alpha edge."""
    if not 1 <= opaque_threshold <= 255:
        raise ValueError("opaque_threshold must be between 1 and 255")
    rgba = np.array(Image.open(path).convert("RGBA"))
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 127)
    if not len(xs):
        raise ValueError("Template has no opaque foreground")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    rgb = rgba[y0:y1, x0:x1, :3]
    a = np.clip(alpha[y0:y1, x0:x1].astype(np.float32) / opaque_threshold, 0.0, 1.0)
    # Canonical width is bounded, but the artwork's aspect ratio is not changed.
    width = min(1200, x1 - x0)
    height = max(2, round((y1 - y0) * width / (x1 - x0)))
    pm = cv2.resize(
        linear(rgb) * a[:, :, None], (width, height), interpolation=cv2.INTER_AREA
    )
    a = cv2.resize(a, (width, height), interpolation=cv2.INTER_AREA)
    prepared = srgb(pm / np.maximum(a[:, :, None], 1e-6))
    return Template(
        prepared,
        a,
        {
            "crop_xyxy": [x0, y0, x1, y1],
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "opaque_threshold": opaque_threshold,
            "canonical_size": [width, height],
        },
    )


def local_bounds(mask, padding=0):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Visible mask is empty")
    h, w = mask.shape
    return [
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(w, int(xs.max()) + 1 + padding),
        min(h, int(ys.max()) + 1 + padding),
    ]


def replace_surface(rgb, mask, quad, template: Template, blur_sigma=0.35):
    """Insert a material, retaining visible-mask holes and rounded artwork."""
    if blur_sigma < 0 or blur_sigma > 20:
        raise ValueError("blur_sigma must be between 0 and 20 image pixels")
    th, tw = template.alpha.shape
    corners = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(corners, quad.astype(np.float32))
    inverse = cv2.getPerspectiveTransform(quad.astype(np.float32), corners)
    rect = cv2.warpPerspective(rgb, inverse, (tw, th), borderMode=cv2.BORDER_REFLECT)
    rect_visibility = cv2.warpPerspective(
        (mask > 0).astype(np.uint8), inverse, (tw, th), flags=cv2.INTER_NEAREST
    )
    # Estimate light from surface pixels, not from a person or branch over it.
    missing = (rect_visibility == 0).astype(np.uint8)
    if missing.any() and not missing.all():
        rect = cv2.inpaint(rect, missing * 255, 5, cv2.INPAINT_TELEA)
    asset, appearance = adapt_asset(
        template.rgb, rect, reference_percentile=99.5, match_black=True
    )
    x0, y0, x1, y1 = local_bounds(mask, 4)
    size = (x1 - x0, y1 - y0)
    transform = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ homography
    alpha = cv2.warpPerspective(
        template.alpha, transform, size, borderMode=cv2.BORDER_CONSTANT
    )
    pm = cv2.warpPerspective(
        linear(asset) * template.alpha[:, :, None],
        transform,
        size,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if blur_sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
        pm = cv2.GaussianBlur(pm, (0, 0), blur_sigma)
    visible = mask[y0:y1, x0:x1] > 0
    visibility_weight = visible.astype(np.float32)
    if blur_sigma > 1.5:
        # A hard SAM threshold is not an optical edge. Feather inward only:
        # protected background/occluder pixels remain untouched.
        distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 3)
        visibility_weight *= np.clip(distance / max(1.0, blur_sigma * 0.75), 0.0, 1.0)
    alpha *= visibility_weight
    pm *= visibility_weight[:, :, None]
    original = rgb[y0:y1, x0:x1]
    provisional = srgb(linear(original) * (1 - alpha[:, :, None]) + pm)
    # Restrict corner cleanup to known surface pixels, never inferred holes.
    polygon = np.zeros(visible.shape, np.uint8)
    cv2.fillConvexPoly(polygon, np.rint(quad - [x0, y0]).astype(np.int32), 1)
    cleanup = visible & (polygon > 0) & (alpha < 0.985)
    if blur_sigma > 1.5:
        cleanup[:] = False  # Inpainting a defocused boundary creates a hard cut.
    if cleanup.any():
        restored = cv2.inpaint(
            provisional, cleanup.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA
        )
        background = original.copy()
        background[cleanup] = restored[cleanup]
    else:
        background = original
    result = srgb(linear(background) * (1 - alpha[:, :, None]) + pm)
    support = (alpha > 1e-6) | cleanup
    result[~support] = original[~support]
    output = rgb.copy()
    output[y0:y1, x0:x1] = result
    return output, {
        "appearance": appearance,
        "bbox_xyxy": [x0, y0, x1, y1],
        "corner_cleanup_pixels": int(cleanup.sum()),
        "blur_sigma_px": blur_sigma,
        "visibility_policy": "strict_source_mask",
        "alpha": "premultiplied_linear",
    }


def redact_surface(rgb, mask, mode="blur", strength=0.12):
    """Redact visible pixels only. Strength is relative to visible box height.

    This does not certify unreadability: tiny plates, mask misses, and video
    reconstruction still require inspection. Grid is stable only for stills.
    """
    if mode not in {"blur", "mosaic"} or not 0 < strength <= 1:
        raise ValueError("mode must be blur/mosaic and strength within (0, 1]")
    x0, y0, x1, y1 = local_bounds(mask)
    raw = rgb[y0:y1, x0:x1]
    visible = mask[y0:y1, x0:x1] > 0
    scale = max(2.0, (y1 - y0) * strength)
    if mode == "blur":
        processed = cv2.GaussianBlur(raw, (0, 0), scale)
    else:
        tiny = cv2.resize(
            raw,
            (max(1, round(raw.shape[1] / scale)), max(1, round(raw.shape[0] / scale))),
            interpolation=cv2.INTER_AREA,
        )
        processed = cv2.resize(
            tiny, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    output = rgb.copy()
    local = output[y0:y1, x0:x1]
    local[visible] = processed[visible]
    return output, {
        "bbox_xyxy": [x0, y0, x1, y1],
        "strength": strength,
        "scale_px": scale,
        "visibility_policy": "strict_source_mask",
        "privacy_certified": False,
    }
