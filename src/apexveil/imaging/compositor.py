"""Visibility-aware replacement and redaction, shared by still/video frames.

Edits are restricted to declared local support and preserve protected foreground.
The inferred plane is not visibility; ambiguous holes remain protected.
"""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .appearance import estimate_context_tint, render_appearance
from .background import estimate_background_repair
from .boundary import extend_boundary_bounds
from .color import linear, luma, srgb
from .degradation import apply_capture_noise
from .scene_appearance import apply_scene_guidance


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


def replace_surface(
    rgb,
    mask,
    quad,
    template: Template,
    blur_sigma=None,
    *,
    appearance_reference=None,
    appearance_parameter_filter=None,
    appearance_estimate_filter=None,
    degradation_estimate_filter=None,
    appearance_estimate=None,
    boundary_estimate=None,
    protected_mask=None,
    degradation_estimate=None,
    noise_seed=0,
    edit_support_out=None,
    estimators=None,
    scene_guidance=None,
    old_surface_mask=None,
    foreground_mask=None,
):
    """Insert a material, retaining visible-mask holes and rounded artwork."""
    if estimators is None:
        from apexveil.estimation import get_default_estimators

        estimators = get_default_estimators()
    degradation = degradation_estimate or estimators.degradation.estimate(
        rgb, mask, quad
    )
    if degradation_estimate_filter is not None:
        degradation = degradation_estimate_filter(
            degradation,
            visible_fraction=min(
                1.0,
                float(np.count_nonzero(mask))
                / max(cv2.contourArea(np.asarray(quad, np.float32)), 1.0),
            ),
        )
    # None delegates entirely to the selected estimator. Every numeric value,
    # including zero, is an explicit override and must not be silently raised.
    blur_sigma = float(degradation.sigma_px if blur_sigma is None else blur_sigma)
    if not np.isfinite(blur_sigma) or not 0 <= blur_sigma <= 20:
        raise ValueError("blur_sigma must be between 0 and 20 image pixels")
    boundary = boundary_estimate or estimators.boundary.estimate(
        rgb, mask, quad, protected_mask=protected_mask, blur_sigma=blur_sigma
    )
    if old_surface_mask is not None:
        if old_surface_mask.shape != mask.shape:
            raise ValueError("old_surface_mask must have source shape")
        if np.any(old_surface_mask):
            boundary = extend_boundary_bounds(boundary, local_bounds(old_surface_mask))
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
    if appearance_reference is not None:
        if (
            appearance_reference.shape != template.rgb.shape
            or appearance_reference.dtype != np.uint8
        ):
            raise ValueError(
                "appearance_reference must be canonical RGB uint8 with the template shape"
            )
        light_reference = appearance_reference
    else:
        light_reference = rect
    context_tint, context_detail = estimate_context_tint(rgb, mask, quad)
    estimated_appearance = appearance_estimate or estimators.appearance.estimate(
        template.rgb,
        light_reference,
        reference_percentile=99.5,
        match_black=True,
        valid_mask=rect_visibility,
        context_tint=context_tint,
        ink_resolved=min(np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4))
        / max(0.5, blur_sigma)
        >= 60,
        context_black_level=context_detail.get("luminance_linear"),
        parameter_filter=None,
    )
    estimated_appearance = apply_scene_guidance(estimated_appearance, scene_guidance)
    if appearance_estimate_filter is not None:
        source_levels = np.percentile(luma(linear(rect))[rect_visibility > 0], [50, 80])
        estimated_appearance = appearance_estimate_filter(
            estimated_appearance,
            source_levels,
            valid_mask=rect_visibility,
            source_reference=rect,
            context_black_level=context_detail.get("luminance_linear"),
        )
    if appearance_parameter_filter is not None:
        source_levels = np.percentile(luma(linear(rect))[rect_visibility > 0], [50, 80])
        gain, veil, tint, temporal = appearance_parameter_filter(
            estimated_appearance.gain,
            estimated_appearance.black,
            estimated_appearance.tint,
            source_levels,
        )
        diagnostics = dict(estimated_appearance.diagnostics)
        diagnostics.update(
            exposure_gain_linear=gain,
            temporal_parameters=temporal,
            black_point_linear=float(np.median(luma(veil))),
            black_field_rgb_p50_linear=np.median(veil.reshape(-1, 3), axis=0).tolist(),
        )
        estimated_appearance = replace(
            estimated_appearance,
            gain=gain,
            black=veil,
            tint=tint,
            diagnostics=diagnostics,
        )
    asset = render_appearance(template.rgb, estimated_appearance)
    appearance = dict(estimated_appearance.diagnostics)
    appearance["context"] = context_detail
    appearance["reference"] = (
        "external_temporal" if appearance_reference is not None else "current_frame"
    )
    x0, y0, x1, y1 = boundary.bbox_xyxy
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
    protection = boundary.protected.copy()
    if foreground_mask is not None:
        if foreground_mask.shape != mask.shape:
            raise ValueError("foreground_mask must have source shape")
        protection = foreground_mask[y0:y1, x0:x1] > 0
    if protected_mask is not None:
        if protected_mask.shape != mask.shape:
            raise ValueError("protected_mask must have the source mask shape")
        protection |= protected_mask[y0:y1, x0:x1] > 0
    removal = boundary.support.copy()
    if foreground_mask is not None and old_surface_mask is None:
        declared_plane = np.zeros(removal.shape, np.uint8)
        cv2.fillConvexPoly(declared_plane, np.rint(quad - [x0, y0]).astype(np.int32), 1)
        removal |= declared_plane > 0
    if old_surface_mask is not None:
        if old_surface_mask.shape != mask.shape:
            raise ValueError("old_surface_mask must have source shape")
        removal = old_surface_mask[y0:y1, x0:x1] > 0
    visible = removal & ~protection
    visibility_weight = visible.astype(np.float32)
    alpha *= visibility_weight
    pm *= visibility_weight[:, :, None]
    original = rgb[y0:y1, x0:x1]
    # Remove the old surface beneath the optical transition. Inpainting the
    # whole visible region supplies nearby background at its exterior only;
    # the opaque material hides the unconstrained interior reconstruction.
    requested_cleanup = visible & (alpha < 0.985)
    plane = np.zeros(visible.shape, np.uint8)
    cv2.fillConvexPoly(plane, np.rint(quad - [x0, y0]).astype(np.int32), 1)
    repair = estimate_background_repair(
        original,
        plane,
        visible,
        protection,
        requested_cleanup,
        blur_sigma=blur_sigma,
        radius=max(2, min(8, boundary.diagnostics.get("band_px", 3))),
    )
    background, cleanup = repair.rgb, repair.applied_mask
    result = srgb(linear(background) * (1 - alpha[:, :, None]) + pm)
    noised = apply_capture_noise(result, degradation, noise_seed)
    # Capture noise belongs to the material and reconstructed background.
    result = np.clip(
        np.rint(
            result.astype(np.float32)
            + (noised.astype(np.float32) - result.astype(np.float32))
            * np.maximum(alpha, cleanup.astype(np.float32))[:, :, None]
        ),
        0,
        255,
    ).astype(np.uint8)
    support = visible
    retained_outer_band = np.zeros_like(support)
    result[~support] = original[~support]
    if edit_support_out is not None:
        edit_support_out.update(
            bbox_xyxy=boundary.bbox_xyxy,
            support=support.copy(),
            protected=protection.copy(),
            old_surface_removal=visible.copy(),
            template_alpha=alpha.copy(),
        )
    output = rgb.copy()
    output[y0:y1, x0:x1] = result
    boundary_diagnostics = dict(boundary.diagnostics)
    repair_fraction = float(cleanup.sum() / max(1, support.sum()))
    boundary_diagnostics["foreground_source"] = (
        "authoritative_foreground_mask"
        if foreground_mask is not None
        else "conservative_native_visibility"
    )
    boundary_diagnostics["background_repair_fraction"] = repair_fraction
    boundary_diagnostics["background_repair"] = repair.diagnostics
    boundary_diagnostics["retained_uncertain_outer_band_pixels"] = int(
        retained_outer_band.sum()
    )
    boundary_diagnostics["boundary_replacement_complete"] = (
        False if retained_outer_band.any() else None
    )
    boundary_diagnostics["ridge_policy"] = (
        "independent_old_removal_and_native_asset_alpha"
    )
    boundary_diagnostics["needs_review"] = (
        bool(boundary_diagnostics.get("needs_review", False))
        or repair_fraction > 0.2
        or not repair.diagnostics["reliable"]
    )
    return output, {
        "appearance": appearance,
        "bbox_xyxy": [x0, y0, x1, y1],
        "corner_cleanup_pixels": int(cleanup.sum()),
        "requested_corner_cleanup_pixels": int(requested_cleanup.sum()),
        "blur_sigma_px": blur_sigma,
        "visibility_policy": "declared_local_support_and_protected_foreground",
        "boundary": boundary_diagnostics,
        "degradation": degradation.diagnostics,
        "declared_edit_support": {
            "bbox_xyxy": boundary.bbox_xyxy,
            "pixels": int(support.sum()),
        },
        "noise_seed": int(noise_seed),
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
