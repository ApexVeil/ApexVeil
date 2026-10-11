"""Declared local edit support, visibility and bounded edge decontamination.

A segmentation is observed visibility, not an immutable surface boundary.
Expansion needs competing source/background color evidence. Ambiguous holes are
protected; this estimator cannot distinguish a screw from a tiny foreground.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoundaryEstimate:
    bbox_xyxy: list[int]
    support: np.ndarray
    visible: np.ndarray
    protected: np.ndarray
    diagnostics: dict
    coverage: np.ndarray | None = None


def estimate_boundary(rgb, visible_mask, quad, *, protected_mask=None, blur_sigma=0.35):
    q = np.asarray(quad, np.float32)
    short = min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    band = int(np.clip(round(short * 0.045 + blur_sigma), 2, 8))
    ys, xs = np.where(visible_mask > 0)
    if not len(xs):
        raise ValueError("Visible mask is empty")
    x0, y0 = (
        max(0, int(min(xs.min(), q[:, 0].min())) - band - 3),
        max(0, int(min(ys.min(), q[:, 1].min())) - band - 3),
    )
    x1, y1 = (
        min(rgb.shape[1], int(max(xs.max(), q[:, 0].max())) + band + 4),
        min(rgb.shape[0], int(max(ys.max(), q[:, 1].max())) + band + 4),
    )
    raw = visible_mask[y0:y1, x0:x1] > 0
    plane = np.zeros(raw.shape, np.uint8)
    cv2.fillConvexPoly(plane, np.rint(q - [x0, y0]).astype(np.int32), 1)
    inside = cv2.distanceTransform(plane, cv2.DIST_L2, 3)
    # Native absent holes are protected even beside the geometric boundary.
    # Their color cannot prove they belong to the replaceable material.
    _, missing_labels = cv2.connectedComponents((~raw).astype(np.uint8))
    exterior_labels = np.unique(
        np.concatenate(
            [
                missing_labels[0],
                missing_labels[-1],
                missing_labels[:, 0],
                missing_labels[:, -1],
            ]
        )
    )
    enclosed = ~raw & ~np.isin(missing_labels, exterior_labels)
    protected = enclosed.copy()
    if protected_mask is not None:
        if protected_mask.shape != visible_mask.shape:
            raise ValueError("protected_mask must have the source mask shape")
        protected |= protected_mask[y0:y1, x0:x1] > 0
    raw &= ~protected
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1,) * 2)
    adjacent = cv2.dilate(raw.astype(np.uint8), kernel) > 0
    candidate = adjacent & ~raw & ~protected
    evidence_rgb = cv2.GaussianBlur(
        rgb[y0:y1, x0:x1], (0, 0), max(0.6, min(1.2, blur_sigma * 0.5))
    )
    lab = cv2.cvtColor(evidence_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    surface = lab[raw & (inside <= band * 2 + 2)]
    surrounding = lab[(cv2.dilate(plane, kernel) > 0) & (plane == 0) & ~adjacent]
    expansion = np.zeros_like(raw)
    if len(surface) >= 12 and len(surrounding) >= 12 and candidate.any():
        # Deterministic small codebooks preserve mixed lighting without fitting
        # a Gaussian to an entire green plate and black glyph population.
        def distance(values, samples):
            samples = samples[:: max(1, len(samples) // 128)]
            return np.min(
                np.sum((values[:, None] - samples[None]) ** 2, axis=2), axis=1
            )

        values = lab[candidate]
        ds, db = distance(values, surface), distance(values, surrounding)
        expansion[candidate] = (ds < db * 0.55) & (ds < 22**2)
        # Only pixels connected to observed visibility can become editable.
        expanded = raw | expansion
        _, labels = cv2.connectedComponents(expanded.astype(np.uint8))
        expansion &= np.isin(labels, np.unique(labels[raw]))
    visible = (raw | expansion) & ~protected
    removal = visible & ~protected
    return BoundaryEstimate(
        [x0, y0, x1, y1],
        removal,
        visible,
        protected,
        {
            "method": "bounded_competing_color_boundary_support",
            "band_px": band,
            "observed_pixels": int(raw.sum()),
            "expanded_pixels": int(expansion.sum()),
            "protected_pixels": int(protected.sum()),
            "enclosed_missing_pixels": int(enclosed.sum()),
            "edit_support_pixels": int(removal.sum()),
            "old_surface_removal_pixels": int(removal.sum()),
            "background_recovery": "independent_old_material_removal_background",
            "protection_confidence": "explicit"
            if protected_mask is not None
            else "conservative_hole_inference",
            "needs_review": bool(protected.any() and protected_mask is None)
            or bool(expansion.sum() > raw.sum() * 0.2),
            "physically_calibrated": False,
            "soft_boundary_method": "native_template_alpha_without_inward_crop",
            "soft_boundary_width_px": 0.0,
            "soft_boundary_pixels": 0,
        },
        None,
    )


def extend_boundary_bounds(boundary, bbox):
    # Explicit old-material support can exceed the new template footprint.
    a, b, c, d = boundary.bbox_xyxy
    x0, y0, x1, y1 = min(a, bbox[0]), min(b, bbox[1]), max(c, bbox[2]), max(d, bbox[3])
    if [x0, y0, x1, y1] == boundary.bbox_xyxy:
        return boundary

    def extend(value):
        if value is None:
            return None
        result = np.zeros((y1 - y0, x1 - x0), value.dtype)
        result[b - y0 : d - y0, a - x0 : c - x0] = value
        return result

    return BoundaryEstimate(
        [x0, y0, x1, y1],
        extend(boundary.support),
        extend(boundary.visible),
        extend(boundary.protected),
        dict(boundary.diagnostics),
        extend(boundary.coverage),
    )
