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
    x0, y0 = max(0, int(xs.min()) - band - 3), max(0, int(ys.min()) - band - 3)
    x1, y1 = (
        min(rgb.shape[1], int(xs.max()) + band + 4),
        min(rgb.shape[0], int(ys.max()) + band + 4),
    )
    raw = visible_mask[y0:y1, x0:x1] > 0
    plane = np.zeros(raw.shape, np.uint8)
    cv2.fillConvexPoly(plane, np.rint(q - [x0, y0]).astype(np.int32), 1)
    inside = cv2.distanceTransform(plane, cv2.DIST_L2, 3)
    absent = (plane > 0) & ~raw
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
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(absent.astype(np.uint8))
    # Protect absent interior evidence. A component may also contain a missed
    # outer strip, so do not label the entire connected component foreground.
    # Boundary-band additions still require competing color evidence.
    for i in range(1, count):
        region = labels == i
        if np.any(inside[region] > band + 1):
            protected |= region & (inside > band)
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
    # Regularize correction strength only at the exterior plane boundary.
    # This does not fill holes or blur an occluder: their visibility/protection
    # still gates rendering exactly. A narrow inward ramp prevents isolated
    # color-supported additions from becoming full-strength repair dots.
    distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 3)
    width = max(1.5, min(3.0, blur_sigma * 2.0))
    t = np.clip(distance / width, 0, 1)
    ramp = t * t * (3 - 2 * t)
    exterior_band = (inside <= band + 1) & visible
    near_protection = (
        cv2.dilate(protected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    )
    coverage = np.ones(visible.shape, np.float32)
    coverage[exterior_band & ~near_protection] = ramp[exterior_band & ~near_protection]
    coverage[~visible] = 0
    return BoundaryEstimate(
        [x0, y0, x1, y1],
        visible.copy(),
        visible,
        protected,
        {
            "method": "bounded_competing_color_boundary_support",
            "band_px": band,
            "observed_pixels": int(raw.sum()),
            "expanded_pixels": int(expansion.sum()),
            "protected_pixels": int(protected.sum()),
            "enclosed_missing_pixels": int(enclosed.sum()),
            "edit_support_pixels": int(visible.sum()),
            "background_recovery": "bounded_local_inpaint_approximation",
            "protection_confidence": "explicit"
            if protected_mask is not None
            else "conservative_hole_inference",
            "needs_review": bool(protected.any() and protected_mask is None)
            or bool(expansion.sum() > raw.sum() * 0.2),
            "physically_calibrated": False,
            "soft_boundary_method": "outer_band_inward_correction_coverage",
            "soft_boundary_width_px": width,
            "soft_boundary_pixels": int(((coverage > 0) & (coverage < 1)).sum()),
        },
        coverage,
    )
