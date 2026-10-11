"""Local background reconstruction for material optical edges.

A donor gap can remove a real frame or ridge. Adjacent and more distant reconstructions expose uncertainty about the unseen
background. Disagreement selects the nearest exterior approximation; it never
shrinks material coverage or retains the old plate as a quality workaround.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BackgroundRepairEstimate:
    rgb: np.ndarray
    applied_mask: np.ndarray
    diagnostics: dict


def estimate_background_repair(
    rgb, plane, visible, protected, requested, *, blur_sigma, radius
):
    requested = requested & visible & ~protected
    count = int(requested.sum())
    if not count:
        return BackgroundRepairEstimate(
            rgb.copy(),
            np.zeros_like(requested),
            {
                "method": "old_surface_removal_nearest_exterior_background",
                "requested_pixels": 0,
                "applied_pixels": 0,
                "suppressed_pixels": 0,
                "reliable": True,
                "reason": "no_background_repair_requested",
            },
        )
    base = ((plane > 0) | visible).astype(np.uint8)
    gap = max(1, min(10, round(blur_sigma * 2 + 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap * 2 + 1,) * 2)
    # Foreground is not a background donor. Reconstruct beneath it for the
    # background layer only; output protection still preserves its RGB exactly.
    near_mask = (base > 0) | protected
    far_mask = (cv2.dilate(base, kernel) > 0) | protected
    near, far = [
        cv2.inpaint(rgb, m.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        for m in (near_mask, far_mask)
    ]
    near_lab, far_lab = [
        cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32) for im in (near, far)
    ]
    disagreement = np.linalg.norm(near_lab - far_lab, axis=2)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    residual = lab - cv2.GaussianBlur(lab, (0, 0), 0.7)
    donors = ~far_mask & ~protected
    # Median exterior high-frequency magnitude gives a robust source-derived
    # noise/texture tolerance. Four encoded Lab units cover quantization and
    # tiny inpaint rounding differences; this is not a hue or darkness test.
    residual_level = (
        float(np.median(np.linalg.norm(residual[donors], axis=1)))
        if donors.any()
        else 0.0
    )
    tolerance = max(4.0, 3.0 * residual_level)
    q75 = float(np.percentile(disagreement[requested], 75))
    reliable = bool(donors.sum() >= 12 and q75 <= tolerance)
    background = rgb.copy()
    applied = requested.copy()
    background[applied] = (far if reliable else near)[applied]
    return BackgroundRepairEstimate(
        background,
        applied,
        {
            "method": "old_surface_removal_nearest_exterior_background",
            "requested_pixels": count,
            "applied_pixels": int(applied.sum()),
            "suppressed_pixels": count - int(applied.sum()),
            "reliable": reliable,
            "reason": "near_and_gap_donors_agree"
            if reliable
            else "uncertain_background_use_nearest_exterior_without_shrinking_asset",
            "donor_gap_px": gap,
            "donor_samples": int(donors.sum()),
            "donor_disagreement_p75_lab_code": q75,
            "tolerance_lab_code": tolerance,
            "source_exterior_residual_lab_code": residual_level,
            "physically_calibrated": False,
        },
    )
