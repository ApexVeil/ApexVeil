"""Offline, same-time fusion of shared scene-plane hypotheses.

A symmetric temporal neighbourhood is transported by measured visible-surface
motion before fusion. Physical translation, acceleration and scale therefore do
not incur the phase lag of screen-coordinate smoothing. Missing native visibility
breaks tracks; neither native masks nor foreground protection are propagated.
"""

from dataclasses import replace

import cv2
import numpy as np

from .geometry import visible_containment, visible_coverage
from .scene_geometry import EDGE_NAMES, SceneGeometryCandidate
from .temporal import valid_quad


def _project(quad, transform):
    return cv2.perspectiveTransform(
        np.asarray(quad, np.float32)[None], np.asarray(transform, np.float64)
    )[0]


def _short(quad):
    return float(np.min(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)))


def _motion_pair(previous, current, previous_mask, current_mask, reference):
    """Fit a conditioned local motion model, with native-visibility/FB checks.

    Projective fits on small or cropped plates extrapolate a few subpixel flow
    errors into large hidden-corner jumps. Start with similarity and introduce
    affine/projective freedom only when spatial support and residual improvement
    justify it. The model-selection guard is on *full-plane* corners.
    """
    detail = {"status": "unavailable", "confidence": 0.0, "feature_count": 0}
    union = ((previous_mask > 0) | (current_mask > 0)).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(union)
    margin = max(24, min(96, max(width, height) // 3))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = (
        min(union.shape[1], x + width + margin),
        min(union.shape[0], y + height + margin),
    )
    a = previous[y0:y1, x0:x1]
    b = current[y0:y1, x0:x1]
    source_mask = (previous_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    target_mask = current_mask[y0:y1, x0:x1] > 0
    # Fit mask only; no data is written back to caller visibility.
    interior = cv2.erode(source_mask, np.ones((3, 3), np.uint8))
    if np.count_nonzero(interior) < 16:
        interior = source_mask
    points = cv2.goodFeaturesToTrack(
        a,
        maxCorners=160,
        qualityLevel=0.005,
        minDistance=2,
        mask=interior * 255,
        blockSize=3,
    )
    if points is None or len(points) < 5:
        detail["status"] = "insufficient_visible_texture"
        return None, detail
    detail["feature_count"] = len(points)
    lk = {
        "winSize": (15, 15),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    }
    moved, forward_good, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **lk)
    if moved is None:
        return None, detail
    returned, backward_good, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None, **lk)
    if returned is None:
        return None, detail
    fb = np.linalg.norm(returned[:, 0] - points[:, 0], axis=1)
    coordinates = moved[:, 0]
    inside = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < b.shape[1])
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < b.shape[0])
        & np.isfinite(coordinates).all(axis=1)
    )
    ix = np.clip(
        np.rint(np.nan_to_num(coordinates[:, 0])).astype(int), 0, b.shape[1] - 1
    )
    iy = np.clip(
        np.rint(np.nan_to_num(coordinates[:, 1])).astype(int), 0, b.shape[0] - 1
    )
    good = forward_good[:, 0].astype(bool) & backward_good[:, 0].astype(bool)
    good &= (fb < 0.9) & inside & target_mask[iy, ix]
    detail["current_visibility_rejected"] = int((inside & ~target_mask[iy, ix]).sum())
    detail["tracked_visible_points"] = int(good.sum())
    if good.sum() < 5:
        detail["status"] = "insufficient_common_visible_texture"
        return None, detail
    source = points[good, 0] + [x0, y0]
    target = moved[good, 0] + [x0, y0]
    similarity, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if similarity is None or int(inliers.sum()) < 5:
        detail["status"] = "motion_fit_failed"
        return None, detail
    inliers = inliers[:, 0].astype(bool)
    source, target = source[inliers], target[inliers]
    transform = np.vstack([similarity, [0.0, 0.0, 1.0]])
    # Even a plausible local fit is not allowed to collapse or reflect the plane.
    if not 0.70 < np.linalg.det(similarity[:, :2]) < 1.45:
        detail["status"] = "implausible_local_scale"
        return None, detail
    model = "lk_ransac_similarity"
    residual = float(
        np.median(np.linalg.norm(_project(source, transform) - target, axis=1))
    )
    coverage = 0.0
    if reference is not None:
        area = max(cv2.contourArea(np.asarray(reference, np.float32)), 1.0)
        coverage = float(
            cv2.contourArea(cv2.convexHull(source.astype(np.float32))) / area
        )
        base_corners = _project(reference, transform)
        short = _short(base_corners)
        # Narrow/censored evidence cannot identify additional degrees of freedom.
        if len(source) >= 10 and coverage >= 0.18:
            affine, _ = cv2.estimateAffine2D(
                source, target, method=cv2.RANSAC, ransacReprojThreshold=0.8
            )
            if affine is not None:
                candidate = np.vstack([affine, [0.0, 0.0, 1.0]])
                error = float(
                    np.median(
                        np.linalg.norm(_project(source, candidate) - target, axis=1)
                    )
                )
                corners = _project(reference, candidate)
                departure = float(np.linalg.norm(corners - base_corners, axis=1).max())
                if (
                    error < residual * 0.72
                    and residual - error > 0.08
                    and departure < max(0.8, short * 0.08)
                    and valid_quad(corners, base_corners)
                ):
                    transform, residual, model = candidate, error, "lk_ransac_affine"
        if len(source) >= 18 and coverage >= 0.35:
            candidate, _ = cv2.findHomography(source, target, cv2.RANSAC, 0.7)
            if candidate is not None:
                error = float(
                    np.median(
                        np.linalg.norm(_project(source, candidate) - target, axis=1)
                    )
                )
                corners = _project(reference, candidate)
                departure = float(np.linalg.norm(corners - base_corners, axis=1).max())
                if (
                    error < residual * 0.65
                    and residual - error > 0.12
                    and departure < max(0.8, short * 0.10)
                    and valid_quad(corners, base_corners)
                ):
                    transform, residual, model = (
                        candidate,
                        error,
                        "lk_ransac_homography",
                    )
    median_fb = float(np.median(fb[good][inliers]))
    confidence = (
        min(1.0, len(source) / 12) * float(inliers.mean()) / (1 + median_fb + residual)
    )
    detail.update(
        status="accepted",
        method=model,
        inliers=len(source),
        full_plane_feature_coverage=coverage,
        median_forward_backward_error_px=median_fb,
        median_reprojection_error_px=residual,
        confidence=float(confidence),
        homography=transform.tolist(),
    )
    if confidence < 0.25:
        detail["status"] = "weak_motion_rejected"
        return None, detail
    return transform, detail


def _observation_weight(candidate, mask):
    """Score geometry evidence without treating a crop as a small physical plane."""
    if candidate is None or not valid_quad(candidate.quad):
        return 0.0
    details = candidate.diagnostics
    if details.get("eligible_for_render") is False:
        return 0.0
    rank = float(np.clip(details.get("ranking_score", candidate.confidence), 0, 1))
    containment = visible_containment(mask, candidate.quad)
    if containment < 0.80:
        return 0.0
    # Physical area, not viewport-clipped area: censored hypotheses should not
    # overrule fully observed neighbours merely because their masks fit a sliver.
    fraction = min(
        1.0, np.count_nonzero(mask) / max(cv2.contourArea(candidate.quad), 1)
    )
    return max(0.005, rank) * containment * max(0.05, fraction) ** 0.5


def _aligned(quad, reference):
    """Preserve texture-corner correspondence if upper-left sorting rotates."""
    choices = [np.roll(quad, i, axis=0) for i in range(4)]
    errors = [np.mean(np.linalg.norm(q - reference, axis=1)) for q in choices]
    index = int(np.argmin(errors))
    return choices[index], index


def _robust_plane(quads, weights):
    """IRLS rejects a single corrupt corner/plane after motion compensation."""
    values = np.asarray(quads, np.float64)
    weights = np.asarray(weights, np.float64)
    result = np.average(values, axis=0, weights=weights)
    effective = weights.copy()
    for _ in range(5):
        distances = np.sqrt(np.mean(np.sum((values - result) ** 2, axis=2), axis=1))
        scale = max(0.60, float(np.median(distances)) * 1.4826)
        effective = weights * np.minimum(1.0, scale / np.maximum(distances, 1e-6))
        result = np.average(values, axis=0, weights=effective)
    return result.astype(np.float32), effective


def _edge_labels(mask, quad):
    """Rebuild viewport censorship for recovered geometry, never copy past crop."""
    labels = {name: "surface" for name in EDGE_NAMES}
    height, width = mask.shape
    for i, (a, b) in enumerate(zip(quad, np.roll(quad, -1, axis=0))):
        if (
            (min(a[0], b[0]) >= width - 4 and np.any(mask[:, -2:]))
            or (max(a[0], b[0]) <= 3 and np.any(mask[:, :2]))
            or (min(a[1], b[1]) >= height - 4 and np.any(mask[-2:]))
            or (max(a[1], b[1]) <= 3 and np.any(mask[:2]))
        ):
            labels[EDGE_NAMES[i]] = "crop"
    return labels


def stabilize_scene_trajectory(
    frames, masks, selected: list[SceneGeometryCandidate | None], *, fps
):
    """Return new SceneGeometryCandidates and inspectable offline diagnostics.

    Inputs are frame-indexed sequences (a lazy frame decoder is supported).
    Fusion uses up to 0.20 s on each side, bounded to 2..8 frames, with a maximum
    0.12 s Gaussian radius. Only motion-connected current native visibility can
    recover a missing plane. No input candidate, mask or image is modified.
    """
    count = len(selected)
    if len(frames) != count or len(masks) != count:
        raise ValueError(
            "Frames, masks and selected candidates must have equal lengths"
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    radius = int(np.clip(round(fps * 0.20), 2, 8))
    sigma = max(1.0, min(radius, fps * 0.12))
    visible = [int(np.count_nonzero(m)) >= 16 for m in masks]
    details = {
        "method": "offline_bidirectional_motion_compensated_scene_planes",
        "visibility_policy": "current_native_mask_unchanged",
        "screen_coordinate_smoothing": False,
        "window_radius_frames": radius,
        "motion_pairs": [],
        "frames": [],
    }
    transforms = [None] * max(0, count - 1)
    qualities = np.zeros(max(0, count - 1))
    # One sequential decode pass; do not keep full-resolution RGB or grayscale
    # clips resident. Each pair works in its bounded native-surface neighbourhood.
    previous = None
    for i in range(count):
        if not visible[i]:
            previous = None
            continue
        rgb = np.asarray(frames[i])
        if masks[i].ndim != 2 or rgb.shape[:2] != masks[i].shape:
            raise ValueError("RGB and native mask dimensions must match")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if previous is not None:
            reference = selected[i - 1].quad if selected[i - 1] is not None else None
            transform, record = _motion_pair(
                previous, gray, masks[i - 1], masks[i], reference
            )
            record.update(source_frame=i - 1, target_frame=i)
            transforms[i - 1] = transform
            qualities[i - 1] = record["confidence"] if transform is not None else 0
            details["motion_pairs"].append(record)
        previous = gray
    # Corner labels belong to a physical plane, not to whichever corner has
    # the smallest x+y in this frame. Preserve correspondence along verified
    # motion even when order_quad changes its starting corner.
    canonical, rotations = [None] * count, [0] * count
    predicted = None
    for i, candidate in enumerate(selected):
        if i and predicted is not None and transforms[i - 1] is not None:
            predicted = _project(predicted, transforms[i - 1])
        else:
            predicted = None
        if candidate is not None and visible[i] and valid_quad(candidate.quad):
            canonical[i] = candidate.quad.copy()
            if predicted is not None:
                canonical[i], rotations[i] = _aligned(canonical[i], predicted)
            predicted = canonical[i]
        if not visible[i]:
            predicted = None
    observation_weights = [
        _observation_weight(c, m) if v else 0.0
        for c, m, v in zip(selected, masks, visible)
    ]
    output = [
        c if weight > 0 else None for c, weight in zip(selected, observation_weights)
    ]
    for i, current in enumerate(selected):
        record = {
            "frame_index": i,
            "status": "observation_retained",
            "support_frames": [],
            "input_corner_index_rotation": rotations[i],
        }
        details["frames"].append(record)
        if not visible[i]:
            output[i] = None
            record["status"] = "native_visibility_absent"
            continue
        samples, weights, source_indices = [], [], []
        if observation_weights[i] > 0:
            samples.append(canonical[i])
            weights.append(observation_weights[i])
            source_indices.append(i)
        for direction in (-1, 1):
            transport = np.eye(3)
            reliability = 1.0
            for distance in range(1, radius + 1):
                j = i + direction * distance
                if not 0 <= j < count or not visible[j]:
                    break
                edge = j if direction < 0 else j - 1
                motion = transforms[edge]
                if motion is None:
                    break
                transport = transport @ (
                    motion if direction < 0 else np.linalg.inv(motion)
                )
                # Penalize weak links, but avoid exponential erasure of otherwise
                # consistent multi-frame evidence from tiny visible surfaces.
                reliability *= np.sqrt(qualities[edge])
                if observation_weights[j] <= 0:
                    continue
                quad = _project(canonical[j], transport)
                if not valid_quad(quad):
                    continue
                weight = (
                    observation_weights[j]
                    * reliability
                    * np.exp(-0.5 * (distance / sigma) ** 2)
                )
                if weight > 0.005:
                    samples.append(quad)
                    weights.append(weight)
                    source_indices.append(j)
        if not samples:
            output[i] = None
            record["status"] = "no_supported_plane"
            continue
        if len(samples) == 1 and source_indices[0] == i:
            record["status"] = "no_verified_temporal_support"
            continue
        reference_index = int(np.argmax(weights))
        reference = samples[reference_index]
        shifts = []
        for k in range(len(samples)):
            samples[k], shift = _aligned(samples[k], reference)
            shifts.append(shift)
        quad, effective = _robust_plane(samples, weights)
        containment = visible_containment(masks[i], quad) if valid_quad(quad) else 0.0
        record.update(
            support_frames=source_indices,
            support_weights=[float(x) for x in effective],
            corner_index_rotations=shifts,
            visible_containment=containment,
        )
        # Native masks may include isolated outliers. Require substantial current
        # support, not agreement with viewport-clipped area or fitted corners.
        if not valid_quad(quad) or containment < 0.88:
            record["status"] = "current_native_support_guard"
            continue
        prototype_index = source_indices[int(np.argmax(effective))]
        current_usable = observation_weights[i] > 0
        prototype = current if current_usable else selected[prototype_index]
        if not current_usable:
            # Recovery is geometry inference only. It never inherits a prior
            # frame's protection mask, visibility labels or eligibility evidence.
            candidate_details = {
                "eligible_for_render": True,
                "ranking_score": float(min(0.65, np.max(effective))),
                "edge_labels": _edge_labels(masks[i], quad),
                "geometry_recovery_source_frame": prototype_index,
            }
            protection = current.protected_mask if current is not None else None
            evidence = ["current_native_visibility", "verified_neighbor_surface_motion"]
            uncertainties = ["motion_recovered_full_plane_requires_review"]
        else:
            candidate_details = dict(current.diagnostics)
            if rotations[i] and "edge_labels" in candidate_details:
                labels = [
                    candidate_details["edge_labels"].get(name, "unknown")
                    for name in EDGE_NAMES
                ]
                candidate_details["edge_labels"] = dict(
                    zip(EDGE_NAMES, np.roll(labels, rotations[i]).tolist())
                )
            protection = current.protected_mask
            evidence = list(current.evidence)
            uncertainties = list(current.uncertainties)
        record["status"] = "motion_fused" if current_usable else "motion_recovered"
        if current is not None:
            aligned_current, _ = _aligned(current.quad, quad)
            record["corner_correction_rms_px"] = float(
                np.sqrt(np.mean((quad - aligned_current) ** 2))
            )
        candidate_details.update(
            visible_containment=containment,
            visible_fraction=visible_coverage(masks[i], quad),
            corners_outside_viewport=bool(
                np.any(quad < 0)
                or np.any(quad[:, 0] >= masks[i].shape[1])
                or np.any(quad[:, 1] >= masks[i].shape[0])
            ),
            visibility_policy="current_native_mask_unchanged",
            temporal=record,
            temporal_motion_uses_bounded_adjacent_chain=True,
        )
        output[i] = replace(
            prototype,
            quad=quad,
            evidence=[*evidence, "bidirectional_same_time_plane_fusion"],
            uncertainties=uncertainties,
            diagnostics=candidate_details,
            protected_mask=protection,
        )
    details["motion_pair_count"] = int(sum(t is not None for t in transforms))
    details["stabilized_frame_count"] = sum(
        r["status"] == "motion_fused" for r in details["frames"]
    )
    details["recovered_frame_count"] = sum(
        r["status"] == "motion_recovered" for r in details["frames"]
    )
    return output, details
