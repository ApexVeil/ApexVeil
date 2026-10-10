"""Offline, motion-normalized geometry consensus for a single native identity.

No image generation, model inference, mask union, or screen-coordinate temporal
filtering occurs here. Each anchor is independently transported to the requested
frame using measured, bidirectionally checked surface motion. Robust consensus
then refines the whole trajectory in those same-time coordinate systems.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from .geometry import (
    SurfaceGeometry,
    estimate_surface_geometry,
    visible_containment,
    visible_coverage,
)
from .temporal import flow_prediction, valid_quad


def _transport(quad, matrix):
    result = cv2.perspectiveTransform(
        np.asarray(quad, np.float32)[None], np.asarray(matrix, np.float64)
    )[0]
    return result if valid_quad(result) else None


def _motion_pair(a, b, mask_a, mask_b, qa, qb):
    """Independent forward and reverse LK/RANSAC fits on visible pixels only."""
    kernel = np.ones((3, 3), np.uint8)
    ma = cv2.erode((mask_a > 0).astype(np.uint8), kernel)
    mb = cv2.erode((mask_b > 0).astype(np.uint8), kernel)
    details = {"status": "insufficient_motion", "forward": {}, "backward": {}}
    if qa is None or qb is None or not ma.any() or not mb.any():
        return None, None, 0.0, details
    _, forward = flow_prediction(a, b, ma, qa, current_mask=mb)
    _, backward = flow_prediction(b, a, mb, qb, current_mask=ma)
    details.update(forward=forward, backward=backward)
    if "homography" not in forward or "homography" not in backward:
        return None, None, 0.0, details
    hf = np.asarray(forward["homography"])
    hb = np.asarray(backward["homography"])
    projected = _transport(qa, hf)
    roundtrip = _transport(qa, hb @ hf)
    if projected is None or roundtrip is None:
        return None, None, 0.0, details
    error = float(np.max(np.linalg.norm(roundtrip - qa, axis=1)))
    short = min(np.linalg.norm(qa[(i + 1) % 4] - qa[i]) for i in range(4))
    limit = max(1.5, short * 0.08)
    confidence = min(forward["confidence"], backward["confidence"]) / (
        1 + error / limit
    )
    details["corner_cycle_error_px"] = error
    # The native mask is an independent check: flow must explain observed pixels,
    # but no penalty applies merely because a foreground occluder removes pixels.
    explained = visible_containment(mask_b, projected)
    details["forward_visible_containment"] = explained
    if error > limit or confidence < 0.25 or explained < 0.80:
        details["status"] = "inconsistent_bidirectional_motion"
        return None, None, 0.0, details
    details["status"] = "bidirectional_motion"
    return hf, hb, float(confidence), details


def _robust_consensus(candidates, weights):
    """Bounded-influence joint corner consensus, initialized by a robust medoid."""
    qs = np.asarray(candidates, np.float64)
    ws = np.asarray(weights, np.float64)
    distances = np.max(np.linalg.norm(qs[:, None] - qs[None], axis=-1), axis=-1)
    medoid = int(np.argmin(np.sum(np.minimum(distances, 20) * ws[None], axis=1)))
    q = qs[medoid].copy()
    short = min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    scale = max(1.25, short * 0.035)
    for _ in range(8):
        residuals = np.max(np.linalg.norm(qs - q, axis=-1), axis=-1)
        robust = np.square(np.maximum(0, 1 - (residuals / (4.685 * scale)) ** 2))
        effective = ws * robust
        if effective.sum() <= 1e-8:
            break
        candidate = np.sum(qs * effective[:, None, None], axis=0) / effective.sum()
        if np.max(np.abs(candidate - q)) < 0.01:
            q = candidate
            break
        q = candidate
    residuals = np.max(np.linalg.norm(qs - q, axis=-1), axis=-1)
    return q.astype(np.float32), residuals, scale


def refine_shot_geometry(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    template_aspect_ratio: float,
    *,
    cut_indices: Sequence[int] = (),
    observed_quads: Sequence[np.ndarray | None] | None = None,
    max_anchor_distance: int = 12,
    identity: str | int | None = None,
) -> list[SurfaceGeometry]:
    """Return one estimate per frame, preserving each original visibility mask.

    Caller must supply one already associated native identity per invocation.
    Alias merging, target selection, and segmentation reruns are not performed.
    cuts[i] means a new shot begins at frame i; no anchor crosses that boundary.
    Empty masks never receive inferred visible pixels or an inferred plane.
    Frames can be a lazy sequence to bound decoding memory. Native masks must
    have the same dimensions as their corresponding RGB frames.
    """
    count = len(frames)
    if len(masks) != count or (
        observed_quads is not None and len(observed_quads) != count
    ):
        raise ValueError(
            "Frames, masks, and optional observations must have equal length"
        )
    if max_anchor_distance < 0:
        raise ValueError("Anchor distance cannot be negative")
    cuts = set(cut_indices)
    if any(i < 0 or i >= count for i in cuts):
        raise ValueError("Cut indices must identify existing frames")
    observations = [
        estimate_surface_geometry(
            frames[i],
            masks[i],
            template_aspect_ratio,
            observed_quad=None if observed_quads is None else observed_quads[i],
        )
        for i in range(count)
    ]
    motions = [None] * max(0, count - 1)
    reverse = [None] * max(0, count - 1)
    confidences = [0.0] * max(0, count - 1)
    pair_details = [{} for _ in range(max(0, count - 1))]
    previous = None
    for i in range(count):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        if i and i not in cuts:
            delta = float(np.mean(np.abs(gray.astype(np.float32) - previous)) / 255)
            if delta > 0.30:
                cuts.add(i)
                pair_details[i - 1] = {
                    "status": "possible_cut_or_flash",
                    "difference": delta,
                }
            else:
                qa = observations[i - 1].diagnostics.get("candidate_quad")
                qb = observations[i].diagnostics.get("candidate_quad")
                qa = None if qa is None else np.asarray(qa, np.float32)
                qb = None if qb is None else np.asarray(qb, np.float32)
                result = _motion_pair(previous, gray, masks[i - 1], masks[i], qa, qb)
                (
                    motions[i - 1],
                    reverse[i - 1],
                    confidences[i - 1],
                    pair_details[i - 1],
                ) = result
        elif i:
            pair_details[i - 1] = {"status": "explicit_cut"}
        previous = gray
    output = []
    for i, observation in enumerate(observations):
        candidates, weights, anchors = [], [], []
        if observation.quad is not None and observation.confidence >= 0.30:
            candidates.append(observation.quad)
            weights.append(observation.confidence)
            anchors.append(
                {
                    "frame": i,
                    "direction": "current",
                    "distance": 0,
                    "motion_confidence": 1.0,
                    "weight": observation.confidence,
                }
            )
        if np.count_nonzero(masks[i]) >= 16:
            for direction in (-1, 1):
                transform = np.eye(3)
                path_confidence = 1.0
                for distance in range(1, max_anchor_distance + 1):
                    anchor = i + direction * distance
                    if anchor < 0 or anchor >= count:
                        break
                    edge = anchor if direction == -1 else anchor - 1
                    matrix = motions[edge] if direction == -1 else reverse[edge]
                    if matrix is None:
                        break
                    transform = transform @ matrix
                    # A long chain accumulates uncertainty even when each link fits.
                    path_confidence = min(path_confidence, confidences[edge])
                    source = observations[anchor]
                    if source.quad is None or source.confidence < 0.30:
                        continue
                    prediction = _transport(source.quad, transform)
                    if (
                        prediction is None
                        or visible_containment(masks[i], prediction) < 0.90
                    ):
                        continue
                    weight = source.confidence * path_confidence * np.exp(-distance / 8)
                    if weight < 0.08:
                        continue
                    candidates.append(prediction)
                    weights.append(weight)
                    anchors.append(
                        {
                            "frame": anchor,
                            "direction": "forward" if direction == -1 else "backward",
                            "distance": distance,
                            "motion_confidence": path_confidence,
                            "weight": float(weight),
                        }
                    )
        details = dict(observation.diagnostics)
        details.update(
            estimator="offline_bidirectional_motion_consensus",
            association={
                "identity": identity,
                "policy": "caller_associated_native_identity",
            },
            anchors=anchors,
            incoming_motion=None if i == 0 else pair_details[i - 1],
            outgoing_motion=None if i + 1 == count else pair_details[i],
            cut_before=i in cuts,
        )
        flags = list(observation.flags)
        quad = None
        confidence = 0.0
        if candidates:
            quad, residuals, scale = _robust_consensus(candidates, weights)
            valid = valid_quad(quad) and visible_containment(masks[i], quad) >= 0.90
            inliers = residuals <= 4.685 * scale
            details.update(
                anchor_residuals_px=residuals.tolist(),
                anchor_inliers=inliers.tolist(),
                consensus_scale_px=scale,
            )
            if not valid:
                quad = observation.quad
                flags.append("invalid_consensus_keep_observation")
            if quad is not None:
                mean_residual = float(
                    np.average(residuals[inliers], weights=np.asarray(weights)[inliers])
                )
                confidence = float(
                    max(np.asarray(weights)[inliers]) / (1 + mean_residual / scale)
                )
                details.update(
                    coverage=visible_coverage(masks[i], quad),
                    visible_containment=visible_containment(masks[i], quad),
                )
                if observation.quad is None:
                    flags.append("temporally_supported_full_plane")
                elif (
                    np.max(np.linalg.norm(quad - observation.quad, axis=1))
                    > 4.685 * scale
                ):
                    flags.append("observation_motion_disagreement")
                if len(candidates) > 1:
                    flags.append("offline_motion_refined")
        if quad is None:
            flags.append("insufficient_full_plane_evidence")
        output.append(
            SurfaceGeometry(
                quad, masks[i], confidence, list(dict.fromkeys(flags)), details
            )
        )
    return output
