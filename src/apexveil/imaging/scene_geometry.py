"""Scene-plane proposals with censored edges and direct keyframe registration.

Observed visibility remains untouched. The image viewport does not bound the
physical plane. Priors and agent guidance are explicit hypotheses, never truth.
"""

from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np

from .geometry import fit_quad, order_quad, visible_containment, visible_coverage

EDGE_NAMES = ("top", "right", "bottom", "left")


@dataclass
class SceneGeometryCandidate:
    quad: np.ndarray
    confidence: float
    evidence: list[str]
    uncertainties: list[str]
    diagnostics: dict = field(default_factory=dict)
    protected_mask: np.ndarray | None = None


def _fit_visibility(mask):
    binary = (mask > 0).astype(np.uint8)
    # Fit-only robustness: a few disconnected single-pixel predictions can
    # pull a hull hundreds of pixels away. Keep the original mask everywhere
    # else, and never discard a substantive visible fragment or occlusion split.
    x0, y0, width, height = cv2.boundingRect(binary)
    local = binary[y0 : y0 + height, x0 : x0 + width]
    _count, labels, stats, _ = cv2.connectedComponentsWithStats(local)
    areas = stats[1:, cv2.CC_STAT_AREA]
    tiny_ids = np.flatnonzero(areas <= 3) + 1
    ignored = int(areas[tiny_ids - 1].sum()) if len(tiny_ids) else 0
    if ignored and ignored <= float(areas.sum()) * 0.01:
        local[np.isin(labels, tiny_ids)] = 0
    else:
        ignored = 0
    return binary, ignored


def _line_quad(mask):
    binary, ignored = _fit_visibility(mask)
    q, method = fit_quad(binary)
    # fit_quad includes an outward optical allowance. A physical UV hypothesis
    # must estimate the line itself: its height controls hidden-side completion.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.concatenate(contours).reshape(-1, 2).astype(np.float32)
    lines = []
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        length = np.linalg.norm(b - a)
        direction = (b - a) / max(length, 1e-6)
        t = (points - a) @ direction / max(length, 1e-6)
        distance = np.abs(np.cross(points - a, direction))
        chosen = points[(t > 0.15) & (t < 0.85) & (distance < max(3, length * 0.06))]
        if len(chosen) < 6:
            return q, method, ignored
        vx, vy, x, y = cv2.fitLine(chosen, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x, y]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        a, v = lines[i - 1]
        b, u = lines[i]
        matrix = np.column_stack((v, -u))
        if abs(np.linalg.det(matrix)) < 0.025:
            return q, method, ignored
        corners.append(a + np.linalg.solve(matrix, b - a)[0] * v)
    try:
        fitted = order_quad(corners)
    except ValueError:
        return q, method, ignored
    if not 0.65 < cv2.contourArea(fitted) / max(cv2.contourArea(q), 1) < 1.5:
        return q, method, ignored
    return fitted, "unbiased_observed_boundary_lines", ignored


def _edge_classes(mask, quad, guidance):
    labels = ["surface"] * 4
    height, width = mask.shape
    for i, (a, b) in enumerate(zip(quad, np.roll(quad, -1, axis=0))):
        if (
            max(a[0], b[0]) <= 3
            and np.any(mask[:, :2])
            or min(a[0], b[0]) >= width - 4
            and np.any(mask[:, -2:])
            or max(a[1], b[1]) <= 3
            and np.any(mask[:2])
            or min(a[1], b[1]) >= height - 4
            and np.any(mask[-2:])
        ):
            labels[i] = "crop"
    supplied = guidance.get("edge_labels", {})
    aliases = {"real_plate": "surface", "image_crop": "crop"}
    if isinstance(supplied, (list, tuple)):
        if len(supplied) != 4:
            raise ValueError("edge_labels sequence must be top/right/bottom/left")
        supplied = dict(zip(EDGE_NAMES, supplied))
    for name, label in supplied.items():
        label = aliases.get(label, label)
        if name not in EDGE_NAMES or label not in {
            "surface",
            "crop",
            "occlusion",
            "unknown",
        }:
            raise ValueError("Invalid edge label")
        labels[EDGE_NAMES.index(name)] = label
    return labels


def _complete_side(fragment, missing, ratio, anchor=None):
    """Extend the two adjacent real lines, preserving the observed opposite side."""
    q = fragment.copy()
    if missing not in (1, 3):
        # Rotate the construction so the missing side is the right edge.
        turns = (missing - 1) % 4
        q = np.roll(q, -turns, axis=0)
        aq = None if anchor is None else np.roll(anchor, -turns, axis=0)
        completed = _complete_side(q, 1, 1 / ratio, aq)
        return np.roll(completed, turns, axis=0)
    opposite = 3 if missing == 1 else 1
    height = np.linalg.norm(q[(opposite + 1) % 4] - q[opposite])
    if anchor is None:
        top_length = bottom_length = height * ratio
    else:
        ah = np.linalg.norm(anchor[(opposite + 1) % 4] - anchor[opposite])
        top_length = height * np.linalg.norm(anchor[1] - anchor[0]) / max(ah, 1e-6)
        bottom_length = height * np.linalg.norm(anchor[2] - anchor[3]) / max(ah, 1e-6)
    top = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1e-6)
    bottom = (q[2] - q[3]) / max(np.linalg.norm(q[2] - q[3]), 1e-6)
    if missing == 1:
        q[1] = q[0] + top * top_length
        q[2] = q[3] + bottom * bottom_length
    else:
        q[0] = q[1] - top * top_length
        q[3] = q[2] - bottom * bottom_length
    return q.astype(np.float32)


def _refine_guided_visible_lines(fragment, labels, quad):
    """Snap only declared physical sides; retain the supplied hidden-side prior."""
    quad = order_quad(quad)
    lines = []
    for i, label in enumerate(labels):
        source = fragment if label == "surface" else quad
        a, b = source[i], source[(i + 1) % 4]
        lines.append((a, b - a))
    corners = []
    for i in range(4):
        a, v = lines[i - 1]
        b, u = lines[i]
        matrix = np.column_stack((v, -u))
        if abs(np.linalg.det(matrix)) < 1e-4:
            return None
        corners.append(a + np.linalg.solve(matrix, b - a)[0] * v)
    try:
        refined = order_quad(corners)
    except ValueError:
        return None
    if not 0.6 < cv2.contourArea(refined) / max(cv2.contourArea(quad), 1) < 1.6:
        return None
    return refined


def _anchor_initial(fragment, labels, anchor_quad):
    censored = [i for i, label in enumerate(labels) if label != "surface"]
    if len(censored) == 1:
        return _complete_side(fragment, censored[0], 1, anchor_quad)
    if censored:
        raise ValueError(
            "More than one censored side requires an explicit full-plane hypothesis"
        )
    # Full visible face: use the current line fit for registration initialization.
    return fragment.copy()


def _anchor_opposite_edge_completion(mask, anchor_quad, missing):
    """Use the visible opposite side when cropped adjacent sides are too short.

    A narrow viewport fragment cannot constrain its top/bottom directions
    reliably. Preserve the anchor's normalized projective shape and fit its
    visible side to the source mask envelope instead of extending noisy lines.
    """
    if missing not in (1, 3):
        return None
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if len(rows) < 8:
        return None
    points = []
    for y in rows:
        columns = np.flatnonzero(mask[y] > 0)
        x = columns.min() if missing == 1 else columns.max()
        points.append([x, y])
    points = np.asarray(points, np.float32)
    fit = points[
        (points[:, 1] > rows.min() + 0.1 * np.ptp(rows))
        & (points[:, 1] < rows.max() - 0.1 * np.ptp(rows))
    ]
    vx, vy, x, y = cv2.fitLine(fit, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vy) < 0.1:
        return None
    ends_y = np.array([rows.min(), rows.max()], np.float32)
    ends = np.column_stack([x + (ends_y - y) * vx / vy, ends_y])
    a, b = (
        (anchor_quad[0], anchor_quad[3])
        if missing == 1
        else (anchor_quad[1], anchor_quad[2])
    )
    u = b - a
    v = ends[1] - ends[0]
    denominator = max(float(u @ u), 1e-6)
    cosine = float(u @ v) / denominator
    sine = float(np.cross(u, v)) / denominator
    transform = np.array([[cosine, -sine], [sine, cosine]], np.float32)
    return ((anchor_quad - a) @ transform.T + ends[0]).astype(np.float32)


def _direct_register(rgb, mask, anchor, initial, aspect):
    """Register directly to a clear keyframe in canonical plate coordinates.

    This never walks a chain of adjacent-frame transforms. The native visible
    subset weights ECC; no occlusion mask is completed or unioned.
    """
    width = 320
    height = int(np.clip(round(width / aspect), 64, 240))
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    aq = order_quad(anchor["quad"])
    ah = cv2.getPerspectiveTransform(corners, aq)
    th = cv2.getPerspectiveTransform(corners, initial)
    reference = cv2.warpPerspective(anchor["rgb"], np.linalg.inv(ah), (width, height))
    target = cv2.warpPerspective(rgb, np.linalg.inv(th), (width, height))
    visibility = cv2.warpPerspective(
        (mask > 0).astype(np.uint8),
        np.linalg.inv(th),
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    am = anchor.get("mask")
    if am is not None:
        reference_visibility = cv2.warpPerspective(
            (am > 0).astype(np.uint8),
            np.linalg.inv(ah),
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        visibility &= reference_visibility
    visibility = cv2.erode(visibility, np.ones((5, 5), np.uint8))
    if visibility.sum() < width * height * 0.12:
        return None, {"status": "insufficient_common_visible_texture"}
    a = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    b = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    # Direct canonical affine registration is better conditioned than fitting
    # eight projective parameters to a tiny, clipped/blurred visible fragment.
    try:
        score, warp = cv2.findTransformECC(
            a,
            b,
            np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5),
            inputMask=visibility * 255,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None, {"status": "direct_registration_did_not_converge"}
    correction = np.vstack([warp, [0, 0, 1]])
    q = cv2.perspectiveTransform(corners[None], th @ correction)[0]
    try:
        q = order_quad(q)
    except ValueError:
        return None, {"status": "invalid_direct_registration"}
    area_ratio = cv2.contourArea(q) / max(cv2.contourArea(initial), 1)
    short = min(np.linalg.norm(initial[(i + 1) % 4] - initial[i]) for i in range(4))
    displacement = float(np.max(np.linalg.norm(q - initial, axis=1)))
    accepted = (
        score >= 0.45
        and 0.65 < area_ratio < 1.5
        and displacement < max(12, short * 0.75)
    )
    return (q if accepted else None), {
        "status": "registered" if accepted else "registration_outside_local_hypothesis",
        "ecc": float(score),
        "canonical_affine": correction.tolist(),
        "area_ratio": float(area_ratio),
        "max_corner_displacement_px": displacement,
        "common_visible_fraction": float(visibility.mean()),
    }


def propose_scene_geometry(rgb, mask, *, aspect, guidance=None, anchors=None):
    """Return ranked full-plane candidates, including explicit failed hypotheses.

    guidance: edge_labels (top/right/bottom/left), physical_aspect, source,
    foreshortening_factors, candidate_quads. anchors: sequence of dictionaries
    containing rgb, mask, quad and optional frame_index/source. Caller owns
    identity association and shot boundaries. Returned confidence is an evidence
    score, not a calibrated probability. select_scene_geometry picks an eligible
    proposal; candidate diagnostics retain assumptions and rejected attempts.
    """
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("aspect must be finite and positive")
    mask = np.asarray(mask)
    if mask.ndim != 2 or rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and native mask dimensions must match")
    guidance = dict(guidance or {})
    if "geometry" in guidance:
        guidance = {**guidance, **guidance["geometry"]}
    if guidance.get("coordinate_space", "source") != "source":
        raise ValueError("Guidance corners must be converted to source pixels")
    if (
        guidance.get("physical_aspect_guess") is not None
        and "physical_aspect" not in guidance
    ):
        guidance["physical_aspect"] = guidance["physical_aspect_guess"]
    anchors = list(anchors or [])[:8]
    if np.count_nonzero(mask) < 16:
        return []
    fit_mask, ignored_microcomponents = _fit_visibility(mask)
    fragment, method, _ = _line_quad(fit_mask)
    labels = _edge_classes(fit_mask, fragment, guidance)
    censored = [i for i, label in enumerate(labels) if label != "surface"]
    physical_aspect = float(guidance.get("physical_aspect", aspect))
    if not np.isfinite(physical_aspect) or physical_aspect <= 0:
        raise ValueError("physical_aspect must be finite and positive")
    proposals = []
    contours, _ = cv2.findContours(fit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.concatenate(contours).reshape(-1, 2).astype(np.float32)
    short = min(np.linalg.norm(fragment[(i + 1) % 4] - fragment[i]) for i in range(4))
    observed_edges = []
    for i in range(4):
        a, b = fragment[i], fragment[(i + 1) % 4]
        direction = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        distance = np.abs(np.cross(points - a, direction))
        t = (points - a) @ direction / max(np.linalg.norm(b - a), 1e-6)
        observed_edges.append(
            points[(distance < max(2, short * 0.04)) & (t > 0.10) & (t < 0.90)]
        )

    def add(quad, confidence, evidence, uncertainties, details=None, eligible=True):
        try:
            quad = order_quad(quad)
        except (ValueError, cv2.error):
            return
        containment = visible_containment(mask, quad)
        height, width = mask.shape
        outside = bool(
            np.any(quad < 0)
            or np.any(quad[:, 0] >= width)
            or np.any(quad[:, 1] >= height)
        )
        details = dict(details or {})
        details.update(
            edge_labels=dict(zip(EDGE_NAMES, labels)),
            visible_containment=containment,
            visible_fraction=visible_coverage(mask, quad),
            corners_outside_viewport=outside,
            guidance_source=guidance.get("source"),
            physical_aspect_assumption=physical_aspect,
            visibility_policy="current_native_mask_unchanged",
            fit_only_ignored_microcomponent_pixels=ignored_microcomponents,
            raw_mask_filtered=False,
        )
        if containment < 0.88:
            eligible = False
            uncertainties = [*uncertainties, "candidate_misses_observed_surface"]
        residuals = []
        for i, label in enumerate(labels):
            if label != "surface" or len(observed_edges[i]) < 4:
                continue
            a, b = quad[i], quad[(i + 1) % 4]
            direction = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            residuals.append(
                float(np.median(np.abs(np.cross(observed_edges[i] - a, direction))))
            )
        residual = float(np.mean(residuals)) if residuals else 0.0
        agreement = float(np.exp(-residual / max(2.0, short * 0.06)))
        details["observed_surface_edge_residual_px"] = residual
        details["assumptions"] = list(guidance.get("assumptions", []))
        details["eligible_for_render"] = bool(eligible)
        details["ranking_score"] = float(
            confidence * containment * agreement if eligible else -1
        )
        protection = None
        if "occlusion" in labels:
            # Explicit occluder-edge guidance justifies conservative protection
            # of absent pixels on this plane; it never grants new visibility.
            plane = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(plane, np.rint(quad).astype(np.int32), 1)
            protection = (plane > 0) & (mask == 0)
            details["protected_mask_basis"] = (
                "explicit_occlusion_guidance_and_native_absence"
            )
        proposals.append(
            SceneGeometryCandidate(
                quad,
                float(confidence),
                list(evidence),
                list(uncertainties),
                details,
                protection,
            )
        )

    add(
        fragment,
        0.72,
        [method, "observed_mask_boundary"],
        ["single_view_semantic_completeness_unverified"]
        if not censored
        else ["visible_fragment_is_not_full_uv"],
        eligible=not censored,
    )

    # A low-resolution boundary fit can collapse onto a glyph/rounded corner.
    # Keep a containing, oriented full-visible fallback rather than dropping a
    # valid native detection. It is deliberately ranked below supported lines.
    if not censored and visible_containment(fit_mask, fragment) < 0.88:
        outer = cv2.boxPoints(cv2.minAreaRect(points))
        add(
            outer,
            0.62,
            ["containing_oriented_boundary_fallback"],
            ["low_resolution_projective_shape_unresolved"],
            {"retry_reason": "line_fit_missed_current_visible_surface"},
        )

    supplied = guidance.get("candidate_quads", [])[:6]
    rejected_guidance = []
    for supplied_index, supplied_quad in enumerate(supplied):
        supplied_quad = (
            supplied_quad.get("quad")
            if isinstance(supplied_quad, dict)
            else supplied_quad
        )
        try:
            supplied_quad = order_quad(supplied_quad)
        except (ValueError, TypeError, cv2.error):
            rejected_guidance.append(
                {"index": supplied_index, "reason": "invalid_quad"}
            )
            continue
        add(
            supplied_quad,
            0.70,
            ["explicit_agent_or_user_geometry_hypothesis"],
            ["guided_corners_are_not_ground_truth"],
        )
        refined = _refine_guided_visible_lines(fragment, labels, supplied_quad)
        if refined is not None:
            add(
                refined,
                0.76,
                ["guided_hidden_side_prior", "source_observed_surface_line_refinement"],
                ["guided_hidden_extent_is_not_ground_truth"],
            )

    for index, anchor in enumerate(anchors):
        if anchor.get("quad") is None:
            continue
        try:
            aq = order_quad(anchor["quad"])
            initial = _anchor_initial(fragment, labels, aq)
        except (ValueError, cv2.error):
            continue
        provenance = {
            "anchor_index": index,
            "anchor_frame": anchor.get("frame_index"),
            "anchor_source": anchor.get("source"),
            "direct_anchor_not_adjacent_chain": True,
        }
        if len(censored) == 1 and labels[censored[0]] == "crop":
            edge_completion = _anchor_opposite_edge_completion(
                fit_mask, aq, censored[0]
            )
            if edge_completion is not None:
                add(
                    edge_completion,
                    0.69,
                    ["same_identity_visible_anchor", "source_opposite_edge_envelope"],
                    ["anchor_pose_assumed_over_narrow_crop"],
                    {
                        **provenance,
                        "retry_reason": "short_cropped_edges_may_be_ill_conditioned",
                    },
                )
        add(
            initial,
            0.77,
            ["same_identity_visible_anchor", "observed_adjacent_boundary_lines"],
            ["local_pose_extrapolation_from_anchor"],
            provenance,
        )
        if anchor.get("rgb") is not None:
            refined, registration = _direct_register(
                rgb, mask, anchor, initial, physical_aspect
            )
            provenance = {**provenance, "registration": registration}
            if refined is not None:
                add(
                    refined,
                    0.78 + 0.16 * max(0, registration["ecc"]),
                    [
                        "direct_keyframe_rgb_registration",
                        "same_identity_visible_anchor",
                    ],
                    ["blur_and_occlusion_limit_subpixel_accuracy"],
                    provenance,
                )
            else:
                # Retain the geometric hypothesis and the failed registration.
                add(
                    initial,
                    0.64,
                    ["anchor_geometry_with_failed_rgb_refinement"],
                    ["rgb_registration_unresolved"],
                    provenance,
                )

    if len(censored) == 1:
        factors = guidance.get("foreshortening_factors", [1.0, 0.85, 0.70])[:5]
        for factor in factors:
            if not 0 < float(factor) <= 1.5:
                raise ValueError("Invalid foreshortening factor")
            q = _complete_side(fragment, censored[0], physical_aspect * float(factor))
            add(
                q,
                0.56 if float(factor) == 1 else 0.50,
                [
                    "three_observed_boundary_lines",
                    "declared_physical_aspect_and_weak_perspective_prior",
                ],
                [
                    "missing_edge_not_uniquely_observable",
                    "pose_and_physical_aspect_are_assumptions",
                ],
                {"foreshortening_factor": float(factor)},
            )
    for proposal in proposals:
        proposal.diagnostics["rejected_guidance_candidates"] = rejected_guidance
    proposals.sort(key=lambda c: c.diagnostics["ranking_score"], reverse=True)
    return proposals


def select_scene_geometry(candidates):
    """Select the highest-ranked renderable hypothesis without hiding failures."""
    return max(
        (c for c in candidates if c.diagnostics.get("eligible_for_render")),
        key=lambda c: c.diagnostics.get("ranking_score", -1),
        default=None,
    )


def select_scene_trajectory(candidate_lists):
    """Select a consistent candidate path without smoothing screen coordinates.

    Translation and image scale are not penalized. The local regularizer acts
    only on normalized side lengths and anchor switching. Empty visibility
    segments remain None and break the path; no mask or plane is invented there.
    """

    def shape(candidate):
        lengths = np.linalg.norm(
            np.roll(candidate.quad, -1, axis=0) - candidate.quad, axis=1
        )
        return np.log(np.maximum(lengths[[0, 2, 3]], 1e-4) / max(lengths[1], 1e-4))

    output = [None] * len(candidate_lists)
    start = 0
    while start < len(candidate_lists):
        if not any(
            c.diagnostics.get("eligible_for_render") for c in candidate_lists[start]
        ):
            start += 1
            continue
        segment = []
        end = start
        while end < len(candidate_lists):
            eligible = [
                c
                for c in candidate_lists[end]
                if c.diagnostics.get("eligible_for_render")
            ]
            if not eligible:
                break
            segment.append(eligible)
            end += 1
        cost = -np.log(
            [max(c.diagnostics.get("ranking_score", 0), 1e-6) for c in segment[0]]
        )
        backpointers = []
        for previous, current in pairwise(segment):
            current_cost = []
            indices = []
            for candidate in current:
                transitions = []
                for prior in previous:
                    delta = np.linalg.norm(shape(candidate) - shape(prior))
                    shape_cost = min(0.5, 0.6 * delta)
                    a = candidate.diagnostics.get("anchor_frame")
                    b = prior.diagnostics.get("anchor_frame")
                    switch_cost = (
                        0.12 if a is not None and b is not None and a != b else 0
                    )
                    transitions.append(shape_cost + switch_cost)
                values = cost + transitions
                index = int(np.argmin(values))
                indices.append(index)
                current_cost.append(
                    values[index]
                    - np.log(max(candidate.diagnostics.get("ranking_score", 0), 1e-6))
                )
            backpointers.append(indices)
            cost = np.asarray(current_cost)
        index = int(np.argmin(cost))
        for offset in range(len(segment) - 1, -1, -1):
            output[start + offset] = segment[offset][index]
            if offset:
                index = backpointers[offset - 1][index]
        start = end
    return output
