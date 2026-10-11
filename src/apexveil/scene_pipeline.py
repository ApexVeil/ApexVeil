"""Thin scene orchestration shared by still images and individual video frames.

Geometry proposals and advisory parameters feed the existing deterministic
compositor. Native visibility, inferred plane, foreground and edit support remain
separate. This module does not implement an alternative renderer.
"""

import hashlib
from dataclasses import asdict, is_dataclass

import cv2
import numpy as np

from .imaging.compositor import replace_surface
from .imaging.geometry import order_quad, visible_containment
from .imaging.scene_geometry import (
    SceneGeometryCandidate,
    propose_scene_geometry,
    select_scene_geometry,
)

_AUTO_SELECTION = object()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _candidate_record(candidate):
    return {
        "quad": candidate.quad.tolist(),
        "confidence": float(candidate.confidence),
        "evidence": list(candidate.evidence),
        "uncertainties": list(candidate.uncertainties),
        "diagnostics": _json_value(candidate.diagnostics),
        "protected_pixels": 0
        if candidate.protected_mask is None
        else int(np.count_nonzero(candidate.protected_mask)),
    }


def _selected_hypothesis(value):
    if isinstance(value, SceneGeometryCandidate):
        return value
    if isinstance(value, dict):
        quad = value["quad"]
        evidence = value.get("evidence", ["explicit_selected_trajectory_geometry"])
    else:
        quad, evidence = value, ["explicit_selected_trajectory_geometry"]
    return SceneGeometryCandidate(
        order_quad(quad),
        0.0,
        evidence,
        ["preselected_geometry_supplied_by_caller"],
        {"eligible_for_render": True, "ranking_score": 0.0},
    )


def solve_scene(
    rgb,
    visible_mask,
    template,
    *,
    geometry_guidance=None,
    appearance_guidance=None,
    appearance_parameter_filter=None,
    appearance_estimate_filter=None,
    degradation_estimate_filter=None,
    foreground_mask=None,
    anchors=None,
    noise_seed=0,
    selected_candidate=_AUTO_SELECTION,
):
    """Return (rendered RGB, JSON-compatible report, geometric candidates).

    selected_candidate accepts an existing SceneGeometryCandidate or a source
    quad/dictionary from trajectory selection. It bypasses proposal generation
    and ranking, preventing per-frame reselection. Omit selected_candidate for
    automatic proposals; explicitly passing None means an unresolved trajectory
    state and does not trigger a new per-frame selection. Explicit foreground evidence
    takes precedence over a candidate's conservative inferred protection; both
    remain separate from the original plate mask. External masks are not GT.
    """
    rgb = np.asarray(rgb)
    visible_mask = np.asarray(visible_mask)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Scene source must be HxWx3 uint8 RGB")
    if visible_mask.shape != rgb.shape[:2]:
        raise ValueError("Native visible mask must match source dimensions")
    if foreground_mask is not None:
        foreground_mask = np.asarray(foreground_mask)
        if foreground_mask.shape != visible_mask.shape:
            raise ValueError("Foreground mask must match source dimensions")
    original_mask_hash = hashlib.sha256(visible_mask.tobytes()).hexdigest()
    preselected = selected_candidate is not _AUTO_SELECTION
    report = {
        "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "source_shape": list(rgb.shape),
        "template": _json_value(template.metadata),
        "status": "unresolved",
        "rendered": False,
        "native_visible_pixels": int(np.count_nonzero(visible_mask)),
        "native_mask_sha256": original_mask_hash,
        "native_visibility_changed": False,
        "geometry_guidance": _json_value(geometry_guidance),
        "appearance_guidance": _json_value(appearance_guidance),
        "appearance_parameter_filter_supplied": appearance_parameter_filter is not None,
        "appearance_estimate_filter_supplied": appearance_estimate_filter is not None,
        "degradation_estimate_filter_supplied": degradation_estimate_filter is not None,
        "foreground_evidence": {
            "supplied": foreground_mask is not None,
            "ground_truth": False,
            "sha256": None
            if foreground_mask is None
            else hashlib.sha256(foreground_mask.tobytes()).hexdigest(),
            "pixels": 0
            if foreground_mask is None
            else int(np.count_nonzero(foreground_mask)),
        },
        "selection_policy": "preselected_trajectory"
        if preselected
        else "ranked_scene_candidates",
        "candidates": [],
        "selected_candidate": None,
        "noise_seed": int(noise_seed),
        "visibility_policy": "native_observation_separate_from_plane_foreground_and_edit_support",
    }
    candidates = []
    if not visible_mask.any():
        report["unresolved_reason"] = "native_visibility_missing"
        return rgb.copy(), report, candidates
    if preselected and selected_candidate is None:
        report["unresolved_reason"] = "trajectory_geometry_unresolved"
        return rgb.copy(), report, candidates
    try:
        if preselected:
            candidates = [_selected_hypothesis(selected_candidate)]
            chosen = candidates[0]
            if not chosen.diagnostics.get("eligible_for_render", False):
                chosen = None
        else:
            aspect = template.alpha.shape[1] / template.alpha.shape[0]
            candidates = propose_scene_geometry(
                rgb,
                visible_mask,
                aspect=aspect,
                guidance=geometry_guidance,
                anchors=anchors,
            )
            chosen = select_scene_geometry(candidates)
        report["candidates"] = [_candidate_record(c) for c in candidates]
        if chosen is None:
            report["unresolved_reason"] = "no_supported_full_plane_candidate"
            return rgb.copy(), report, candidates
        order_quad(chosen.quad)
        containment = visible_containment(visible_mask, chosen.quad)
        if containment < 0.88:
            report["unresolved_reason"] = "selected_plane_misses_observed_surface"
            return rgb.copy(), report, candidates
    except (ValueError, TypeError, cv2.error) as error:
        report["unresolved_reason"] = "geometry_input_or_estimation_error"
        report["error"] = f"{type(error).__name__}: {error}"
        return rgb.copy(), report, candidates
    report["selected_candidate"] = _candidate_record(chosen)
    report["selected_candidate"]["visible_containment"] = containment
    inferred_protection = chosen.protected_mask if foreground_mask is None else None
    report["protection_policy"] = (
        "provided_foreground_overrides_conservative_geometry_protection"
        if foreground_mask is not None
        else "candidate_conservative_protection"
    )
    support = {}
    try:
        output, compositor = replace_surface(
            rgb,
            visible_mask,
            chosen.quad,
            template,
            scene_guidance=appearance_guidance,
            appearance_parameter_filter=appearance_parameter_filter,
            appearance_estimate_filter=appearance_estimate_filter,
            degradation_estimate_filter=degradation_estimate_filter,
            foreground_mask=foreground_mask,
            protected_mask=inferred_protection,
            noise_seed=noise_seed,
            edit_support_out=support,
        )
    except (ValueError, TypeError, cv2.error) as error:
        report["unresolved_reason"] = "compositor_rejected_scene_parameters"
        report["error"] = f"{type(error).__name__}: {error}"
        return rgb.copy(), report, candidates
    x0, y0, x1, y1 = support["bbox_xyxy"]
    allowed = np.zeros(visible_mask.shape, bool)
    allowed[y0:y1, x0:x1] = support["support"]
    changed = np.any(output != rgb, axis=2)
    outside = int(np.count_nonzero(changed & ~allowed))
    protected_changed = int(
        np.count_nonzero(changed[y0:y1, x0:x1] & support["protected"])
    )
    foreground_changed = (
        0
        if foreground_mask is None
        else int(np.count_nonzero(changed & (foreground_mask > 0)))
    )
    mask_changed = (
        hashlib.sha256(visible_mask.tobytes()).hexdigest() != original_mask_hash
    )
    if outside or protected_changed or foreground_changed or mask_changed:
        raise RuntimeError(
            "Scene renderer violated edit support, foreground, or native visibility invariants"
        )
    needs_review = bool(chosen.uncertainties) or compositor.get("boundary", {}).get(
        "needs_review", False
    )
    report.update(
        status="rendered_needs_review" if needs_review else "rendered_unreviewed",
        rendered=True,
        changed_pixels=int(np.count_nonzero(changed)),
        outside_declared_support_changed_pixels=outside,
        protected_changed_pixels=protected_changed,
        foreground_changed_pixels=foreground_changed,
        declared_edit_support={
            "bbox_xyxy": support["bbox_xyxy"],
            "pixels": int(allowed.sum()),
        },
        compositor=_json_value(compositor),
    )
    return output, report, candidates
