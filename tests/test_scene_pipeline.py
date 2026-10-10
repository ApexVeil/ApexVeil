import json

import numpy as np
import pytest

import apexveil.scene_pipeline as pipeline
from apexveil.imaging.compositor import Template
from apexveil.imaging.scene_geometry import SceneGeometryCandidate


def scene():
    rgb = np.full((90, 180, 3), 40, np.uint8)
    rgb[25:65, 30:150] = [40, 150, 80]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[25:65, 30:150] = 1
    template = Template(
        np.full((40, 120, 3), 25, np.uint8), np.ones((40, 120), np.float32), {}
    )
    quad = np.array([[30, 25], [149, 25], [149, 64], [30, 64]], np.float32)
    return rgb, mask, template, quad


def test_scene_pipeline_uses_existing_compositor_and_keeps_native_visibility():
    rgb, mask, template, _ = scene()
    before = mask.copy()
    output, report, candidates = pipeline.solve_scene(
        rgb, mask, template, noise_seed=23
    )
    assert report["rendered"]
    assert candidates
    assert report["compositor"]["noise_seed"] == 23
    assert report["outside_declared_support_changed_pixels"] == 0
    np.testing.assert_array_equal(mask, before)
    assert np.any(output != rgb)
    json.dumps(report)


def test_empty_native_mask_is_explicitly_unresolved():
    rgb, mask, template, quad = scene()
    output, report, candidates = pipeline.solve_scene(
        rgb, np.zeros_like(mask), template, selected_candidate=quad
    )
    np.testing.assert_array_equal(output, rgb)
    assert report["status"] == "unresolved"
    assert report["unresolved_reason"] == "native_visibility_missing"
    assert not report["rendered"]
    assert not candidates


def test_preselected_trajectory_quad_bypasses_per_frame_proposals(monkeypatch):
    rgb, mask, template, quad = scene()

    def forbidden(*args, **kwargs):
        raise AssertionError("Trajectory geometry must not be reselected")

    monkeypatch.setattr(pipeline, "propose_scene_geometry", forbidden)
    candidate = SceneGeometryCandidate(
        quad, 0.8, ["trajectory"], [], {"eligible_for_render": True}
    )
    _, report, returned = pipeline.solve_scene(
        rgb,
        mask,
        template,
        selected_candidate=candidate,
        appearance_guidance={"exposure_gain_linear": 0.2},
    )
    assert report["selection_policy"] == "preselected_trajectory"
    assert returned[0] is candidate
    assert report["compositor"]["appearance"]["exposure_gain_linear"] == 0.2


def test_external_foreground_is_preserved_and_reported_not_ground_truth():
    rgb, mask, template, quad = scene()
    foreground = np.zeros_like(mask)
    foreground[15:70, 65:80] = 1
    rgb[foreground > 0] = [160, 90, 70]
    mask[foreground > 0] = 0
    output, report, _ = pipeline.solve_scene(
        rgb, mask, template, foreground_mask=foreground, selected_candidate=quad
    )
    np.testing.assert_array_equal(output[foreground > 0], rgb[foreground > 0])
    assert report["foreground_changed_pixels"] == 0
    assert report["foreground_evidence"]["ground_truth"] is False


def test_invalid_guidance_is_unresolved_not_a_successful_original():
    rgb, mask, template, _ = scene()
    output, report, _ = pipeline.solve_scene(
        rgb, mask, template, geometry_guidance={"coordinate_space": "crop"}
    )
    assert report["status"] == "unresolved"
    assert report["unresolved_reason"] == "geometry_input_or_estimation_error"
    assert "source pixels" in report["error"]
    np.testing.assert_array_equal(output, rgb)


def test_declared_support_violation_is_not_hidden(monkeypatch):
    rgb, mask, template, quad = scene()

    def broken(rgb, mask, quad, template, **kwargs):
        output = rgb.copy()
        output[0, 0] = 255
        kwargs["edit_support_out"].update(
            bbox_xyxy=[30, 25, 150, 65],
            support=np.ones((40, 120), bool),
            protected=np.zeros((40, 120), bool),
        )
        return output, {}

    monkeypatch.setattr(pipeline, "replace_surface", broken)
    with pytest.raises(RuntimeError, match="invariants"):
        pipeline.solve_scene(rgb, mask, template, selected_candidate=quad)


def test_explicitly_missing_trajectory_candidate_cannot_restart_selection(monkeypatch):
    rgb, mask, template, _ = scene()

    def forbidden(*args, **kwargs):
        raise AssertionError("An unresolved trajectory must not be reselected")

    monkeypatch.setattr(pipeline, "propose_scene_geometry", forbidden)
    output, report, candidates = pipeline.solve_scene(
        rgb, mask, template, selected_candidate=None
    )
    assert report["status"] == "unresolved"
    assert report["unresolved_reason"] == "trajectory_geometry_unresolved"
    assert not candidates
    np.testing.assert_array_equal(output, rgb)


def test_scene_pipeline_forwards_temporal_appearance_filter():
    rgb, mask, template, quad = scene()
    calls = []

    def parameter_filter(gain, black, tint, levels):
        calls.append(levels.copy())
        return gain * 0.5, black, tint, {"test_temporal_filter": True}

    _, report, _ = pipeline.solve_scene(
        rgb,
        mask,
        template,
        selected_candidate=quad,
        appearance_parameter_filter=parameter_filter,
    )
    assert len(calls) == 1
    assert report["appearance_parameter_filter_supplied"]
    assert report["compositor"]["appearance"]["temporal_parameters"][
        "test_temporal_filter"
    ]


def test_video_filters_run_before_ink_fit_and_keep_visibility_immutable():
    from dataclasses import replace

    rgb, mask, template, quad = scene()
    before = mask.copy()
    calls = []

    def focus(estimate, **context):
        assert 0 < context["visible_fraction"] <= 1
        calls.append("focus")
        return replace(estimate, sigma_px=2.0)

    def material(estimate, levels, **context):
        calls.append("material")
        assert context["source_reference"].shape == template.rgb.shape
        assert context["valid_mask"].shape == template.alpha.shape
        return replace(estimate, gain=0.25)

    _, report, _ = pipeline.solve_scene(
        rgb,
        mask,
        template,
        selected_candidate=quad,
        degradation_estimate_filter=focus,
        appearance_estimate_filter=material,
    )
    assert calls == ["focus", "material"]
    assert report["rendered"]
    assert report["degradation_estimate_filter_supplied"]
    assert report["compositor"]["blur_sigma_px"] == 2.0
    np.testing.assert_array_equal(mask, before)
