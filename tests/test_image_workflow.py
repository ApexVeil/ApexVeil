import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apexveil.backends.image import Detection
from apexveil.diagnostic_export import stable_color
from apexveil.workflows import image as workflow


def fixture(tmp_path):
    rgb = np.full((120, 360, 3), 60, np.uint8)
    masks = {0: np.zeros(rgb.shape[:2], np.uint8), 1: np.zeros(rgb.shape[:2], np.uint8)}
    masks[0][25:75, 25:185] = 1
    masks[1][80:105, 240:345] = 1
    for mask in masks.values():
        rgb[mask > 0] = [35, 145, 65]
    rgb[80:105:2, 240:345:2] = [220, 220, 220]
    foreground = np.zeros(rgb.shape[:2], np.uint8)
    foreground[20:90, 14:28] = 1
    rgb[foreground > 0] = [140, 60, 85]
    masks[0][foreground > 0] = 0
    source = tmp_path / "source.png"
    Image.fromarray(rgb).save(source)
    rgba = np.full((50, 180, 4), 255, np.uint8)
    rgba[:, :, :3] = 12
    rgba[20:30, 60:120, :3] = 200
    template = tmp_path / "asset.png"
    Image.fromarray(rgba).save(template)

    class Segmenter:
        def __init__(self):
            self.calls = []
            self.metadata = {"load_seconds": 7.5, "backend": "already-loaded-test"}

        def detect(self, image, prompt="license plate", refine=True):
            self.calls.append(prompt)
            selected = masks if prompt == "license plate" else {4: foreground}
            return [
                Detection(oid, 0.9, mask.copy(), {"refined": True})
                for oid, mask in selected.items()
            ], {"prompt": prompt}

    return source, template, Segmenter(), rgb, masks, foreground


def fake_scene(rgb, mask, template, **kwargs):
    changed = mask > 0
    fg = kwargs.get("foreground_mask")
    if fg is not None:
        changed &= ~fg
    result = rgb.copy()
    result[changed] = 18
    ys, xs = np.where(mask > 0)
    quad = [
        [int(xs.min()), int(ys.min())],
        [int(xs.max()), int(ys.min())],
        [int(xs.max()), int(ys.max())],
        [int(xs.min()), int(ys.max())],
    ]
    return (
        result,
        {
            "rendered": True,
            "status": "rendered_unreviewed",
            "selected_candidate": {"quad": quad},
            "foreground_changed_pixels": 0,
        },
        [],
    )


def test_prepare_preserves_native_masks_and_matches_public_palette(tmp_path):
    source, _, segmenter, rgb, masks, _ = fixture(tmp_path)
    events = []
    prepared = workflow.prepare_image(
        source,
        tmp_path / "prepare",
        segmenter,
        progress=lambda message, fraction: events.append((message, fraction)),
    )
    assert segmenter.calls == ["license plate"]
    assert prepared["private"]["segmenter"] is segmenter
    assert prepared["public"]["width"] == rgb.shape[1]
    assert prepared["public"]["targets"][0]["color_rgb"] == list(
        stable_color(0, workflow.SCOPE)
    )
    np.testing.assert_array_equal(prepared["private"]["masks"][0], masks[0])
    json.dumps(prepared["public"])
    assert events[-1][1] == 1
    assert prepared["timings"]["model_metadata_load_seconds"] == 7.5


def test_automatic_replace_reuses_person_detector_and_blurs_other_targets(
    tmp_path, monkeypatch
):
    source, template, segmenter, rgb, masks, foreground = fixture(tmp_path)
    monkeypatch.setattr(workflow, "solve_scene", fake_scene)
    prepared = workflow.prepare_image(source, tmp_path / "prepare", segmenter)
    result = workflow.render_image(prepared, template, tmp_path / "render", target_id=0)
    output = np.array(Image.open(result["output_path"]))
    assert segmenter.calls == ["license plate", "person"]
    assert result["model_calls"] == 0
    np.testing.assert_array_equal(output[foreground > 0], rgb[foreground > 0])
    assert np.any(output[masks[1] > 0] != rgb[masks[1] > 0])
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["model_loaded_by_workflow"] is False
    assert report["foreground"]["ground_truth"] is False
    assert report["other_redacted_ids"] == [1]
    assert Path(result["detail_diagnostic_path"]).is_file()
    assert report["output_encoding"] == {
        "format": "PNG",
        "lossless": True,
        "compress_level": 3,
    }
    for stage in (
        "png_export_seconds",
        "diagnostic_export_seconds",
        "other_redaction_and_merge_seconds",
    ):
        assert 0 <= report["timings"][stage] <= report["timings"]["render_wall_seconds"]


def test_redaction_does_not_invoke_person_or_advisor(tmp_path):
    source, _, segmenter, _, _, _ = fixture(tmp_path)
    prepared = workflow.prepare_image(source, tmp_path / "prepare", segmenter)
    result = workflow.render_image(
        prepared,
        None,
        tmp_path / "render",
        target_id=1,
        mode="mosaic",
        advisor_profile="luna-medium",
    )
    assert result["model_calls"] == 0
    assert segmenter.calls == ["license plate"]


def test_one_advisor_call_uses_exact_profile_source_bound_evidence_and_rerenders(
    tmp_path, monkeypatch
):
    source, template, segmenter, _, _, _ = fixture(tmp_path)
    calls = []
    renders = []

    def scene(*args, **kwargs):
        renders.append(kwargs)
        return fake_scene(*args, **kwargs)

    monkeypatch.setattr(workflow, "solve_scene", scene)

    class Advisor:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, directory, prompt, images):
            assert "CONTEXT:" in prompt
            for path in images:
                assert Path(path).is_file()
                assert Path(path).is_relative_to(tmp_path / "render")
            context = json.loads(prompt.split("CONTEXT:\n", 1)[1])
            assert context["observed_source_light_samples"] is not None
            appearance = {
                key: None
                for key in [
                    "exposure_gain_linear",
                    "additive_rgb_linear",
                    "flare_control_points",
                    "shading_control_points",
                    "diffuse_tint_rgb",
                ]
            }
            appearance["exposure_gain_linear"] = 0.3
            return {
                "plate_confirmed": True,
                "coordinate_space": "crop_pixels",
                "geometry": {
                    "edge_labels": {
                        s: "real_plate" for s in ["top", "right", "bottom", "left"]
                    },
                    "physical_aspect_guess": None,
                    "candidate_quads": [],
                    "assumptions": [],
                },
                "appearance": appearance,
                "assessment": {
                    "acceptable": False,
                    "issues": ["shown baseline is dark"],
                    "next_action": "rerender",
                },
            }, {"status": "proposal_available", "requested_model": "gpt-6.1-sol"}

    monkeypatch.setattr(workflow, "CodexSceneAdvisor", Advisor)
    prepared = workflow.prepare_image(source, tmp_path / "prepare", segmenter)
    result = workflow.render_image(
        prepared,
        template,
        tmp_path / "render",
        target_id=0,
        advisor_profile="sol-medium",
        protect_foreground=False,
    )
    assert len(calls) == 1 and calls[0]["profile"] == "sol-medium"
    assert (
        len(renders) == 2
        and renders[-1]["appearance_guidance"]["exposure_gain_linear"] == 0.3
    )
    assert result["model_calls"] == 1
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["final_quality_accepted"] is None
    assert report["advisor_assessment_scope"] == "shown_baseline_only"


def test_unresolved_replace_never_returns_successful_original(tmp_path, monkeypatch):
    source, template, segmenter, _, _, _ = fixture(tmp_path)
    prepared = workflow.prepare_image(source, tmp_path / "prepare", segmenter)
    monkeypatch.setattr(
        workflow,
        "solve_scene",
        lambda rgb, *args, **kwargs: (
            rgb.copy(),
            {
                "rendered": False,
                "unresolved_reason": "no_supported_full_plane_candidate",
            },
            [],
        ),
    )
    with pytest.raises(ValueError, match="Replacement unresolved"):
        workflow.render_image(
            prepared,
            template,
            tmp_path / "render",
            target_id=0,
            protect_foreground=False,
        )
    assert not (tmp_path / "render/result.png").exists()
    with pytest.raises(ValueError, match="Selected target"):
        workflow.render_image(prepared, template, tmp_path / "bad", target_id=99)


def test_lossless_export_preserves_all_pixels_size_and_icc(tmp_path, monkeypatch):
    source, template, segmenter, rgb, masks, foreground = fixture(tmp_path)
    monkeypatch.setattr(workflow, "solve_scene", fake_scene)
    prepared = workflow.prepare_image(source, tmp_path / "prepare", segmenter)
    prepared["private"]["icc_profile"] = b"unchanged-source-profile-bytes"
    target_result, _, _ = fake_scene(
        rgb, masks[0], None, foreground_mask=foreground > 0
    )
    expected, _ = workflow.redact_surface(rgb, masks[1], "blur")
    changed = np.any(target_result != rgb, axis=2)
    expected[changed] = target_result[changed]
    expected[foreground > 0] = rgb[foreground > 0]
    legacy_png = tmp_path / "legacy.png"
    Image.fromarray(expected).save(
        legacy_png, icc_profile=prepared["private"]["icc_profile"]
    )
    result = workflow.render_image(prepared, template, tmp_path / "render", target_id=0)
    with (
        Image.open(legacy_png) as legacy,
        Image.open(result["output_path"]) as exported,
    ):
        assert exported.size == legacy.size == (rgb.shape[1], rgb.shape[0])
        assert exported.info["icc_profile"] == legacy.info["icc_profile"]
        np.testing.assert_array_equal(np.array(exported), np.array(legacy))


def test_non_target_overlap_preserves_unchanged_selected_pixels(tmp_path, monkeypatch):
    source, template, segmenter, rgb, masks, _ = fixture(tmp_path)
    masks[1] |= masks[0]
    rgb[35:65:2, 40:180:2] = 220
    Image.fromarray(rgb).save(source)
    expected_target = rgb.copy()
    expected_target[45:55, 60:80] = 20

    def partial_scene(*args, **kwargs):
        return (
            expected_target.copy(),
            {"rendered": True, "status": "rendered_unreviewed"},
            [],
        )

    monkeypatch.setattr(workflow, "solve_scene", partial_scene)
    prepared = workflow.prepare_image(source, tmp_path / "prepare", segmenter)
    result = workflow.render_image(
        prepared,
        template,
        tmp_path / "render",
        target_id=0,
        protect_foreground=False,
    )
    with Image.open(result["output_path"]) as image:
        exported = np.asarray(image)
    np.testing.assert_array_equal(exported[masks[0] > 0], expected_target[masks[0] > 0])
    other_only = (masks[1] > 0) & ~(masks[0] > 0)
    assert np.any(exported[other_only] != rgb[other_only])


def test_evidence_distinguishes_existing_source_quads_from_advised_crop_quads(tmp_path):
    from apexveil.workflows.evidence import build_scene_evidence

    _, template, _, rgb, masks, _ = fixture(tmp_path)
    mask = masks[1]
    quad = [[240, 80], [344, 80], [344, 104], [240, 104]]
    evidence = build_scene_evidence(
        rgb,
        mask,
        template,
        tmp_path / "evidence",
        scene_report={"selected_candidate": {"quad": quad}},
    )
    numeric = evidence["manifest"]["numeric_context"]
    assert numeric["crop_xyxy"][0] > 0
    assert "advisor output" in numeric["geometry_coordinate_space"]
    assert "full source pixels" in numeric["current_scene_geometry_coordinate_space"]
    assert (
        numeric["current_scene_is_not_ground_truth"]["selected_candidate"]["quad"]
        == quad
    )
    assert "subtract the crop origin" in evidence["prompt"]
