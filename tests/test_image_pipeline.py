import cv2
import numpy as np
import pytest
from PIL import Image

from apexveil.backends.image import Detection
from apexveil.image_pipeline import prepare_surfaces, process_image, render
from apexveil.imaging.compositor import (
    Template,
    load_template,
    redact_surface,
    replace_surface,
)
from apexveil.imaging.geometry import order_quad


def fixture_scene():
    rgb = np.random.default_rng(13).integers(0, 256, (110, 220, 3), dtype=np.uint8)
    mask = np.zeros((110, 220), np.uint8)
    mask[30:80, 35:185] = 1
    mask[43:67, 100:117] = 0  # foreground occluder, not a hole to fill
    quad = np.array([[35, 30], [184, 30], [184, 79], [35, 79]], np.float32)
    template = Template(
        np.full((50, 150, 3), 50, np.uint8), np.ones((50, 150), np.float32), {}
    )
    return rgb, mask, quad, template


@pytest.mark.parametrize("mode", ["replace", "blur", "mosaic"])
def test_every_effect_preserves_foreground_holes_and_outside_pixels(mode):
    rgb, mask, quad, template = fixture_scene()
    if mode == "replace":
        out, _ = replace_surface(rgb, mask, quad, template)
    else:
        out, _ = redact_surface(rgb, mask, mode)
    assert np.array_equal(out[mask == 0], rgb[mask == 0])
    assert np.any(out[mask > 0] != rgb[mask > 0])


def test_template_keeps_rounded_transparency_and_normalizes_solid_face(tmp_path):
    rgba = np.full((60, 180, 4), 100, np.uint8)
    rgba[:, :, 3] = 0
    rgba[5:55, 10:170, 3] = 252
    rgba[5:9, 10:14, 3] = 0
    path = tmp_path / "template.png"
    Image.fromarray(rgba).save(path)
    asset = load_template(path)
    assert asset.alpha[0, 0] == 0
    assert asset.alpha[25, 80] == 1
    assert asset.rgb.shape[:2] == (50, 160)
    with pytest.raises(ValueError):
        load_template(path, 0)


def test_diamond_quad_has_four_distinct_corners():
    q = order_quad([[0, 10], [10, 0], [20, 10], [10, 20]])
    assert len(np.unique(q, axis=0)) == 4
    assert cv2.isContourConvex(q)
    with pytest.raises(ValueError):
        order_quad([[0, 0], [0, 0], [1, 1], [1, 0]])


def test_multi_target_requires_selection_then_blurs_other_target(tmp_path):
    rgb, mask, quad, template = fixture_scene()
    second = np.zeros(mask.shape, np.uint8)
    second[5:20, 30:80] = 1
    detections = [Detection(0, 0.9, mask, {}), Detection(1, 0.8, second, {})]
    surfaces = [
        {"object_id": 0, "quad": quad.tolist(), "flags": []},
        {"object_id": 1, "flags": []},
    ]
    untouched, report = render(rgb, detections, surfaces, "replace", template)
    assert report["flags"] == ["select_primary_target"]
    assert np.array_equal(untouched, rgb)
    rendered, report = render(
        rgb, detections, surfaces, "replace", template, target_id=0
    )
    assert [e["action"] for e in report["edits"]] == ["replace", "blur"]
    assert np.any(rendered[second > 0] != rgb[second > 0])


def test_uncertain_geometry_is_not_silently_rendered():
    rgb, mask, quad, template = fixture_scene()
    detections = [Detection(0, 0.9, mask, {})]
    surfaces = [
        {"object_id": 0, "quad": quad.tolist(), "flags": ["partially_out_of_frame"]}
    ]
    untouched, report = render(rgb, detections, surfaces, "replace", template)
    assert report["status"] == "needs_review"
    assert np.array_equal(untouched, rgb)
    rendered, report = render(
        rgb, detections, surfaces, "replace", template, allow_uncertain=True
    )
    assert report["status"] == "needs_review"
    assert np.any(rendered != rgb)


def test_clipped_surface_is_flagged_without_changing_visible_mask(tmp_path):
    rgb, mask, _, _ = fixture_scene()
    mask[30:80, :100] = 1
    original_mask = mask.copy()
    surfaces = prepare_surfaces(rgb, [Detection(0, 0.9, mask, {})], tmp_path)
    assert "partially_out_of_frame" in surfaces[0]["flags"]
    assert np.array_equal(mask, original_mask)


def test_cpu_pipeline_preserves_icc_source_and_exports_diagnostics(tmp_path):
    rgb, mask, _, _ = fixture_scene()
    path = tmp_path / "input.png"
    Image.fromarray(rgb).save(path, icc_profile=b"test-profile")
    original_bytes = path.read_bytes()

    class FakeSegmenter:
        def detect(self, image, prompt, refine):
            return [Detection(0, 0.95, mask, {})], {"segmentation_seconds": 0}

    out = tmp_path / "out"
    result = process_image(path, out, FakeSegmenter(), mode="blur", comparison=True)
    assert path.read_bytes() == original_bytes
    assert result["results"]["blur"]["outside_visible_masks_changed_pixels"] == 0
    with Image.open(out / "blur.png") as image:
        assert image.info["icc_profile"] == b"test-profile"
    assert (out / "report.json").is_file()
    assert (out / "target_000_comparison.jpg").is_file()


def test_edge_softness_recovers_defocus_without_blurring_sharp_surface():
    from apexveil.imaging.focus import estimate_edge_softness

    rgb = np.full((320, 700, 3), 180, np.uint8)
    rgb[100:220, 100:600] = 25
    quad = np.array([[100, 100], [599, 100], [599, 219], [100, 219]], np.float32)
    sharp = estimate_edge_softness(rgb, quad)
    blurred = estimate_edge_softness(cv2.GaussianBlur(rgb, (0, 0), 8), quad)
    assert sharp["sigma_px"] < 2
    assert 6 < blurred["sigma_px"] < 10
    assert blurred["reliable"]


def test_defocused_replacement_still_preserves_occluder_pixels():
    rgb, mask, quad, template = fixture_scene()
    output, detail = replace_surface(rgb, mask, quad, template, blur_sigma=8)
    assert np.array_equal(output[mask == 0], rgb[mask == 0])
    assert detail["boundary"]["protected_pixels"] > 0
    assert detail["corner_cleanup_pixels"] > 0


def test_small_visible_fragment_is_not_stretched_into_full_artwork():
    rgb, mask, _, template = fixture_scene()
    quad = np.array([[35, 30], [70, 30], [70, 79], [35, 79]], np.float32)
    surfaces = [{"object_id": 0, "quad": quad.tolist(), "flags": []}]
    output, report = render(
        rgb, [Detection(0, 0.9, mask, {})], surfaces, "replace", template
    )
    assert report["status"] == "needs_review"
    assert np.array_equal(output, rgb)


def test_compositor_diagnostics_are_json_serializable():
    import json

    rgb, mask, quad, template = fixture_scene()
    _, detail = replace_surface(rgb, mask, quad, template)
    json.dumps(detail)


def test_comparison_exports_raw_sam_three_panel_diagnostics(tmp_path):
    import json

    rgb, mask, _, template = fixture_scene()
    path = tmp_path / "source.png"
    Image.fromarray(rgb).save(path)

    class Segmenter:
        def detect(self, image, prompt, refine):
            return [Detection(0, 0.9, mask, {})], {"fixture": True}

    out = tmp_path / "result"
    report = process_image(path, out, Segmenter(), template=template, comparison=True)
    assert (out / "index.html").is_file()
    assert report["diagnostics"]["provenance"]["category"] == "automatic"
    assert (
        report["results"]["replace"]["outside_declared_edit_support_changed_pixels"]
        == 0
    )
    for name in report["diagnostics"]["diagnostics"].values():
        assert (out / name).is_file()
    json.dumps(report)
