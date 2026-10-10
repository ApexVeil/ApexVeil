"""Behavioral provider contract, with no model downloads or GPU imports."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from apexveil.backends.image import Detection
from apexveil.estimation import get_default_estimators
from apexveil.image_pipeline import prepare_surfaces, render
from apexveil.imaging.compositor import Template, replace_surface


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def estimate(self, *args, **kwargs):
        self.calls += 1
        return self.inner.estimate(*args, **kwargs)


def scene():
    image = np.full((110, 230, 3), 100, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[35:76, 40:191] = 1
    image[mask > 0] = [80, 150, 90]
    image[44:67, 100:105] = 20
    template = Template(
        np.full((40, 150, 3), 30, np.uint8), np.ones((40, 150), np.float32), {}
    )
    return image, mask, template


def test_product_pipeline_injects_all_estimators_and_declares_support(tmp_path):
    default = get_default_estimators()
    wrappers = {
        key: Recorder(getattr(default, key))
        for key in ("geometry", "boundary", "degradation", "appearance")
    }
    providers = replace(default, **wrappers, provider_id="contract-fixture")
    image, mask, template = scene()
    detections = [Detection(0, 0.9, mask, {})]
    surfaces = prepare_surfaces(
        image, detections, tmp_path, estimators=providers, template_aspect_ratio=3.75
    )
    support = {}
    output, report = render(
        image,
        detections,
        surfaces,
        "replace",
        template,
        estimators=providers,
        edit_support_out=support,
    )
    assert all(x.calls == 1 for x in wrappers.values())
    assert report["estimators"]["provider_id"] == "contract-fixture"
    assert report["outside_declared_edit_support_changed_pixels"] == 0
    assert np.array_equal(image[~support["support"]], output[~support["support"]])
    assert np.any(image != output)
    json.dumps(report)


def test_foreground_is_immutable_with_injected_boundary_and_optical_blur():
    image, mask, template = scene()
    protected = np.zeros(mask.shape, bool)
    protected[38:71, 108:123] = True
    mask[protected] = 0
    quad = np.array([[40, 35], [190, 35], [190, 75], [40, 75]], np.float32)
    support = {}
    output, _ = replace_surface(
        image,
        mask,
        quad,
        template,
        blur_sigma=3,
        protected_mask=protected,
        edit_support_out=support,
    )
    assert np.array_equal(output[protected], image[protected])
    whole = np.zeros(mask.shape, bool)
    x0, y0, x1, y1 = support["bbox_xyxy"]
    whole[y0:y1, x0:x1] = support["support"]
    assert np.array_equal(output[~whole], image[~whole])


def test_provider_import_does_not_load_neural_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import apexveil.estimation; assert 'torch' not in sys.modules; assert 'sam3' not in sys.modules",
        ],
        check=True,
    )


def test_invalid_provider_fails_before_processing():
    with pytest.raises(TypeError, match="geometry"):
        replace(get_default_estimators(), geometry=object())
    with pytest.raises(ValueError, match="provider_id"):
        replace(get_default_estimators(), provider_id="")


def test_injected_zero_blur_is_consistent_in_image_video_and_diagnostic():
    from apexveil.imaging.geometry import SurfaceGeometry
    from apexveil.imaging.photometry import TemporalPhotometry
    from apexveil.video_pipeline import render_surface_pair
    from scripts.evaluate_vnext import _render_replacement

    image, mask, template = scene()
    quad = np.array([[40, 35], [190, 35], [190, 75], [40, 75]], np.float32)
    default = get_default_estimators()

    class ZeroBlur:
        def estimate(self, rgb, visible, q):
            return replace(default.degradation.estimate(rgb, visible, q), sigma_px=0.0)

    providers = replace(
        default, degradation=ZeroBlur(), provider_id="zero-blur-contract"
    )
    observation = SurfaceGeometry(quad, mask, 1.0, [], {})
    _, image_report = render(
        image,
        [Detection(0, 0.9, mask, {})],
        [{"object_id": 0, "quad": quad.tolist(), "flags": []}],
        "replace",
        template,
        estimators=providers,
    )
    video_result = render_surface_pair(
        image,
        mask,
        observation,
        observation,
        template,
        providers,
        TemporalPhotometry(30),
        9176,
    )
    _, diagnostic_report, _ = _render_replacement(
        image, mask, quad, template, providers, TemporalPhotometry(30), 9176
    )
    assert image_report["edits"][0]["blur_sigma_px"] == 0
    assert video_result[3]["blur_sigma_px"] == 0
    assert diagnostic_report["blur_sigma_px"] == 0
