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
