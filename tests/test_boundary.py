import numpy as np

from apexveil.imaging.appearance import AppearanceEstimate
from apexveil.imaging.boundary import estimate_boundary
from apexveil.imaging.compositor import Template, replace_surface
from apexveil.imaging.degradation import DegradationEstimate


def scene():
    rgb = np.full((120, 240, 3), [40, 40, 40], np.uint8)
    rgb[30:90, 30:210] = [40, 160, 80]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[31:89, 31:209] = 1
    quad = np.array([[30, 30], [209, 30], [209, 89], [30, 89]], np.float32)
    return rgb, mask, quad


def test_color_supported_boundary_expands_into_missed_surface_and_not_background():
    rgb, mask, quad = scene()
    estimate = estimate_boundary(rgb, mask, quad)
    x0, y0, x1, y1 = estimate.bbox_xyxy
    full = np.zeros(mask.shape, bool)
    full[y0:y1, x0:x1] = estimate.support
    assert full[30, 100] and full[89, 100]
    assert not full[29, 100]
    assert estimate.diagnostics["expanded_pixels"] > 0


def test_declared_foreground_connected_to_exterior_remains_protected():
    rgb, mask, quad = scene()
    rgb[20:70, 90:104] = [175, 90, 60]
    mask[20:70, 90:104] = 0
    foreground = np.zeros(mask.shape, np.uint8)
    foreground[20:70, 90:104] = 1
    estimate = estimate_boundary(rgb, mask, quad, protected_mask=foreground)
    x0, y0, _x1, _y1 = estimate.bbox_xyxy
    assert estimate.protected[50 - y0, 95 - x0]
    assert not estimate.support[30 - y0, 95 - x0]


def test_declared_support_and_protected_foreground_are_exact_invariants():
    rgb, mask, quad = scene()
    rgb[20:70, 90:104] = [175, 90, 60]
    mask[20:70, 90:104] = 0
    protected = np.zeros(mask.shape, np.uint8)
    protected[20:70, 90:104] = 1
    alpha = np.ones((60, 180), np.float32)
    # Rounded transparent corner must expose reconstructed background, not green.
    alpha[:6, :6] = 0
    template = Template(np.full((60, 180, 3), 15, np.uint8), alpha, {})
    appearance = AppearanceEstimate(
        1,
        np.zeros((60, 180, 3), np.float32),
        np.ones(3),
        np.ones((60, 180), np.float32),
        {},
    )
    degradation = DegradationEstimate(0, 0, 0, {})
    support = {}
    output, report = replace_surface(
        rgb,
        mask,
        quad,
        template,
        0,
        protected_mask=protected,
        foreground_mask=protected,
        appearance_estimate=appearance,
        degradation_estimate=degradation,
        edit_support_out=support,
    )
    x0, y0, x1, y1 = support["bbox_xyxy"]
    declared = np.zeros(mask.shape, bool)
    declared[y0:y1, x0:x1] = support["support"]
    assert np.array_equal(output[~declared], rgb[~declared])
    assert np.array_equal(output[protected > 0], rgb[protected > 0])
    assert output[33, 33, 1] < 100
    assert report["corner_cleanup_pixels"] > 0


def test_near_edge_enclosed_hole_cannot_be_expanded_despite_same_surface_color():
    rgb, mask, quad = scene()
    mask[50:60, 32:34] = 0
    hole = np.zeros(mask.shape, bool)
    hole[50:60, 32:34] = True
    estimate = estimate_boundary(rgb, mask, quad, blur_sigma=0)
    x0, y0, x1, y1 = estimate.bbox_xyxy
    local_hole = hole[y0:y1, x0:x1]
    assert estimate.protected[local_hole].all()
    assert not estimate.support[local_hole].any()
    assert estimate.diagnostics["enclosed_missing_pixels"] == 20
    assert estimate.diagnostics["needs_review"]
    template = Template(
        np.full((60, 180, 3), 15, np.uint8), np.ones((60, 180), np.float32), {}
    )
    appearance = AppearanceEstimate(
        1,
        np.zeros((60, 180, 3), np.float32),
        np.ones(3),
        np.ones((60, 180), np.float32),
        {},
    )
    degradation = DegradationEstimate(0, 0, 0, {})
    output, _ = replace_surface(
        rgb,
        mask,
        quad,
        template,
        0,
        appearance_estimate=appearance,
        degradation_estimate=degradation,
    )
    np.testing.assert_array_equal(output[hole], rgb[hole])


def test_default_blur_uses_injected_zero_degradation_without_legacy_focus_floor():
    from dataclasses import replace

    from apexveil.estimation import get_default_estimators

    class ZeroDegradation:
        def estimate(self, rgb, mask, quad):
            return DegradationEstimate(0, 0, 0, {"provider": "zero-test"})

    rgb, mask, quad = scene()
    template = Template(
        np.full((60, 180, 3), 15, np.uint8), np.ones((60, 180), np.float32), {}
    )
    estimators = replace(get_default_estimators(), degradation=ZeroDegradation())
    _, detail = replace_surface(rgb, mask, quad, template, estimators=estimators)
    assert detail["blur_sigma_px"] == 0


def test_explicit_numeric_blur_override_is_never_raised_to_estimated_sigma():
    rgb, mask, quad = scene()
    template = Template(
        np.full((60, 180, 3), 15, np.uint8), np.ones((60, 180), np.float32), {}
    )
    degradation = DegradationEstimate(5, 0, 0, {})
    for override in (0, 0.35, 2):
        _, detail = replace_surface(
            rgb, mask, quad, template, override, degradation_estimate=degradation
        )
        assert detail["blur_sigma_px"] == override
