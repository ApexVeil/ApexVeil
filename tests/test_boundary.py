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


def test_foreground_hole_and_branch_connected_to_exterior_remain_protected():
    rgb, mask, quad = scene()
    rgb[20:70, 90:104] = [175, 90, 60]
    mask[20:70, 90:104] = 0
    estimate = estimate_boundary(rgb, mask, quad)
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


def test_soft_outer_correction_does_not_soften_interior_occlusion_visibility():
    rgb, mask, quad = scene()
    mask[45:70, 100:110] = 0
    estimate = estimate_boundary(rgb, mask, quad, blur_sigma=1.8)
    x0, y0, _, _ = estimate.bbox_xyxy
    assert 0 < estimate.coverage[31 - y0, 100 - x0] < 1
    assert estimate.coverage[55 - y0, 99 - x0] == 1
    assert estimate.coverage[55 - y0, 105 - x0] == 0
    assert np.all(estimate.coverage[estimate.protected] == 0)
    assert np.all(estimate.coverage[~estimate.support] == 0)


def test_sparse_outer_support_does_not_receive_full_strength_correction():
    from dataclasses import replace

    rgb, mask, quad = scene()
    mask[29, 50] = 1
    estimate = estimate_boundary(rgb, mask, quad, blur_sigma=1.8)
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
    degradation = DegradationEstimate(1.8, 0, 0, {})
    hard, _ = replace_surface(
        rgb,
        mask,
        quad,
        template,
        1.8,
        boundary_estimate=replace(estimate, coverage=None),
        appearance_estimate=appearance,
        degradation_estimate=degradation,
    )
    soft, _ = replace_surface(
        rgb,
        mask,
        quad,
        template,
        1.8,
        boundary_estimate=estimate,
        appearance_estimate=appearance,
        degradation_estimate=degradation,
    )
    hard_delta = np.abs(hard[29, 50].astype(float) - rgb[29, 50]).mean()
    soft_delta = np.abs(soft[29, 50].astype(float) - rgb[29, 50]).mean()
    assert hard_delta > 0
    assert soft_delta < hard_delta * 0.8
