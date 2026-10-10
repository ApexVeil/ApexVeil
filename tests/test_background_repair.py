import cv2
import numpy as np

from apexveil.imaging.background import estimate_background_repair


def fixture():
    rgb = np.full((120, 240, 3), 40, np.uint8)
    plane = np.zeros(rgb.shape[:2], np.uint8)
    plane[30:90, 30:210] = 1
    rgb[plane > 0] = [40, 160, 80]
    protected = np.zeros(plane.shape, bool)
    requested = np.zeros(plane.shape, bool)
    requested[30:35, 30:45] = True
    return rgb, plane, protected, requested


def test_stable_donors_reconstruct_known_background_at_transparent_corner():
    rgb, plane, protected, requested = fixture()
    result = estimate_background_repair(
        rgb, plane, plane > 0, protected, requested, blur_sigma=0.5, radius=3
    )
    assert result.diagnostics["reliable"]
    assert result.diagnostics["applied_pixels"] == int(requested.sum())
    np.testing.assert_allclose(result.rgb[requested], 40, atol=2)
    np.testing.assert_array_equal(result.rgb[~requested], rgb[~requested])


def test_narrow_dark_frame_on_bright_body_abstains_from_whole_surface_repair():
    rgb, plane, protected, requested = fixture()
    rgb[:] = 220
    # One-pixel real ridge is a donor layer, not contamination to erase.
    frame = cv2.dilate(plane, np.ones((3, 3), np.uint8)) > 0
    rgb[frame] = 20
    rgb[plane > 0] = [40, 160, 80]
    result = estimate_background_repair(
        rgb, plane, plane > 0, protected, requested, blur_sigma=0.5, radius=3
    )
    assert not result.diagnostics["reliable"]
    assert result.diagnostics["applied_pixels"] == 0
    assert result.diagnostics["suppressed_pixels"] == int(requested.sum())
    assert not result.applied_mask.any()
    np.testing.assert_array_equal(result.rgb, rgb)


def test_repair_never_replaces_explicit_foreground_or_pixels_outside_request():
    rgb, plane, protected, requested = fixture()
    rgb[28:55, 36:40] = [160, 50, 95]
    protected[28:55, 36:40] = True
    result = estimate_background_repair(
        rgb, plane, plane > 0, protected, requested, blur_sigma=1.5, radius=3
    )
    assert not result.applied_mask[protected].any()
    np.testing.assert_array_equal(result.rgb[protected], rgb[protected])
    np.testing.assert_array_equal(result.rgb[~requested], rgb[~requested])


def test_unstable_donors_do_not_chop_observed_ridge_with_ragged_visibility():
    from apexveil.imaging.appearance import AppearanceEstimate
    from apexveil.imaging.boundary import BoundaryEstimate
    from apexveil.imaging.compositor import Template, replace_surface
    from apexveil.imaging.degradation import DegradationEstimate

    rgb, plane, _, _ = fixture()
    rgb[:] = 220
    frame = cv2.dilate(plane, np.ones((3, 3), np.uint8)) > 0
    rgb[frame] = 20
    rgb[plane > 0] = 50
    rgb[33:87, 33:207] = [40, 160, 80]
    mask = plane.copy()
    mask[30, 35:205:3] = 0
    mask[50:60, 100:104] = 0
    rgb[50:60, 100:104] = [170, 70, 90]
    quad = np.array([[30, 30], [209, 30], [209, 89], [30, 89]], np.float32)
    alpha = np.ones((60, 180), np.float32)
    alpha[:3, :3] = 0
    template = Template(np.full((60, 180, 3), 15, np.uint8), alpha.copy(), {})
    appearance = AppearanceEstimate(
        1,
        np.zeros((60, 180, 3), np.float32),
        np.ones(3),
        np.ones((60, 180), np.float32),
        {},
    )
    degradation = DegradationEstimate(0.5, 0, 0, {})
    # Supply known native visibility to isolate the renderer donor fallback
    # from the independent automatic boundary-expansion heuristic.
    boundary = BoundaryEstimate(
        [0, 0, 240, 120], mask > 0, mask > 0, (plane > 0) & (mask == 0), {"band_px": 3}
    )
    output, report = replace_surface(
        rgb,
        mask,
        quad,
        template,
        appearance_estimate=appearance,
        degradation_estimate=degradation,
        boundary_estimate=boundary,
    )
    assert not report["boundary"]["background_repair"]["reliable"]
    assert report["boundary"]["boundary_replacement_complete"] is False
    np.testing.assert_array_equal(output[30:32, 35:205], rgb[30:32, 35:205])
    np.testing.assert_array_equal(output[50:60, 100:104], rgb[50:60, 100:104])
    np.testing.assert_array_equal(template.alpha, alpha)
    assert output[65, 140, 1] < 50
