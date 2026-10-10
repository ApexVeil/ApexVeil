import cv2
import numpy as np

from apexveil.imaging.appearance import AppearanceEstimate, estimate_appearance
from apexveil.imaging.color import linear, srgb
from apexveil.imaging.compositor import Template, replace_surface
from apexveil.imaging.degradation import DegradationEstimate
from apexveil.imaging.scene_appearance import apply_scene_guidance


def test_spatial_rgb_flare_is_not_replaced_by_global_context_dark_tint():
    h, w = 120, 480
    face = np.full((h, w, 3), [0.08, 0.3, 0.1], np.float32)
    for x in range(30, 450, 55):
        cv2.rectangle(face, (x, 20), (x + 12, 100), (0.002, 0.002, 0.002), -1)
    veil = np.zeros_like(face)
    veil[:, :, 0] = np.linspace(0.05, 0.02, w)
    veil[:, :, 1] = 0.007
    veil[:, :, 2] = 0.003
    estimate = estimate_appearance(
        np.zeros((h, w, 3), np.uint8), srgb(face * 0.35 + veil), context_tint=np.ones(3)
    )
    assert estimate.black[:, :, :1].mean() > estimate.black[:, :, 1:2].mean() * 2
    assert (
        estimate.black[:, 30:100, 0].mean() > estimate.black[:, 350:430, 0].mean() * 1.4
    )


def test_guidance_produces_spatial_veil_separate_from_material_shading():
    asset = np.zeros((30, 120, 3), np.uint8)
    asset[10:20, 45:75] = 220
    original = asset.copy()
    base = AppearanceEstimate(
        0.5, np.zeros(asset.shape, np.float32), np.ones(3), np.ones(asset.shape[:2]), {}
    )
    guided = apply_scene_guidance(
        base,
        {
            "flare_control_points": [
                {"xy": [0, 0.5], "rgb_linear": [0.05, 0.01, 0.003]},
                {"xy": [1, 0.5], "rgb_linear": [0.003, 0.001, 0.001]},
            ],
            "provenance": "known synthetic veil",
        },
    )
    result = linear(guided.render(asset))
    assert result[5, 0, 0] > result[5, -1, 0] * 10
    np.testing.assert_array_equal(asset, original)
    np.testing.assert_array_equal(guided.spatial_field, base.spatial_field)


def test_authoritative_foreground_and_old_removal_are_independent_of_template_alpha():
    rgb = np.full((100, 240, 3), 40, np.uint8)
    rgb[25:75, 30:210] = [30, 160, 80]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[25:75, 30:210] = 1
    mask[40:50, 31:33] = 0
    quad = np.array([[30, 25], [209, 25], [209, 74], [30, 74]], np.float32)
    fg = np.zeros(mask.shape, np.uint8)
    fg[40:50, 100:108] = 1
    rgb[fg > 0] = [140, 60, 85]
    alpha = np.ones((50, 180), np.float32)
    alpha[:3, :3] = 0
    template = Template(np.full((50, 180, 3), 15, np.uint8), alpha.copy(), {})
    estimate = AppearanceEstimate(
        1, np.zeros((50, 180, 3), np.float32), np.ones(3), np.ones((50, 180)), {}
    )
    support = {}
    out, _ = replace_surface(
        rgb,
        mask,
        quad,
        template,
        0,
        foreground_mask=fg,
        appearance_estimate=estimate,
        degradation_estimate=DegradationEstimate(0, 0, 0, {}),
        edit_support_out=support,
    )
    np.testing.assert_array_equal(out[fg > 0], rgb[fg > 0])
    assert out[45, 32, 1] < 30  # a known non-foreground native mask hole is removed
    assert out[65, 140, 1] < 30
    np.testing.assert_array_equal(template.alpha, alpha)


def test_guided_exposure_fit_spike_is_filtered_after_guidance():
    from apexveil.imaging.photometry import TemporalPhotometry

    rgb = np.full((100, 240, 3), 70, np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[25:75, 30:210] = 1
    quad = np.array([[30, 25], [209, 25], [209, 74], [30, 74]], np.float32)
    template = Template(
        np.full((50, 180, 3), 150, np.uint8), np.ones((50, 180), np.float32), {}
    )
    state = TemporalPhotometry(30)
    replace_surface(
        rgb,
        mask,
        quad,
        template,
        0,
        scene_guidance={"exposure_gain_linear": 0.2},
        appearance_parameter_filter=state.filter,
    )
    _, detail = replace_surface(
        rgb,
        mask,
        quad,
        template,
        0,
        scene_guidance={"exposure_gain_linear": 0.8},
        appearance_parameter_filter=state.filter,
    )
    assert detail["appearance"]["exposure_gain_linear"] < 0.22
    assert (
        detail["appearance"]["temporal_parameters"]["raw_exposure_gain_linear"] == 0.8
    )
