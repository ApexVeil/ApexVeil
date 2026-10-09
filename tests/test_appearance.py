"""Known-scene regression checks for relighting approximations."""

import cv2
import numpy as np
from apexveil.imaging.appearance import adapt_asset
from apexveil.imaging.color import srgb, linear


def scene():
    h, w = 120, 480
    matte = np.broadcast_to(np.array([0.08, 0.3, 0.1], np.float32), (h, w, 3)).copy()
    for x in range(30, 450, 55):
        cv2.rectangle(matte, (x, 20), (x + 12, 100), (0.002, 0.002, 0.002), -1)
    return matte


def test_colored_veil_survives_on_new_black_material_without_copying_glyphs():
    base = scene()
    h, w = base.shape[:2]
    veil = np.zeros_like(base)
    veil[:, :, 0] = np.linspace(0.05, 0.02, w)
    veil[:, :, 1] = 0.007
    veil[:, :, 2] = 0.003
    original = srgb(base * 0.35 + veil)
    result, _ = adapt_asset(np.zeros_like(original), original)
    floor = linear(result)
    assert np.mean(np.abs(floor - veil)) < 0.008
    assert floor[:, :, 0].mean() > floor[:, :, 1].mean() * 2
    # Fitted veil should not carry repeated character strokes.
    assert np.abs(np.diff(floor[:, :, 0], n=2, axis=1)).mean() < 0.0003


def test_near_black_capture_does_not_gain_an_artificial_white_logo_floor():
    original = np.zeros((120, 480, 3), np.uint8)
    original[:] = 1
    asset = np.zeros_like(original)
    asset[40:80, 160:320] = 240
    result, _ = adapt_asset(asset, original)
    assert result.max() <= 2


def test_partial_visible_region_does_not_fit_colored_surface_into_black_asset():
    original = srgb(scene())
    valid = np.zeros(original.shape[:2], np.uint8)
    valid[:, 300:] = 1
    # An occluder / filled region is deliberately colorful and irrelevant.
    original[:, :300] = [250, 25, 200]
    result, detail = adapt_asset(np.zeros_like(original), original, valid_mask=valid)
    pixels = result[valid > 0]
    assert np.max(np.ptp(pixels.astype(float), axis=1)) == 0
    assert pixels.mean() < 25
    assert detail["color_floor_reliable"] is False


def test_broad_shadow_remains_visible_on_new_surface():
    original = scene()
    original[:, 240:] *= 0.2
    asset = np.full(original.shape, 150, np.uint8)
    result, _ = adapt_asset(asset, srgb(original))
    lum = linear(result).mean(axis=2)
    assert lum[25:95, 100:180].mean() > lum[25:95, 300:380].mean() * 2


def test_neutral_surroundings_prevent_green_face_color_becoming_material_color():
    from apexveil.imaging.appearance import estimate_context_tint

    rgb = np.full((200, 600, 3), 30, np.uint8)
    mask = np.zeros((200, 600), np.uint8)
    mask[40:160, 60:540] = 1
    rgb[40:160, 60:540] = srgb(scene())
    quad = np.array([[60, 40], [539, 40], [539, 159], [60, 159]], np.float32)
    tint, _ = estimate_context_tint(rgb, mask, quad)
    result, _ = adapt_asset(
        np.zeros((120, 480, 3), np.uint8), rgb[40:160, 60:540], context_tint=tint
    )
    assert np.max(np.ptp(result.astype(float), axis=2)) == 0


def test_compression_noise_in_dark_context_does_not_become_purple_tint():
    from apexveil.imaging.appearance import estimate_context_tint

    rgb = np.full((100, 300, 3), [5, 2, 7], np.uint8)
    mask = np.zeros((100, 300), np.uint8)
    mask[30:70, 70:230] = 1
    quad = np.array([[70, 30], [229, 30], [229, 69], [70, 69]], np.float32)
    tint, _ = estimate_context_tint(rgb, mask, quad)
    np.testing.assert_allclose(tint, np.ones(3))


def test_unresolved_ink_cannot_lift_new_black_plate_to_old_green_face_brightness():
    original = np.full((100, 300, 3), [40, 130, 80], np.uint8)
    result, _ = adapt_asset(
        np.zeros_like(original),
        original,
        ink_resolved=False,
        context_black_level=0.001,
        context_tint=np.ones(3),
    )
    assert result.max() <= 8
