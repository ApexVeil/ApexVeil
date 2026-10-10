import cv2
import numpy as np
import pytest

from apexveil.imaging.geometry import estimate_surface_geometry, template_geometry_flags
from apexveil.imaging.offline_geometry import refine_shot_geometry


def surface():
    mask = np.zeros((200, 400), np.uint8)
    quad = np.array([[70, 60], [290, 60], [290, 140], [70, 140]], np.float32)
    cv2.fillConvexPoly(mask, quad.astype(int), 255)
    image = np.full((200, 400, 3), 30, np.uint8)
    texture = np.random.default_rng(918).integers(40, 220, image.shape, dtype=np.uint8)
    image[mask > 0] = texture[mask > 0]
    return image, mask, quad


def sequence(offsets):
    image, mask, q = surface()
    frames, masks, truth = [], [], []
    for dx in offsets:
        h = np.array(
            [[1, 0.0005 * dx, dx], [0, 1, 0.05 * dx], [0.00001 * dx, 0, 1]], float
        )
        frames.append(cv2.warpPerspective(image, h, (400, 200)))
        masks.append(cv2.warpPerspective(mask, h, (400, 200), flags=cv2.INTER_NEAREST))
        truth.append(cv2.perspectiveTransform(q[None], h)[0])
    return frames, masks, truth


def test_foreshortened_projective_surface_has_no_artwork_aspect_gate():
    q = np.array([[60, 40], [145, 60], [139, 140], [65, 150]], np.float32)
    mask = np.zeros((200, 240), np.uint8)
    cv2.fillConvexPoly(mask, q.astype(int), 255)
    result = estimate_surface_geometry(None, mask, 4.7)
    assert template_geometry_flags(q, 4.7) == []
    assert result.quad is not None
    assert result.confidence > 0.6
    assert np.max(np.linalg.norm(result.quad - q, axis=1)) < 3


def test_missing_corner_uses_supported_lines_and_keeps_occlusion():
    _, mask, truth = surface()
    cv2.fillConvexPoly(mask, np.array([[255, 60], [291, 60], [291, 90]]), 0)
    mask[85:115, 160:185] = 0
    before = mask.copy()
    result = estimate_surface_geometry(None, mask, 3)
    assert result.quad is not None
    assert np.max(np.linalg.norm(result.quad - truth, axis=1)) < 3
    np.testing.assert_array_equal(result.visible_mask, before)


def test_single_frame_clipping_is_not_a_complete_plane():
    image, mask, _ = surface()
    mask[:, :80] = 0
    mask = mask[:, 80:]
    result = estimate_surface_geometry(image[:, 80:], mask, 3)
    assert result.quad is None
    assert "partially_out_of_frame" in result.flags


def test_accelerating_projective_trajectory_has_no_screen_space_lag():
    frames, masks, truth = sequence([0, 2, 6, 12, 20, 30, 42])
    result = refine_shot_geometry(frames, masks, 3, observed_quads=truth)
    for item, q in zip(result, truth):
        assert item.quad is not None
        assert np.max(np.linalg.norm(item.quad - q, axis=1)) < 2
    assert any(len(item.diagnostics["anchors"]) > 2 for item in result)


def test_corrupt_corner_does_not_contaminate_trajectory():
    frames, masks, truth = sequence([0, 2, 4, 6, 8, 10, 12])
    observations = [q.copy() for q in truth]
    observations[3][1] += [30, -25]
    result = refine_shot_geometry(frames, masks, 3, observed_quads=observations)
    assert result[3].quad is not None
    assert np.max(np.linalg.norm(result[3].quad - truth[3], axis=1)) < 2
    assert "temporally_supported_full_plane" in result[3].flags


def test_temporal_evidence_recovers_clipped_plane_bidirectionally():
    frames, masks, truth = sequence([-85, -75, -65, -55, -45, -35])
    result = refine_shot_geometry(frames, masks, 3, observed_quads=truth)
    assert result[0].quad is not None
    assert "temporally_supported_full_plane" in result[0].flags
    assert any(a["direction"] == "backward" for a in result[0].diagnostics["anchors"])
    assert np.max(np.linalg.norm(result[0].quad - truth[0], axis=1)) < 3


def test_cut_and_empty_mask_break_all_anchor_paths():
    frames, masks, truth = sequence([0, 1, 2, 3, 4, 5])
    masks[2] = np.zeros_like(masks[2])
    result = refine_shot_geometry(
        frames, masks, 3, cut_indices=[4], observed_quads=truth
    )
    assert result[2].quad is None
    assert all(a["frame"] < 2 for a in result[1].diagnostics["anchors"])
    assert all(a["frame"] >= 4 for a in result[4].diagnostics["anchors"])
    assert result[4].diagnostics["incoming_motion"]["status"] == "explicit_cut"


def test_total_occlusion_never_becomes_a_mask_union():
    frames, masks, truth = sequence([0, 1, 2])
    masks[1][:] = 0
    originals = [m.copy() for m in masks]
    result = refine_shot_geometry(frames, masks, 3, observed_quads=truth)
    assert result[1].quad is None
    for item, original in zip(result, originals):
        np.testing.assert_array_equal(item.visible_mask, original)


def test_invalid_contract_rejected():
    with pytest.raises(ValueError):
        refine_shot_geometry([], [np.zeros((2, 2), np.uint8)], 3)
    with pytest.raises(ValueError):
        estimate_surface_geometry(None, np.zeros((2, 2), np.uint8), 0)


def test_collinear_visible_pixels_return_uncertainty_instead_of_crashing():
    mask = np.zeros((40, 100), np.uint8)
    mask[20, 10:90] = 255
    result = estimate_surface_geometry(None, mask, 3)
    assert result.quad is None
    assert "degenerate_visible_surface" in result.flags


def test_repeated_offline_estimation_is_deterministic():
    frames, masks, truth = sequence([0, 2, 5])
    first = refine_shot_geometry(frames, masks, 3, observed_quads=truth)
    second = refine_shot_geometry(frames, masks, 3, observed_quads=truth)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.quad, b.quad)
        assert a.confidence == b.confidence
        assert a.flags == b.flags
