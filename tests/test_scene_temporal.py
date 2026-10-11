"""Measured moving-image regressions; ground truth never enters the stabilizer."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from apexveil.imaging.scene_geometry import SceneGeometryCandidate
from apexveil.imaging.scene_temporal import stabilize_scene_trajectory


def moving_scene(count=25, *, jitter=0.0, crop=False):
    rng = np.random.default_rng(37)
    base = np.full((200, 440, 3), 28, np.uint8)
    base[70:126, 65:229] = rng.integers(35, 230, (56, 164, 3), dtype=np.uint8)
    mask = np.zeros(base.shape[:2], np.uint8)
    mask[70:126, 65:229] = 255
    quad = np.array([[65, 70], [228, 70], [228, 125], [65, 125]], np.float32)
    frames, masks, candidates, truth = [], [], [], []
    for i in range(count):
        # Deliberate acceleration and a direction reversal; a screen EMA would
        # trail this by multiple pixels. Perspective and zoom also change.
        dx = 0.29 * i**2 if crop else 34 * np.sin(i / 5) + 0.20 * i**2
        transform = np.array(
            [
                [1 + i * 0.002, i * 0.0007, dx],
                [i * -0.0003, 1 + i * 0.0015, 3 * np.sin(i / 4)],
                [i * 0.0000008, i * 0.0000005, 1],
            ],
            np.float32,
        )
        frame = cv2.warpPerspective(base, transform, (440, 200))
        native = cv2.warpPerspective(
            mask, transform, (440, 200), flags=cv2.INTER_NEAREST
        )
        actual = cv2.perspectiveTransform(quad[None], transform)[0]
        observed = (actual + rng.normal(0, jitter, (4, 2))).astype(np.float32)
        candidates.append(
            SceneGeometryCandidate(
                observed,
                0.85,
                ["synthetic_boundary_observation"],
                [],
                {
                    "eligible_for_render": True,
                    "ranking_score": 0.8,
                    "edge_labels": dict(
                        zip(("top", "right", "bottom", "left"), ["surface"] * 4)
                    ),
                },
            )
        )
        frames.append(frame)
        masks.append(native)
        truth.append(actual)
    return frames, masks, candidates, np.asarray(truth)


def test_motion_compensated_fusion_reduces_jitter_with_acceleration_and_no_lag():
    frames, masks, candidates, truth = moving_scene(jitter=1.6)
    original = np.asarray([c.quad.copy() for c in candidates])
    before_masks = np.asarray(masks).copy()
    result, report = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    actual = np.asarray([c.quad for c in result])
    raw_error = np.sqrt(np.mean((original - truth) ** 2))
    new_error = np.sqrt(np.mean((actual - truth) ** 2))
    assert report["motion_pair_count"] >= len(frames) - 2
    assert new_error < raw_error * 0.55
    # Same-time truth wins over either one-frame delay or one-frame anticipation.
    assert new_error < np.sqrt(np.mean((actual[1:] - truth[:-1]) ** 2)) * 0.35
    assert new_error < np.sqrt(np.mean((actual[:-1] - truth[1:]) ** 2)) * 0.35
    assert np.abs(np.mean(actual - truth, axis=(0, 1))).max() < 0.4
    np.testing.assert_array_equal([c.quad for c in candidates], original)
    np.testing.assert_array_equal(masks, before_masks)
    assert not report["screen_coordinate_smoothing"]


def test_precise_moving_observations_keep_instantaneous_motion_and_endpoints():
    frames, masks, candidates, truth = moving_scene()
    result, _ = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    actual = np.asarray([c.quad for c in result])
    assert np.max(np.linalg.norm(actual - truth, axis=2)) < 0.6
    np.testing.assert_allclose(actual[[0, -1]], truth[[0, -1]], atol=0.35)
    np.testing.assert_allclose(
        np.diff(actual.mean(axis=1), axis=0),
        np.diff(truth.mean(axis=1), axis=0),
        atol=0.25,
    )


def test_missing_geometry_can_recover_only_with_current_native_visibility():
    frames, masks, candidates, truth = moving_scene(count=15, jitter=0.6)
    candidates[4] = None
    candidates[5] = None
    masks[9][:] = 0
    candidates[10] = None
    masks[10][:] = 0
    result, report = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    for i in (4, 5):
        assert result[i] is not None
        assert report["frames"][i]["status"] == "motion_recovered"
        assert np.max(np.linalg.norm(result[i].quad - truth[i], axis=1)) < 1.0
        assert result[i].protected_mask is None
    assert result[9] is result[10] is None
    assert all(j < 9 for j in report["frames"][8]["support_frames"])
    assert all(j > 10 for j in report["frames"][11]["support_frames"])


def test_native_occluder_holes_and_current_protection_are_unchanged():
    frames, masks, candidates, _ = moving_scene(count=12, jitter=0.4)
    for i in (4, 5, 6):
        hole = np.zeros(masks[i].shape, np.uint8)
        hole[75:140, 155:180] = 1
        frames[i][hole > 0] = 10
        masks[i][hole > 0] = 0
        candidates[i].protected_mask = hole.astype(bool)
    before = np.asarray(masks).copy()
    protection = candidates[5].protected_mask.copy()
    result, _ = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    np.testing.assert_array_equal(masks, before)
    np.testing.assert_array_equal(result[5].protected_mask, protection)
    assert result[5].protected_mask is candidates[5].protected_mask
    assert not np.any(masks[5][protection])


def test_cropped_motion_retains_full_plane_outside_viewport():
    frames, masks, candidates, truth = moving_scene(count=31, jitter=0.5, crop=True)
    result, report = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    assert truth[-1, :, 0].max() > frames[-1].shape[1] + 50
    assert result[-1].quad[:, 0].max() > frames[-1].shape[1] + 50
    assert np.sqrt(np.mean((np.asarray([c.quad for c in result]) - truth) ** 2)) < 0.6
    assert report["motion_pair_count"] >= len(frames) - 3


def test_motion_correspondence_prevents_cyclic_uv_corner_rotation():
    frames, masks, candidates, truth = moving_scene(count=13, jitter=0.2)
    candidates[6] = replace(candidates[6], quad=np.roll(candidates[6].quad, 1, axis=0))
    result, report = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    assert np.max(np.linalg.norm(result[6].quad - truth[6], axis=1)) < 1.0
    assert report["frames"][6]["input_corner_index_rotation"] == 3


def test_untextured_frames_do_not_imply_verified_motion_or_fill_missing_geometry():
    frames, masks, candidates, _ = moving_scene(count=5)
    frames = [np.full_like(frame, 100) for frame in frames]
    candidates[2] = None
    result, report = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    assert report["motion_pair_count"] == 0
    assert result[2] is None
    assert result[0] is candidates[0]


def test_bad_geometry_is_not_rendered_when_no_temporal_evidence_exists():
    frames, masks, candidates, _ = moving_scene(count=1)
    candidates[0] = replace(candidates[0], quad=candidates[0].quad + [0, 80])
    result, report = stabilize_scene_trajectory(frames, masks, candidates, fps=30)
    assert result == [None]
    assert report["frames"][0]["status"] == "no_supported_plane"


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="equal lengths"):
        stabilize_scene_trajectory([], [], [None], fps=30)
    with pytest.raises(ValueError, match="fps"):
        stabilize_scene_trajectory([], [], [], fps=float("nan"))
