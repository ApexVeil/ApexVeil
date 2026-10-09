import cv2
import numpy as np
import pytest
from apexveil.imaging.temporal import (
    TemporalGeometry,
    TemporalAppearance,
    rectified_visible_reference,
)
from apexveil.video_pipeline import timestamp_mapping


def textured_surface():
    image = np.full((180, 340, 3), 70, np.uint8)
    image[55:125, 80:260] = np.random.default_rng(12).integers(
        25, 210, (70, 180, 3), dtype=np.uint8
    )
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[55:125, 80:260] = 1
    quad = np.array([[80, 55], [259, 55], [259, 124], [80, 124]], np.float32)
    return image, mask, quad


def test_motion_compensation_follows_translation_and_reduces_observation_noise():
    previous, mask, quad = textured_surface()
    matrix = np.array([[1, 0, 3], [0, 1, -2]], np.float32)
    current = cv2.warpAffine(
        previous, matrix, (340, 180), borderMode=cv2.BORDER_REFLECT
    )
    visible = cv2.warpAffine(mask, matrix, (340, 180))
    truth = quad + [3, -2]
    observed = truth + [2, 1]
    estimator = TemporalGeometry()
    estimator.step(previous, mask, quad)
    result = estimator.step(current, visible, observed.astype(np.float32))
    assert result.diagnostics["status"] == "motion_fused"
    assert result.diagnostics["flow"]["inliers"] >= 8
    assert np.mean(np.linalg.norm(result.quad - truth, axis=1)) < np.mean(
        np.linalg.norm(observed - truth, axis=1)
    )
    # It follows the true translation; no screen-coordinate EMA delay.
    assert np.mean(result.quad - quad, axis=0)[0] > 2.5


def test_no_observation_and_no_features_does_not_invent_geometry():
    image = np.zeros((100, 200, 3), np.uint8)
    mask = np.ones((100, 200), np.uint8)
    estimator = TemporalGeometry()
    result = estimator.step(image, mask, None)
    assert result.quad is None
    assert "insufficient_geometry" in result.diagnostics["flags"]


def test_scene_flash_resets_geometry_instead_of_propagating_old_plane():
    image, mask, quad = textured_surface()
    estimator = TemporalGeometry()
    estimator.step(image, mask, quad)
    result = estimator.step(np.full_like(image, 255), mask, None)
    assert result.quad is None
    assert "possible_cut_or_flash_state_reset" in result.diagnostics["flags"]


def test_invisible_foreground_is_excluded_from_appearance_reference():
    image = np.full((100, 200, 3), 90, np.uint8)
    mask = np.zeros((100, 200), np.uint8)
    mask[20:80, 20:180] = 1
    image[35:65, 70:110] = [255, 0, 0]
    mask[35:65, 70:110] = 0
    quad = np.array([[20, 20], [179, 20], [179, 79], [20, 79]], np.float32)
    reference, coverage = rectified_visible_reference(image, mask, quad, (160, 60))
    assert coverage < 1
    assert np.max(np.abs(reference[:, :, 0].astype(float) - reference[:, :, 1])) <= 3


def test_real_light_change_uses_fast_response_not_long_temporal_average():
    state = TemporalAppearance()
    state.step(np.full((60, 160, 3), 80, np.uint8), 1.0)
    result, _, metadata = state.step(np.full((60, 160, 3), 180, np.uint8), 1.0)
    assert metadata["appearance_weight_current"] == 0.85
    assert result.mean() > 160
    missing, _, metadata = state.step(None, 1.0)
    assert missing is None
    assert metadata["flags"]


def test_cfr_mapping_keeps_source_offset_explicit():
    mapping = timestamp_mapping(60, 30, 240)
    assert mapping[0]["source_sample_time_seconds"] == 240
    assert mapping[-1]["source_sample_time_seconds"] == pytest.approx(241 + 29 / 30)
    assert mapping[-1]["output_time_seconds"] == pytest.approx(59 / 30)
    with pytest.raises(ValueError):
        timestamp_mapping(60, 0, 240)


def test_tracking_cache_rejects_replaced_source_content(tmp_path):
    import json
    from apexveil.video_pipeline import file_sha256, verify_cache_provenance

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original-encoded-clip")
    (tmp_path / "input-provenance.json").write_text(
        json.dumps(
            {
                "clip_sha256": file_sha256(clip),
                "source_sha256": "source-sha",
                "start": 240,
                "duration": 2,
                "frame_count": 60,
            }
        )
    )
    verify_cache_provenance(tmp_path, clip, "source-sha", 240, 2)
    clip.write_bytes(b"different-encoded-clip")
    with pytest.raises(ValueError, match="hash"):
        verify_cache_provenance(tmp_path, clip, "source-sha", 240, 2)


def test_video_multi_target_requires_explicit_selection(tmp_path):
    import json
    from apexveil.video_pipeline import selected_masks

    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "objects": [
                            {"object_id": 0, "mask": "a.png"},
                            {"object_id": 1, "mask": "b.png"},
                        ],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="target-id"):
        selected_masks(tmp_path, 1, None)
    masks, target, _ = selected_masks(tmp_path, 1, 1)
    assert target == 1
    assert masks[0].name == "b.png"


def test_new_foreground_cannot_supply_flow_for_the_hidden_surface():
    previous, mask, quad = textured_surface()
    estimator = TemporalGeometry()
    estimator.step(previous, mask, quad)
    # Texture still matches perfectly, but it is explicitly no longer visible.
    # A flow tracker that ignores current visibility would keep hallucinating it.
    result = estimator.step(previous.copy(), np.zeros_like(mask), None)
    assert result.quad is None
    assert result.diagnostics["flow"]["current_visibility_rejected"] >= 5
