import cv2
import numpy as np
import pytest

from apexveil.imaging.scene_geometry import (
    propose_scene_geometry,
    select_scene_geometry,
)


def scene(shift=0):
    original = np.full((140, 260, 3), 40, np.uint8)
    texture = np.random.default_rng(21).integers(35, 210, (60, 180, 3), dtype=np.uint8)
    original[40:100, 40:220] = texture
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[40:100, 40:220] = 255
    q = np.array([[40, 40], [219, 40], [219, 99], [40, 99]], np.float32)
    h = np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1]], np.float32)
    return (
        cv2.warpPerspective(original, h, (260, 140)),
        cv2.warpPerspective(mask, h, (260, 140), flags=cv2.INTER_NEAREST),
        cv2.perspectiveTransform(q[None], h)[0],
    )


def test_cropped_plane_has_outside_corners_and_preserves_native_mask():
    image, mask, truth = scene(-80)
    before = mask.copy()
    candidates = propose_scene_geometry(image, mask, aspect=179 / 59)
    best = select_scene_geometry(candidates)
    assert best is not None
    assert best.diagnostics["edge_labels"]["left"] == "crop"
    assert best.quad[:, 0].min() < 0
    assert np.max(np.linalg.norm(best.quad - truth, axis=1)) < 3
    assert "missing_edge_not_uniquely_observable" in best.uncertainties
    np.testing.assert_array_equal(mask, before)
    assert any(not c.diagnostics["eligible_for_render"] for c in candidates)


def test_occluder_edge_is_not_fitted_as_full_plane_side():
    image, mask, truth = scene()
    occluder = np.zeros_like(mask)
    cv2.fillConvexPoly(occluder, np.array([[0, 0], [125, 0], [95, 140], [0, 140]]), 1)
    mask[occluder > 0] = 0
    candidates = propose_scene_geometry(
        image,
        mask,
        aspect=179 / 59,
        guidance={
            "source": "agent_keyframe_hypothesis",
            "coordinate_space": "source",
            "geometry": {
                "edge_labels": ["real_plate", "real_plate", "real_plate", "occlusion"],
                "physical_aspect_guess": 179 / 59,
            },
        },
    )
    best = select_scene_geometry(candidates)
    assert best is not None
    assert np.max(np.linalg.norm(best.quad - truth, axis=1)) < 4
    assert best.protected_mask[70, 70]
    assert not best.protected_mask[70, 180]
    assert best.diagnostics["guidance_source"] == "agent_keyframe_hypothesis"


def test_direct_anchor_needs_no_intermediate_frames_or_chain():
    reference, reference_mask, reference_quad = scene()
    current, mask, truth = scene(90)
    candidates = propose_scene_geometry(
        current,
        mask,
        aspect=179 / 59,
        anchors=[
            {
                "rgb": reference,
                "mask": reference_mask,
                "quad": reference_quad,
                "frame_index": 0,
                "source": "same_shot_native_identity",
            }
        ],
    )
    best = select_scene_geometry(candidates)
    assert best is not None
    assert np.max(np.linalg.norm(best.quad - truth, axis=1)) < 3
    assert best.diagnostics["direct_anchor_not_adjacent_chain"]
    assert best.diagnostics["anchor_frame"] == 0
    assert any(
        c.diagnostics.get("registration", {}).get("status") == "registered"
        for c in candidates
    )


def test_guidance_cannot_bypass_observed_pixel_constraints():
    image, mask, _ = scene()
    bad = [[150, 30], [230, 30], [230, 80], [150, 80]]
    candidates = propose_scene_geometry(
        image,
        mask,
        aspect=3,
        guidance={"candidate_quads": [bad], "confidence": 1.0, "source": "agent"},
    )
    guided = [
        c
        for c in candidates
        if "explicit_agent_or_user_geometry_hypothesis" in c.evidence
    ]
    assert len(guided) == 1
    assert not guided[0].diagnostics["eligible_for_render"]
    assert select_scene_geometry(candidates) is not guided[0]


def test_top_viewport_clipping_preserves_corner_orientation():
    image, mask, q = scene()
    h = np.array([[1, 0, 0], [0, 1, -65], [0, 0, 1]], np.float32)
    image = cv2.warpPerspective(image, h, (260, 140))
    mask = cv2.warpPerspective(mask, h, (260, 140), flags=cv2.INTER_NEAREST)
    truth = cv2.perspectiveTransform(q[None], h)[0]
    best = select_scene_geometry(propose_scene_geometry(image, mask, aspect=179 / 59))
    assert best is not None
    assert best.diagnostics["edge_labels"]["top"] == "crop"
    assert np.max(np.linalg.norm(best.quad - truth, axis=1)) < 3


def test_crop_local_guidance_is_rejected_before_pixel_interpretation():
    image, mask, _ = scene()
    with pytest.raises(ValueError, match="source pixels"):
        propose_scene_geometry(
            image, mask, aspect=3, guidance={"coordinate_space": "crop"}
        )
