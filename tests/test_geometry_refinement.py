"""Regressions exposed by small moving plates in the held development clip."""

import cv2
import numpy as np
from apexveil.imaging.geometry import fit_quad, edge_refine


def test_small_surface_does_not_expand_by_a_full_pixel_on_every_edge():
    mask = np.zeros((90, 200), np.uint8)
    mask[30:51, 40:141] = 1
    truth = np.array([[40, 30], [140, 30], [140, 50], [40, 50]], np.float32)
    q, _ = fit_quad(mask)
    assert np.max(np.abs(q - truth)) < 0.5


def test_stronger_neighboring_body_edge_does_not_pull_surface_into_background():
    rgb = np.full((220, 420, 3), 100, np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[60:141, 60:341] = 1
    rgb[mask > 0] = 99
    rgb[53:57, 45:355] = 240
    rgb[145:149, 45:355] = 240
    rgb[45:155, 53:57] = 240
    rgb[45:155, 345:349] = 240
    truth = np.array([[60, 60], [340, 60], [340, 140], [60, 140]], np.float32)
    q, logs = edge_refine(rgb, truth, visible_mask=mask)
    assert np.max(np.linalg.norm(q - truth, axis=1)) < 2
    assert any(
        x.get("status") == "visibility_support_guard_keep_previous" for x in logs
    )


def test_refinement_never_changes_the_observed_occlusion_mask():
    rgb = np.full((160, 300, 3), 120, np.uint8)
    mask = np.zeros((160, 300), np.uint8)
    mask[50:110, 60:240] = 1
    mask[50:110, 140:160] = 0
    before = mask.copy()
    q = np.array([[60, 50], [239, 50], [239, 109], [60, 109]], np.float32)
    edge_refine(rgb, q, visible_mask=mask)
    np.testing.assert_array_equal(mask, before)
