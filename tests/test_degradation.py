import cv2
import numpy as np

from apexveil.imaging.degradation import (
    DegradationEstimate,
    apply_capture_noise,
    estimate_degradation,
)
from apexveil.imaging.focus import estimate_edge_softness


def test_noise_estimator_recovers_known_flat_capture_noise_without_copying_texture():
    rng = np.random.default_rng(5)
    base = np.zeros((180, 540, 3), np.float32)
    for x in range(0, 540, 90):
        base[:, x : x + 90] = 30 + x / 3
    noisy = np.clip(
        np.rint(base + rng.normal(0, 3, base.shape[:2])[:, :, None]), 0, 255
    ).astype(np.uint8)
    mask = np.zeros((180, 540), np.uint8)
    mask[50:130, 60:480] = 1
    quad = np.array([[60, 50], [479, 50], [479, 129], [60, 129]], np.float32)
    est = estimate_degradation(noisy, mask, quad)
    assert est.diagnostics["noise_reliable"]
    recovered = np.sqrt(est.noise_intercept + est.noise_slope * 0.5)
    assert 2 < recovered < 4


def test_noise_seed_is_reproducible_and_changes_between_frames():
    rgb = np.full((200, 300, 3), 100, np.uint8)
    est = DegradationEstimate(0, 9, 4, {})
    one = apply_capture_noise(rgb, est, 2)
    two = apply_capture_noise(rgb, est, 3)
    assert np.array_equal(one, apply_capture_noise(rgb, est, 2))
    assert not np.array_equal(one, two)
    assert 2.5 < np.std(one.astype(float) - rgb) < 4


def test_known_edge_spread_orders_sharp_and_defocused_capture():
    rgb = np.full((300, 600, 3), 25, np.uint8)
    rgb[80:220, 80:520] = 170
    quad = np.array([[80, 80], [519, 80], [519, 219], [80, 219]], np.float32)
    sharp = estimate_edge_softness(rgb, quad)
    blurry = estimate_edge_softness(cv2.GaussianBlur(rgb, (0, 0), 2.5), quad)
    assert blurry["reliable"]
    assert blurry["sigma_px"] > sharp["sigma_px"] + 1.5
    assert 1.8 < blurry["sigma_px"] < 3.3
