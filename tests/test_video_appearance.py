"""Temporal regressions use generic light/crop changes, not particular footage."""

from dataclasses import replace

import numpy as np
import pytest

from apexveil.imaging.appearance import AppearanceEstimate
from apexveil.imaging.color import linear, luma, srgb
from apexveil.imaging.degradation import DegradationEstimate
from apexveil.imaging.photometry import TemporalVideoAppearance


def appearance(gain=0.5, veil=0.003, shape=(48, 160)):
    black = np.broadcast_to(np.asarray(veil, np.float32), (*shape, 3)).copy()
    return AppearanceEstimate(
        gain, black, np.ones(3, np.float32), np.ones(shape, np.float32), {}
    )


def source(level=0.2, shape=(48, 160)):
    return srgb(np.full((*shape, 3), level, np.float32))


def filtered(state, estimate, reference, mask=None, **kwargs):
    valid = np.ones(reference.shape[:2], bool) if mask is None else mask > 0
    return state.filter_appearance(
        estimate,
        np.percentile(luma(linear(reference))[valid], [50, 80]),
        source_reference=reference,
        valid_mask=valid,
        **kwargs,
    )


def degradation(sigma, reliable=True):
    return DegradationEstimate(
        sigma,
        0,
        0,
        {"focus": {"sigma_px": sigma, "reliable": reliable}, "noise_reliable": True},
    )


def test_focus_estimator_drop_cannot_flip_ink_resolution_in_one_frame():
    state = TemporalVideoAppearance(30)
    state.filter_degradation(degradation(2.5))
    result = state.filter_degradation(degradation(0.7), visible_fraction=0.85)
    assert 1.9 < result.sigma_px < 2.5
    assert 54 / max(0.5, result.sigma_px) < 60
    assert result.diagnostics["focus"]["sigma_px"] == result.sigma_px
    assert result.diagnostics["temporal_capture"]["raw_sigma_px"] == 0.7


def test_focus_fallback_is_not_evidence_for_sharpening():
    state = TemporalVideoAppearance()
    state.filter_degradation(degradation(3))
    result = state.filter_degradation(degradation(0.35, reliable=False))
    assert result.sigma_px == pytest.approx(3)
    assert result.diagnostics["temporal_capture"]["parameter_weight_current"] == 0


def test_persistent_focus_change_converges_and_noise_is_reported_consistently():
    state = TemporalVideoAppearance()
    state.filter_degradation(degradation(3))
    for _ in range(30):
        result = state.filter_degradation(
            replace(degradation(0.5), noise_intercept=4, noise_slope=2)
        )
    assert result.sigma_px == pytest.approx(0.5, abs=0.001)
    assert result.noise_intercept == pytest.approx(4, abs=0.002)
    assert result.diagnostics["noise_slope_code_variance"] == result.noise_slope


def test_unexplained_colored_spatial_floor_fit_cannot_flash_black_material():
    state = TemporalVideoAppearance()
    before = appearance()
    filtered(state, before, source())
    after = appearance(gain=1.2, veil=[0.03, 0.20, 0.25])
    after.black[:, 80:, :] *= 1.7
    result = filtered(state, after, source())
    assert result.gain < before.gain * 1.08
    assert result.black.max() < 0.004
    assert not result.diagnostics["temporal_parameters"]["fast_light_response"]
    assert result.diagnostics["temporal_parameters"]["spatial_veil_envelope_bounded"]
    assert result.diagnostics["black_point_linear"] == pytest.approx(
        np.median(luma(result.black))
    )
    # The rejected spike leaves no large colored decay tail on the next frame.
    recovery = filtered(state, before, source())
    assert recovery.black.max() < 0.004


def test_crop_percentile_change_is_not_a_source_light_change():
    state = TemporalVideoAppearance()
    shape = (48, 160)
    level = np.broadcast_to(
        np.linspace(0.03, 0.5, shape[1])[None, :, None], (*shape, 3)
    ).copy()
    reference = srgb(level)
    filtered(state, appearance(), reference)
    valid = np.zeros(shape, bool)
    valid[:, 100:] = True
    result = filtered(state, appearance(gain=1, veil=0.1), reference, valid)
    detail = result.diagnostics["temporal_parameters"]
    assert (
        detail["light_evidence_method"] == "matching_visible_canonical_source_samples"
    )
    assert detail["source_overlap_fraction"] == 1
    assert abs(detail["source_supported_gain_log_step"]) < 1e-6
    assert not detail["fast_light_response"]
    assert result.gain < 0.54
    assert result.black.max() < 0.004


def test_inpainted_or_outside_source_pixels_do_not_supply_light_evidence():
    state = TemporalVideoAppearance()
    ref = source(0.1)
    filtered(state, appearance(), ref)
    valid = np.zeros(ref.shape[:2], bool)
    valid[:, :50] = True
    ref[~valid] = 255
    result = filtered(state, appearance(), ref, valid)
    assert (
        abs(result.diagnostics["temporal_parameters"]["source_supported_gain_log_step"])
        < 1e-6
    )
    assert result.gain == pytest.approx(0.5)


def test_real_light_change_updates_immediately_without_shadow_history():
    state = TemporalVideoAppearance()
    filtered(state, appearance(gain=0.2, veil=0.002), source(0.05))
    after = appearance(gain=0.8, veil=0.008)
    after.spatial_field[:, 80:] = 0.25
    result = filtered(state, after, source(0.2))
    assert result.gain > 0.76
    assert result.black.mean() > 0.0075
    assert result.diagnostics["temporal_parameters"]["fast_light_response"]
    np.testing.assert_array_equal(result.spatial_field, after.spatial_field)


def test_moving_current_veil_peak_never_remains_at_its_old_location():
    state = TemporalVideoAppearance()
    before, after = appearance(), appearance()
    before.black[:, 10:50, 0] = 0.06
    after.black[:, 110:150, 0] = 0.06
    before.spatial_field[:, :80] = 0.25
    after.spatial_field[:, 80:] = 0.25
    filtered(state, before, source())
    result = filtered(state, after, source())
    np.testing.assert_allclose(result.black, after.black, atol=1e-6)
    np.testing.assert_array_equal(result.spatial_field, after.spatial_field)
    assert result.black[:, 10:50, 0].max() < 0.004
    assert result.black[:, 110:150, 0].min() > 0.059


def test_reset_and_shape_change_do_not_carry_material_history():
    state = TemporalVideoAppearance()
    filtered(state, appearance(gain=1, veil=0.1), source())
    state.reset()
    fresh = filtered(state, appearance(gain=0.1, veil=0), source())
    assert fresh.gain == pytest.approx(0.1)
    assert fresh.black.max() == 0
    reshaped = appearance(gain=0.8, veil=0.05, shape=(20, 60))
    result = filtered(state, reshaped, source(shape=(20, 60)))
    assert result.gain == pytest.approx(0.8)
    np.testing.assert_allclose(result.black, reshaped.black)


def test_parameter_only_fallback_does_not_fast_gate_on_crop_change():
    state = TemporalVideoAppearance()
    first = appearance()
    state.filter_appearance(first, [0.1, 0.2])
    mask = np.zeros(first.black.shape[:2], bool)
    mask[:, :80] = True
    result = state.filter_appearance(
        appearance(gain=1, veil=0.1), [0.3, 0.4], valid_mask=mask
    )
    assert result.gain < 0.54
    assert not result.diagnostics["temporal_parameters"]["fast_light_response"]


def test_elapsed_frame_indices_and_input_validation():
    state = TemporalVideoAppearance()
    filtered(state, appearance(), source(), frame_index=2)
    result = filtered(state, appearance(), source(), frame_index=5)
    assert result.diagnostics["temporal_parameters"][
        "elapsed_seconds"
    ] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="increasing"):
        filtered(state, appearance(), source(), frame_index=5)
    with pytest.raises(ValueError, match="visible_fraction"):
        state.filter_degradation(degradation(1), visible_fraction=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        filtered(state, appearance(veil=-1), source())
    with pytest.raises(ValueError, match="visible reference"):
        state.filter_appearance(
            appearance(), [0.1, 0.2], valid_mask=np.zeros((48, 160), bool)
        )


def test_filter_returns_new_objects_without_mutating_raw_estimates():
    state = TemporalVideoAppearance()
    raw = appearance()
    before_black = raw.black.copy()
    result = filtered(state, raw, source())
    assert result is not raw
    np.testing.assert_array_equal(raw.black, before_black)
    assert not raw.diagnostics
    capture = degradation(1)
    stabilized = state.filter_degradation(capture)
    assert stabilized is not capture
    assert "temporal_capture" not in capture.diagnostics


def test_matching_bias_cannot_accumulate_veil_or_gain_beyond_observed_bounds():
    state = TemporalVideoAppearance()
    rng = np.random.default_rng(8)
    estimate = appearance(gain=0.5, veil=[0.003, 0.004, 0.005])
    # Moving unresolved glyphs, quantization and slight warp error produce
    # large low-signal ratios even when their underlying material is unchanged.
    for _ in range(80):
        ref = srgb(rng.uniform(0.001, 0.3, (48, 160, 3)).astype(np.float32))
        result = filtered(state, estimate, ref)
        assert result.gain == pytest.approx(estimate.gain)
        np.testing.assert_allclose(result.black, estimate.black, rtol=1e-6)


def test_source_change_opposing_raw_estimate_does_not_extrapolate():
    state = TemporalVideoAppearance()
    first = appearance(gain=0.5, veil=0.005)
    filtered(state, first, source(0.1))
    result = filtered(state, appearance(gain=0.4, veil=0.004), source(0.4))
    assert 0.4 <= result.gain <= 0.5
    assert 0.004 <= result.black.mean() <= 0.005


def test_tiny_visible_corner_fit_spike_cannot_hide_above_a_percentile():
    state = TemporalVideoAppearance()
    filtered(state, appearance(), source())
    after = appearance()
    after.black[:2, :2, 1] = 0.5
    result = filtered(state, after, source())
    assert result.black.max() < 0.004
    assert (
        result.diagnostics["temporal_parameters"]["raw_veil_rgb_peak_linear"][1] == 0.5
    )
