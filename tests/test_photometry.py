import numpy as np

from apexveil.imaging.appearance import adapt_asset
from apexveil.imaging.color import linear, srgb
from apexveil.imaging.photometry import TemporalPhotometry


def parameters(gain=1.0, black=0.002):
    return gain, np.full((60, 180, 3), black, np.float32), np.ones(3, np.float32)


def test_unexplained_one_frame_exposure_fit_spike_does_not_flash_material():
    state = TemporalPhotometry(30)
    state.filter(*parameters(), [0.1, 0.2])
    gain, _, _, meta = state.filter(*parameters(3), [0.1, 0.2])
    assert 1 < gain < 1.08
    assert not meta["fast_light_response"]
    gain, _, _, _ = state.filter(*parameters(), [0.1, 0.2])
    assert gain < 1.08


def test_observed_real_exposure_change_gets_fast_response():
    state = TemporalPhotometry(30)
    state.filter(*parameters(0.25), [0.025, 0.05])
    gain, _, _, meta = state.filter(*parameters(1), [0.1, 0.2])
    assert gain > 0.7
    assert meta["fast_light_response"]
    for _ in range(2):
        gain, _, _, _ = state.filter(*parameters(1), [0.1, 0.2])
    assert gain > 0.97


def test_current_spatial_shadow_moves_instead_of_leaving_a_history_trail():
    shape = (120, 480, 3)
    plate = np.broadcast_to(np.array([0.1, 0.3, 0.12], np.float32), shape).copy()
    a = plate.copy()
    a[:, :240] *= 0.2
    b = plate.copy()
    b[:, 240:] *= 0.2
    asset = np.full(shape, 150, np.uint8)
    state = TemporalPhotometry(30)
    out_a, _ = adapt_asset(
        asset, srgb(a), match_black=False, parameter_filter=state.filter
    )
    out_b, detail = adapt_asset(
        asset, srgb(b), match_black=False, parameter_filter=state.filter
    )
    aa, bb = linear(out_a).mean(2), linear(out_b).mean(2)
    assert bb[:, 60:150].mean() > aa[:, 60:150].mean() * 2
    assert bb[:, 300:390].mean() < aa[:, 300:390].mean() / 2
    assert not detail["temporal_parameters"]["spatial_shadow_field_filtered"]


def test_reset_does_not_carry_old_exposure_or_color_into_a_new_shot():
    state = TemporalPhotometry()
    state.filter(*parameters(1, 0.1), [0.1, 0.2])
    state.reset()
    gain, black, _, _ = state.filter(*parameters(0.05, 0), [0.002, 0.004])
    assert abs(gain - 0.05) < 1e-6
    assert black.max() == 0


def test_moving_spatial_rgb_veil_has_no_history_peak_with_stable_global_level():
    state = TemporalPhotometry(30)
    left = np.full((60, 180, 3), 0.002, np.float32)
    right = left.copy()
    left[:, 20:50, 0] += 0.06
    right[:, 120:150, 0] += 0.06
    state.filter(1, left, np.ones(3), [0.1, 0.2])
    _, current, _, meta = state.filter(1, right, np.ones(3), [0.1, 0.2])
    assert current[:, 20:50, 0].max() < 0.003
    assert current[:, 120:150, 0].min() > 0.06
    assert meta["spatial_veil_residual_filtered"] is False


def test_uniform_veil_retains_previous_scalar_ema_and_real_light_fast_response():
    state = TemporalPhotometry(30)
    state.filter(*parameters(1, 0.002), [0.1, 0.2])
    _, black, _, meta = state.filter(*parameters(1, 0.02), [0.1, 0.2])
    expected = 0.002 + (0.02 - 0.002) * meta["parameter_weight_current"]
    np.testing.assert_allclose(black, expected, atol=1e-7)
    _, black, _, meta = state.filter(*parameters(1, 0.08), [0.4, 0.8])
    assert meta["fast_light_response"]
    assert black.mean() > 0.06
