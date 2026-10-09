"""Pinned upstream regression: a fresh point session has no VG frame cache."""

import numpy as np
from apexveil.backends.sam31 import install_interactive_output_compat


class UpstreamEarlyReturn:
    def _build_sam2_output(self, state, frame, refined_obj_id_to_mask=None):
        if frame not in state["cached_frame_outputs"]:
            return {}
        result = state["cached_frame_outputs"][frame].copy()
        if refined_obj_id_to_mask is not None:
            result.update(refined_obj_id_to_mask)
        return result


def test_fresh_tracker_prediction_survives_without_inventing_cached_state():
    model = UpstreamEarlyReturn()
    mask = np.array([[0, 1], [1, 0]], dtype=bool)
    state = {
        "cached_frame_outputs": {},
        "action_history": [{"type": "add", "obj_ids": [5]}],
    }
    install_interactive_output_compat(model)
    result = model._build_sam2_output(state, 12, {5: mask})
    np.testing.assert_array_equal(result[5], mask)
    assert state["cached_frame_outputs"] == {}
    assert state["action_history"] == [{"type": "add", "obj_ids": [5]}]


def test_existing_cache_keeps_other_objects_and_uses_current_refinement():
    model = UpstreamEarlyReturn()
    install_interactive_output_compat(model)
    a, b, c = object(), object(), object()
    state = {"cached_frame_outputs": {3: {0: a, 1: b}}}
    assert model._build_sam2_output(state, 3, {0: c}) == {0: c, 1: b}
    assert state["cached_frame_outputs"][3] == {0: a, 1: b}


def test_empty_tracking_does_not_fabricate_masks_and_install_is_idempotent():
    model = UpstreamEarlyReturn()
    install_interactive_output_compat(model)
    first = model._build_sam2_output
    install_interactive_output_compat(model)
    assert model._build_sam2_output is first
    assert model._build_sam2_output({"cached_frame_outputs": {}}, 2, {}) == {}
