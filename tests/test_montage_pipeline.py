import numpy as np
import pytest
from apexveil.montage_pipeline import (
    validate_manifest,
    full_shot_request,
    select_primary_alias,
    title_pixels,
)


def sample_manifest():
    return {
        "fps": 30,
        "frame_count": 5,
        "shots": [
            {
                "id": "shot_000",
                "start_frame": 0,
                "end_frame": 2,
                "frame_count": 2,
                "kind": "shot",
            },
            {
                "id": "shot_001",
                "start_frame": 2,
                "end_frame": 5,
                "frame_count": 3,
                "kind": "transition",
            },
        ],
    }


def test_montage_includes_transition_time_without_gaps_or_overlap():
    manifest = sample_manifest()
    assert validate_manifest(manifest)
    manifest["shots"][1]["start_frame"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        validate_manifest(manifest)


def test_montage_rejects_missing_tail_and_duplicate_shot_ids():
    manifest = sample_manifest()
    manifest["frame_count"] = 6
    with pytest.raises(ValueError, match="complete timeline"):
        validate_manifest(manifest)
    manifest = sample_manifest()
    manifest["shots"][1]["id"] = "shot_000"
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(manifest)


@pytest.mark.parametrize("count", [1, 10, 15, 16, 31, 32, 60, 69, 118, 132])
def test_native_full_shot_request_satisfies_both_upstream_end_conventions(count):
    request = full_shot_request("session", count)
    # Pinned upstream processing_order clamps an inclusive end, while its
    # grounding feature batches cap at an exclusive end. T-1 breaks the tail.
    inclusive_last = min(request["max_frame_num_to_track"], count - 1)
    processing = list(range(inclusive_last + 1))
    feature_exclusive_end = request["max_frame_num_to_track"]
    assert processing == list(range(count))
    assert all(frame < feature_exclusive_end for frame in processing)


def test_primary_alias_switch_does_not_merge_masks_or_choose_another_vehicle():
    a = np.ones((4, 4), np.uint8)
    b = np.eye(4, dtype=np.uint8)
    assert select_primary_alias({0: a, 1: b}, [0, 1], 0) == 0
    assert select_primary_alias({1: b, 2: a}, [0, 1], 0) == 1
    assert select_primary_alias({2: a}, [0, 1], 1) is None
    assert b.sum() == 4


def test_title_strip_does_not_consume_native_image_rows():
    strip = title_pixels()
    assert strip.shape == (48, 3840, 3)
    assert strip.dtype == np.uint8


@pytest.mark.parametrize("stage", ["render", "finalize"])
def test_missing_template_has_clear_cli_error_before_file_access(
    monkeypatch, capsys, stage
):
    from apexveil.montage_pipeline import main

    monkeypatch.setattr(
        "sys.argv", ["montage_pipeline", stage, "--root", "/not-a-real-root"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--template is required for render/finalize" in capsys.readouterr().err
