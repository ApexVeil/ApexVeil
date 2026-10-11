import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from apexveil.workflows import video


class FakePredictor:
    def __init__(self, masks, fail=False):
        self.masks = masks
        self.requests = []
        self.fail = fail

    def response(self, index):
        items = self.masks[index]
        return {
            "frame_index": index,
            "outputs": {
                "out_obj_ids": np.array(list(items), np.int64),
                "out_binary_masks": np.array(list(items.values()), np.uint8),
                "out_probs": np.full(len(items), 0.91, np.float32),
            },
        }

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "native-session"}
        if request["type"] == "add_prompt":
            return self.response(0)
        if request["type"] == "close_session":
            return {}
        raise AssertionError(request)

    def handle_stream_request(self, request):
        self.requests.append(request)
        for i in range(len(self.masks)):
            if self.fail and i == 2:
                raise RuntimeError("synthetic native failure")
            yield self.response(i)


def fixture(tmp_path, *, missing_last=False, hard_cut=False):
    frames, masks = [], []
    for i in range(4):
        rgb = np.full((80, 160, 3), 70, np.uint8)
        rgb[25:55, 30 + i : 120 + i] = [45, 155, 85]
        cv2.putText(
            rgb, "AB27", (39 + i, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 20, 15), 1
        )
        first = np.zeros(rgb.shape[:2], np.uint8)
        first[25:55, 30 + i : 120 + i] = 1
        second = np.zeros_like(first)
        if i >= 2:
            rgb[59:73, 130:156] = [170, 150, 60]
            second[59:73, 130:156] = 1
        items = {0: first}
        if i >= 2:
            items[7] = second
        if missing_last and i == 3:
            del items[0]
        if hard_cut and i >= 2:
            rgb[:] = 255
        frames.append(rgb)
        masks.append(items)
    source = tmp_path / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "160x80",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            "-threads",
            "1",
            str(source),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True,
    )
    asset = np.full((40, 120, 4), 20, np.uint8)
    asset[:, :, 3] = 255
    asset[14:27, 35:90, :3] = 220
    template = tmp_path / "template.png"
    Image.fromarray(asset).save(template)
    return source, template, FakePredictor(masks)


def prepare(tmp_path, **options):
    source, template, predictor = fixture(tmp_path, **options)
    result = video.prepare_video(
        source,
        tmp_path / "prepared",
        SimpleNamespace(confidence_threshold=0.5),
        predictor=predictor,
        duration=0.4,
    )
    return result, template, predictor


def advice():
    return {
        "plate_confirmed": True,
        "coordinate_space": "crop_pixels",
        "geometry": {
            "edge_labels": dict.fromkeys(
                ["top", "right", "bottom", "left"], "real_plate"
            ),
            "physical_aspect_guess": 3.0,
            "candidate_quads": [[[30, 25], [119, 25], [119, 54], [30, 54]]],
            "assumptions": ["synthetic fixture only"],
        },
        "appearance": {
            "exposure_gain_linear": 0.4,
            "additive_rgb_linear": [0.02, 0.02, 0.02],
            "flare_control_points": [
                {"xy": [0.5, 0.5], "rgb_linear": [0.05, 0.05, 0.05]}
            ],
            "shading_control_points": [{"xy": [0.5, 0.5], "value": 1.0}],
            "diffuse_tint_rgb": [1.0, 1.0, 1.0],
        },
        "assessment": {
            "acceptable": False,
            "issues": ["not quality accepted"],
            "next_action": "inspect render",
        },
    }


def test_prepare_reuses_predictor_keeps_text_native_ids_and_preview_coordinates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        video, "build_predictor", lambda _: pytest.fail("Must reuse supplied predictor")
    )
    prepared, _, predictor = prepare(tmp_path)
    public = prepared["public"]
    assert public["width"] == 160 and public["height"] == 80
    assert public["fps"] == 10 and public["frame_count"] == 4
    assert {t["id"] for t in public["targets"]} == {0, 7}
    assert prepared["timings"]["sam_cold_load_seconds"] == 0
    for target in public["targets"]:
        index = target["representative_frame_index"]
        mask = video._read_masks(prepared, index)[target["id"]]
        x, y, w, h = cv2.boundingRect(mask)
        assert target["bbox"] == [x, y, x + w, y + h]
        assert target["color_rgb"] == list(
            video.stable_color(target["id"], prepared["_palette_scope"])
        )
        assert Image.open(target["source_preview_path"]).size == (160, 80)
    seed = next(r for r in predictor.requests if r["type"] == "add_prompt")
    stream = next(r for r in predictor.requests if r["type"] == "propagate_in_video")
    assert seed["text"] == "license plate" and "points" not in seed
    assert stream["propagation_direction"] == "forward"
    assert stream["max_frame_num_to_track"] == 4
    assert predictor.requests[-1]["type"] == "close_session"
    json.dumps(public)


def test_prepare_rejects_cut_before_predictor_use(tmp_path):
    source, _, predictor = fixture(tmp_path, hard_cut=True)
    with pytest.raises(ValueError, match="cut/flash"):
        video.prepare_video(
            source,
            tmp_path / "prepared",
            SimpleNamespace(confidence_threshold=0.5),
            predictor=predictor,
            duration=0.4,
        )
    assert predictor.requests == []


def test_prepare_bounds_and_resize_policy(tmp_path):
    with pytest.raises(ValueError, match="8 seconds"):
        video.prepare_video("missing", tmp_path, object(), duration=8.1)
    assert video._processing_size({"width": 3840, "height": 2160}) == (
        (3840, 2160),
        (1920, 1080),
    )
    assert video._processing_size(
        {"width": 1920, "height": 1080, "tags": {"rotate": "90"}}
    ) == ((1080, 1920), (606, 1080))


def test_native_failure_keeps_warning_and_closes_session(tmp_path):
    source, _, predictor = fixture(tmp_path)
    predictor.fail = True
    result = video.prepare_video(
        source,
        tmp_path / "prepared",
        SimpleNamespace(confidence_threshold=0.5),
        predictor=predictor,
        duration=0.4,
    )
    assert any("Native tracking incomplete" in w for w in result["warnings"])
    assert result["_records"][3]["native_output_missing"]
    assert predictor.requests[-1]["type"] == "close_session"


def test_render_uses_fixed_scene_path_and_preserves_missing_masks(
    tmp_path, monkeypatch
):
    prepared, template, _ = prepare(tmp_path, missing_last=True)
    original = video.solve_scene
    seen = []

    def track(*args, **kwargs):
        assert "selected_candidate" in kwargs
        seen.append(kwargs["selected_candidate"])
        return original(*args, **kwargs)

    monkeypatch.setattr(video, "solve_scene", track)
    result = video.render_video(prepared, template, tmp_path / "render", target_id=0)
    report = json.loads(Path(result["report_path"]).read_text())
    assert len(seen) == 4 and seen[-1] is None
    assert report["rendered_target_frames"] == 3
    assert report["calls"]["advisor_attempted"] == 0
    assert report["encoded_verification"]["output"]["encoded_frame_count"] == 4
    assert report["frames"][2]["other_blurred_ids"] == [7]
    assert (
        report["frames"][3]["scene"]["unresolved_reason"] == "native_visibility_missing"
    )


@pytest.mark.parametrize("mode", ["blur", "mosaic"])
def test_redaction_modes_skip_ai_and_scene_geometry(tmp_path, monkeypatch, mode):
    prepared, _, _ = prepare(tmp_path)
    monkeypatch.setattr(
        video,
        "propose_scene_geometry",
        lambda *a, **k: pytest.fail("No geometry needed"),
    )
    monkeypatch.setattr(
        video,
        "CodexSceneAdvisor",
        lambda *a, **k: pytest.fail("No advisor for redaction"),
    )
    result = video.render_video(
        prepared,
        None,
        tmp_path / mode,
        target_id=0,
        mode=mode,
        advisor_profile="luna-medium",
    )
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["rendered_target_frames"] == 4
    assert report["calls"]["advisor_attempted"] == 0


def test_one_advisor_call_relative_only_and_no_locked_uv_fields(tmp_path, monkeypatch):
    prepared, template, _ = prepare(tmp_path)
    monkeypatch.setattr(
        video,
        "_make_evidence",
        lambda *a, **k: {
            "prompt": "one continuous shot",
            "image_paths": [template],
            "crop_xyxy": [0, 0, 160, 80],
        },
    )
    calls = []

    class Advisor:
        def __init__(self, **kwargs):
            pass

        def run(self, *args):
            calls.append(args)
            return advice(), {"status": "proposal_available", "quality_accepted": False}

    monkeypatch.setattr(video, "CodexSceneAdvisor", Advisor)
    original = video.solve_scene
    passed = []

    def record(*args, **kwargs):
        passed.append(kwargs.get("appearance_guidance"))
        return original(*args, **kwargs)

    monkeypatch.setattr(video, "solve_scene", record)
    result = video.render_video(
        prepared,
        template,
        tmp_path / "render",
        target_id=0,
        advisor_profile="sol-medium",
    )
    report = json.loads(Path(result["report_path"]).read_text())
    assert len(calls) == 1
    assert (
        report["calls"]["advisor_attempted"]
        == report["calls"]["advisor_succeeded"]
        == 1
    )
    assert all(p is None for p in passed)
    assert set(report["advisor"]["ignored_video_appearance_fields"]) == {
        "additive_rgb_linear",
        "flare_control_points",
        "shading_control_points",
    }
    assert report["automatic_quality_review_run"] is False
    assert report["appearance_correction_ratios"]["exposure"] != 1


def test_failed_advisor_does_not_retry_or_claim_ai_success(tmp_path, monkeypatch):
    prepared, template, _ = prepare(tmp_path)
    monkeypatch.setattr(
        video,
        "_make_evidence",
        lambda *a, **k: {
            "prompt": "p",
            "image_paths": [template],
            "crop_xyxy": [0, 0, 160, 80],
        },
    )
    calls = []

    class Advisor:
        def __init__(self, **kwargs):
            pass

        def run(self, *args):
            calls.append(args)
            return None, {
                "status": "failed",
                "error": "offline",
                "quality_accepted": False,
            }

    monkeypatch.setattr(video, "CodexSceneAdvisor", Advisor)
    result = video.render_video(
        prepared,
        template,
        tmp_path / "render",
        target_id=0,
        advisor_profile="luna-medium",
    )
    report = json.loads(Path(result["report_path"]).read_text())
    assert len(calls) == 1
    assert (
        report["calls"]["advisor_attempted"] == 1
        and report["calls"]["advisor_succeeded"] == 0
    )
    assert report["rendered_target_frames"] == 4
    assert any("AI proposal failed" in w for w in result["warnings"])


def test_all_unresolved_replace_is_not_successful_original_output(
    tmp_path, monkeypatch
):
    prepared, template, _ = prepare(tmp_path)
    monkeypatch.setattr(video, "propose_scene_geometry", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="unresolved on every target frame"):
        video.render_video(prepared, template, tmp_path / "render", target_id=0)


def test_decoded_cache_reuses_gaps_without_changing_pixels(tmp_path):
    source, _, _ = fixture(tmp_path)
    uncached = video._VideoFrames(source, 4, cache_bytes=0)
    expected = [uncached[i].copy() for i in range(4)]
    uncached.close()
    cached = video._VideoFrames(source, 4, cache_bytes=sum(f.nbytes for f in expected))
    # A clear anchor near the end must populate the preceding frames once.
    assert np.array_equal(cached[3], expected[3])
    assert set(cached.decoded) == {0, 1, 2, 3}
    for i in (0, 2, 1, 3, 0):
        assert np.array_equal(cached[i], expected[i])
    assert cached.next_index == 4
    cached.close()
    assert not cached.decoded and cached.capture is None
    assert np.array_equal(cached[1], expected[1])
    cached.close()


def test_decoded_cache_budget_falls_back_to_single_frame(tmp_path):
    source, _, _ = fixture(tmp_path)
    frames = video._VideoFrames(source, 4, cache_bytes=160 * 80 * 3 * 4 - 1)
    first = frames[0].copy()
    _ = frames[3]
    assert frames.cache_complete_clip is False
    assert not frames.decoded
    assert np.array_equal(frames[0], first)
    frames.close()


def test_stage1_video_cache_reopens_without_sam_and_rejects_new_interval(
    tmp_path, monkeypatch
):
    prepared, _, predictor = prepare(tmp_path)
    calls = len(predictor.requests)
    monkeypatch.setattr(
        video, "build_predictor", lambda *_: pytest.fail("Stage 2 loaded SAM")
    )
    restored = video.restore_video_prepared(
        tmp_path / "prepared",
        source_path=prepared["_source_path"],
        start=0,
        duration=0.4,
    )
    assert restored["_predictor"] is None
    assert len(predictor.requests) == calls
    for i in range(4):
        for identity, mask in video._read_masks(prepared, i).items():
            np.testing.assert_array_equal(
                mask, video._read_masks(restored, i)[identity]
            )
    with pytest.raises(ValueError, match="interval changed"):
        video.restore_video_prepared(
            tmp_path / "prepared",
            source_path=prepared["_source_path"],
            start=0.1,
            duration=0.3,
        )
    path = restored["_mask_paths"][0][0]
    path.unlink()
    with pytest.raises(ValueError, match="mask is missing"):
        video.restore_video_prepared(
            tmp_path / "prepared",
            source_path=prepared["_source_path"],
            start=0,
            duration=0.4,
        )


@pytest.mark.parametrize(
    ("start", "duration", "expected"),
    [(0, 0.11, 1), (0, 0.21, 2), (0, 0.31, 3), (0.01, 0.21, 2)],
)
def test_fractional_interval_reopens_exact_selection_and_legacy_cache(
    tmp_path, start, duration, expected
):
    source, _, predictor = fixture(tmp_path)
    predictor.masks = predictor.masks[:expected]
    directory = tmp_path / "prepared"
    prepared = video.prepare_video(
        source,
        directory,
        SimpleNamespace(confidence_threshold=0.5),
        predictor=predictor,
        start=start,
        duration=duration,
    )
    assert prepared["public"]["frame_count"] == expected
    assert not any("Native tracking incomplete" in w for w in prepared["warnings"])
    report_path = directory / "prepare-report.json"
    report = json.loads(report_path.read_text())
    assert report["source_selection"] == {
        "start": start,
        "requested_duration": duration,
        "effective_duration": duration,
    }
    restored = video.restore_video_prepared(
        directory, source_path=source, start=start, duration=duration
    )
    assert len(restored["_frames"]) == expected
    # New caches reject a different request, even within the same CFR frame.
    with pytest.raises(ValueError, match="interval changed"):
        video.restore_video_prepared(
            directory, source_path=source, start=start, duration=duration + 0.01
        )
    del report["source_selection"]
    report_path.write_text(json.dumps(report))
    legacy = video.restore_video_prepared(
        directory, source_path=source, start=start, duration=duration
    )
    assert len(legacy["_frames"]) == expected
    with pytest.raises(ValueError, match="interval changed"):
        video.restore_video_prepared(
            directory, source_path=source, start=start, duration=duration + 0.1
        )
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        video.restore_video_prepared(
            directory, source_path=source, start=start, duration=duration
        )


def test_audio_shorter_than_video_does_not_truncate_render(tmp_path):
    source, _, predictor = fixture(tmp_path)
    mixed = tmp_path / "short-audio.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-stream_loop",
            "4",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=400:duration=0.2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            str(mixed),
        ],
        check=True,
    )
    predictor.masks *= 5
    prepared = video.prepare_video(
        mixed,
        tmp_path / "prepared",
        SimpleNamespace(confidence_threshold=0.5),
        predictor=predictor,
        duration=2,
    )
    result = video.render_video(
        prepared, None, tmp_path / "render", target_id=0, mode="blur"
    )
    for key in ("output_path", "diagnostic_path"):
        assert (
            video.verify_encoded_output(Path(result[key]), 20)["encoded_frame_count"]
            == 20
        )
        assert any(
            stream["codec_type"] == "audio"
            for stream in video.probe(result[key])["streams"]
        )


def test_prepare_releases_decoder_when_cut_check_fails(tmp_path, monkeypatch):
    source, _, predictor = fixture(tmp_path)
    opened = []

    def fail(frames):
        _ = frames[0]
        opened.append(frames)
        raise RuntimeError("cut check failure")

    monkeypatch.setattr(video, "_check_continuous", fail)
    with pytest.raises(RuntimeError, match="cut check failure"):
        video.prepare_video(
            source,
            tmp_path / "prepared",
            SimpleNamespace(confidence_threshold=0.5),
            predictor=predictor,
            duration=0.4,
        )
    assert opened[0].capture is None and not opened[0].decoded


def test_render_releases_decoder_when_geometry_fails(tmp_path, monkeypatch):
    prepared, template, _ = prepare(tmp_path)

    def fail(*args, **kwargs):
        assert prepared["_frames"].capture is not None
        raise RuntimeError("geometry failure")

    monkeypatch.setattr(video, "propose_scene_geometry", fail)
    with pytest.raises(RuntimeError, match="geometry failure"):
        video.render_video(prepared, template, tmp_path / "render", target_id=0)
    assert prepared["_frames"].capture is None and not prepared["_frames"].decoded


def test_render_reaps_first_encoder_when_second_cannot_start(tmp_path, monkeypatch):
    import io

    prepared, _, _ = prepare(tmp_path)
    first = SimpleNamespace(stdin=io.BytesIO(), stderr=io.BytesIO(), stopped=False)
    first.poll = lambda: 0 if first.stopped else None
    first.kill = lambda: setattr(first, "stopped", True)
    first.wait = lambda: 0
    starts = []

    def start(*args):
        starts.append(args)
        if len(starts) == 1:
            return first
        raise OSError("second encoder could not start")

    monkeypatch.setattr(video, "_writer", start)
    with pytest.raises(OSError, match="second encoder"):
        video.render_video(
            prepared, None, tmp_path / "render", target_id=0, mode="blur"
        )
    assert first.stopped and first.stdin.closed and first.stderr.closed
