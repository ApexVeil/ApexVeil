import json
import subprocess

import numpy as np
from PIL import Image

from apexveil.estimation import (
    DeterministicAppearance,
    DeterministicBoundary,
    DeterministicDegradation,
    Estimators,
)
from apexveil.imaging.geometry import SurfaceGeometry
from scripts.evaluate_vnext import render_vnext


class FixedGeometry:
    def estimate(self, rgb, visible_mask, aspect, *, observed_quad=None):
        quad = np.array([[28, 26], [68, 26], [68, 43], [28, 43]], np.float32)
        return SurfaceGeometry(
            quad if visible_mask.any() else None,
            visible_mask,
            1.0 if visible_mask.any() else 0.0,
            [],
            {"candidate_quad": quad.tolist(), "method": "test_fixed_quad"},
        )


def _write_video(path, frames, fps=30):
    height, width = frames[0].shape[:2]
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
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(path),
        ],
        input=np.stack(frames).tobytes(),
        check=True,
        capture_output=True,
    )


def _fixture(tmp_path, kind="shot"):
    root = tmp_path / "cache"
    (root / "shots").mkdir(parents=True)
    tracking = root / "tracking" / "shot_000"
    tracking.mkdir(parents=True)
    frames = []
    for index in range(5):
        rng = np.random.default_rng(index)
        frame = np.full((72, 96, 3), 72, np.uint8)
        frame[26:44, 28:69] = 220
        frame[29:41, 34:63] = rng.integers(20, 225, (12, 29, 3), dtype=np.uint8)
        frame[49:59, 74:86] = 150
        frames.append(frame)
        folder = tracking / f"frame_{index:06d}"
        folder.mkdir()
        plate = np.zeros((72, 96), np.uint8)
        plate[26:44, 28:69] = 255
        traffic = np.zeros((72, 96), np.uint8)
        traffic[49:59, 74:86] = 255
        Image.fromarray(plate).save(folder / "target.png")
        Image.fromarray(traffic).save(folder / "traffic.png")
    _write_video(root / "shots" / "shot_000.mkv", frames)
    records = [
        {
            "frame_index": i,
            "objects": [
                {"object_id": 0, "mask": "target.png"},
                {"object_id": 8, "mask": "traffic.png"},
            ],
        }
        for i in range(5)
    ]
    (tracking / "report.json").write_text(
        json.dumps({"shot_id": "shot_000", "frame_count": 5, "frames": records})
    )
    manifest = {
        "schema_version": 1,
        "source_path": str(tmp_path / "unneeded-source.mp4"),
        "source_sha256": "source-hash",
        "source_interval_seconds": [0, 5 / 30],
        "duration_seconds": 5 / 30,
        "fps": 30,
        "frame_count": 5,
        "native_panel_dimensions": [96, 72],
        "output_dimensions": [288, 136],
        "shots": [
            {
                "id": "shot_000",
                "start_frame": 0,
                "end_frame": 5,
                "frame_count": 5,
                "kind": kind,
                "primary_ids": [0] if kind == "shot" else [],
                "exclude_ids": [8],
            }
        ],
    }
    (root / "shot_manifest.json").write_text(json.dumps(manifest))
    template_path = tmp_path / "template.png"
    template = np.zeros((28, 70, 4), np.uint8)
    template[..., :3] = [10, 20, 30]
    template[..., 3] = 255
    Image.fromarray(template).save(template_path)
    estimators = Estimators(
        geometry=FixedGeometry(),
        boundary=DeterministicBoundary(),
        degradation=DeterministicDegradation(),
        appearance=DeterministicAppearance(),
        provider_id="fixed-test",
    )
    return root, template_path, estimators, frames


def test_five_frame_subset_exports_replace_and_blur_with_raw_masks(tmp_path):
    root, template, estimators, _ = _fixture(tmp_path)
    result = render_vnext(
        root,
        tmp_path / "output",
        template_path=template,
        shot_ids=["shot_000"],
        workers=2,
        estimators=estimators,
        seed=17,
    )
    assert result["status"] == "subset_preview"
    assert result["complete_timeline"] is False
    assert result["gpu_used"] is False
    assert result["sam_inference_run"] is False
    assert len(result["frames"]) == 5
    assert result["frames"][0]["primary_ids"] == [0]
    assert result["frames"][0]["exclude_ids"] == [8]
    assert result["frames"][0]["native_returned_ids"] == [0, 8]
    assert any(
        action["object_id"] == 0 and action["action"] == "replace"
        for action in result["frames"][0]["actions"]
    )
    for path in result["output_paths"].values():
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "json",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert (
            int(stream["width"]),
            int(stream["height"]),
            int(stream["nb_frames"]),
        ) == (288, 120, 5)


def test_non_plate_categories_are_passthrough_and_background_only_blurs(tmp_path):
    root, template, estimators, source_frames = _fixture(tmp_path, kind="no_plate")
    result = render_vnext(
        root,
        tmp_path / "passthrough",
        template_path=template,
        shot_ids=["shot_000"],
        workers=1,
        estimators=estimators,
    )
    assert all(row["status"] == "passthrough_no_plate" for row in result["frames"])
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            result["output_paths"]["replace"],
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    panel = np.frombuffer(encoded, np.uint8).reshape(120, 288, 3)
    right = panel[48:, 192:288]
    assert np.mean(np.abs(right.astype(int) - source_frames[0].astype(int))) < 12


def test_background_masks_only_non_excluded_native_ids(tmp_path):
    root, template, estimators, source_frames = _fixture(tmp_path, kind="background")
    result = render_vnext(
        root,
        tmp_path / "background",
        template_path=template,
        shot_ids=["shot_000"],
        workers=1,
        estimators=estimators,
    )
    row = result["frames"][0]
    assert row["primary_ids"] == []
    assert row["exclude_ids"] == [8]
    assert any(
        action["object_id"] == 0 and action["action"] == "blur_other"
        for action in row["actions"]
    )
    assert not any(
        action.get("object_id") == 8 and action.get("action") != "none"
        for action in row["actions"]
    )
    path = result["output_paths"]["replace"]
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    panel = np.frombuffer(encoded, np.uint8).reshape(120, 288, 3)
    right = panel[48:, 192:288].astype(np.int16)
    original = source_frames[0].astype(np.int16)
    changed = np.abs(right - original).mean(axis=2)
    assert changed[28:44, 30:67].mean() > changed[49:59, 74:86].mean()
