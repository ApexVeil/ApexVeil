import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.export_diagnostics import (
    color_native_masks, export_image_diagnostics, export_video_diagnostics,
    manifest_frame_map, stable_color, write_image_index,
)


def test_palette_stable_and_masks_include_excluded_ids():
    assert stable_color(4, "shot_1") == stable_color(4, "shot_1")
    assert stable_color(4, "shot_1") != stable_color(5, "shot_1")
    frame = np.full((24, 32, 3), 80, np.uint8)
    a = np.zeros((24, 32), bool); a[3:10, 2:8] = True
    b = np.zeros((24, 32), bool); b[12:18, 20:27] = True
    result, union = color_native_masks(frame, {1: a, 9: b}, shot_id="s", exclude_ids=[9])
    assert union.sum() == a.sum() + b.sum()
    assert not np.array_equal(result[5, 4], frame[5, 4])
    assert not np.array_equal(result[14, 22], frame[14, 22])


def test_image_export_panels_index_and_input_immutability(tmp_path):
    source = np.full((40, 60, 3), [40, 80, 120], np.uint8)
    replace = np.full_like(source, [150, 70, 30]); blur = np.full_like(source, [20, 160, 60])
    original = source.copy()
    mask = np.zeros((40, 60), bool); mask[12:20, 21:38] = True
    report = export_image_diagnostics(source, replace, blur, {0: mask, 7: np.zeros_like(mask)},
        tmp_path, image_id="sample.JPG", target_id=0, primary_ids=[0], exclude_ids=[7],
        provenance={"kind": "manual_corners_preview", "manual_corners_are_ground_truth": False})
    assert np.array_equal(source, original)
    assert report["empty_native_ids"] == [7]
    assert report["excluded_ids_still_visible"] == [7]
    assert report["provenance"]["kind"] == "manual_corners_preview"
    for name in report["diagnostics"].values():
        with Image.open(tmp_path / name) as im:
            assert im.size == (2160, 524)
    index = write_image_index(tmp_path, [report])
    page = index.read_text()
    assert "data-replace" in page and "data-blur" in page


def _solid(color, n=5):
    return [np.full((24, 32, 3), color, np.uint8) for _ in range(n)]


def test_five_frame_video_layout_native_mask_and_timeline(tmp_path):
    tracking = tmp_path / "tracking"; tracking.mkdir()
    frames = []
    for i in range(5):
        folder = tracking / f"frame_{i:06d}"; folder.mkdir()
        a = np.zeros((24, 32), np.uint8); a[3:10, 2:8] = 255
        b = np.zeros((24, 32), np.uint8); b[12:18, 20:27] = 255
        Image.fromarray(a).save(folder / "native_a.png")
        Image.fromarray(b).save(folder / "native_b.png")
        frames.append({"frame_index": i, "objects": [
            {"object_id": 1, "mask": "native_a.png"},
            {"object_id": 9, "mask": "native_b.png"},
        ]})
    report = {"frame_count": 5, "frames": frames}
    meta = export_video_diagnostics(_solid([20, 30, 180]), report, tracking,
        _solid([30, 190, 50]), _solid([180, 40, 30]), tmp_path / "out",
        shot_id="shot_0", fps=5, target_id=1, primary_ids=[1], exclude_ids=[9])
    assert meta["frame_count"] == 5
    assert meta["native_ids_visible"] == [1, 9]
    assert meta["excluded_ids_still_visible"] == [9]
    for path in meta["outputs"].values():
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames", "-of", "json", path],
            check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (96, 78, 5)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", meta["outputs"]["replace"],
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, capture_output=True).stdout
    decoded = np.frombuffer(raw, np.uint8).reshape(78, 96, 3)
    # Top 24px after the 54px header: left source, center mask overlay, right Replace.
    y = 54 + 5
    assert decoded[y, 4, 2] > decoded[y, 4, 0]  # RGB source is blue-dominant
    assert not np.array_equal(decoded[y, 32 + 4], decoded[y, 4])  # ID 1 overlay
    assert not np.array_equal(decoded[54 + 14, 32 + 22], decoded[54 + 14, 22])  # excluded ID 9 still rendered
    assert decoded[y, 64 + 4, 1] > decoded[y, 64 + 4, 0]  # Replace is green-dominant


def test_manifest_global_to_local_map_checks_contiguous_intervals():
    manifest = {"shots": [
        {"id": "shot_0", "start_frame": 0, "end_frame": 2, "frame_count": 2, "target_id": 3},
        {"id": "shot_1", "start_frame": 2, "end_frame": 5, "frame_count": 3, "exclude_ids": [8]},
    ]}
    mapping = manifest_frame_map(manifest)
    assert mapping[0]["local_frame"] == 0
    assert mapping[2]["shot_id"] == "shot_1" and mapping[2]["local_frame"] == 0
    assert mapping[4]["local_frame"] == 2
    with pytest.raises(ValueError, match="contiguous"):
        manifest_frame_map({"shots": [
            {"id": "a", "start_frame": 0, "end_frame": 1, "frame_count": 1},
            {"id": "b", "start_frame": 2, "end_frame": 3, "frame_count": 1},
        ]})

