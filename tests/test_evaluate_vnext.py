import hashlib
import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from apexveil.estimation import (
    DeterministicAppearance,
    DeterministicBoundary,
    DeterministicDegradation,
    Estimators,
)
from apexveil.imaging.compositor import load_template
from apexveil.imaging.degradation import DegradationEstimate
from apexveil.imaging.geometry import SurfaceGeometry
from apexveil.imaging.photometry import TemporalPhotometry
from apexveil.video_pipeline import render_surface_pair
from scripts.evaluate_vnext import (
    _frame_seed,
    _is_replaceable,
    _render_replacement,
    render_vnext,
)


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
    source_video = root / "shots" / "shot_000.mkv"
    _write_video(source_video, frames)
    resource_sha256 = hashlib.sha256(source_video.read_bytes()).hexdigest()
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
        json.dumps(
            {
                "shot_id": "shot_000",
                "resource_sha256": resource_sha256,
                "frame_count": 5,
                "frames": records,
            }
        )
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
    (root / "tracking" / "shot_000" / "report.json").unlink()
    result = render_vnext(
        root,
        tmp_path / "passthrough",
        template_path=template,
        shot_ids=["shot_000"],
        workers=1,
        estimators=estimators,
    )
    assert all(row["status"] == "passthrough_no_plate" for row in result["frames"])
    assert result["shots"][0]["tracking"]["status"] == "not_applicable"
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
    assert np.mean(np.abs(right.astype(int) - source_frames[0].astype(int))) < 42


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


class MissingGeometry:
    def estimate(self, rgb, visible_mask, aspect, *, observed_quad=None):
        return SurfaceGeometry(
            None,
            visible_mask,
            0.0,
            ["insufficient_full_plane_evidence"],
            {"method": "missing_geometry_test"},
        )


def test_rejected_quad_keeps_replace_rgb_while_blur_variant_redacts(tmp_path):
    root, template, estimators, _ = _fixture(tmp_path)
    estimators = Estimators(
        geometry=MissingGeometry(),
        boundary=estimators.boundary,
        degradation=estimators.degradation,
        appearance=estimators.appearance,
        provider_id="missing-geometry-test",
    )
    result = render_vnext(
        root,
        tmp_path / "rejected",
        template_path=template,
        shot_ids=["shot_000"],
        workers=1,
        estimators=estimators,
    )
    action = next(
        action
        for action in result["frames"][0]["actions"]
        if action.get("object_id") == 0 and action.get("action") != "blur_variant"
    )
    assert action["action"] == "replacement_rejected_geometry"
    assert action["quad_rejection_reasons"] == ["quadrilateral_missing"]
    assert result["frames"][0]["status"] == "needs_review_replacement_rejected"

    decoded = {}
    for variant in ("replace", "blur"):
        output = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                result["output_paths"][variant],
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
        decoded[variant] = np.frombuffer(output, np.uint8).reshape(120, 288, 3)
    left = decoded["replace"][48:, :96].astype(np.int16)
    replace_right = decoded["replace"][48:, 192:288].astype(np.int16)
    blur_right = decoded["blur"][48:, 192:288].astype(np.int16)
    assert np.mean(np.abs(left[26:44, 28:69] - replace_right[26:44, 28:69])) < 4
    assert np.mean(np.abs(left[26:44, 28:69] - blur_right[26:44, 28:69])) > 4


def test_core_valid_quad_is_not_blocked_by_uncalibrated_flags_or_confidence():
    mask = np.zeros((24, 32), np.uint8)
    quad = np.array([[2, 2], [20, 2], [20, 14], [2, 14]], np.float32)
    estimate = SurfaceGeometry(
        quad,
        mask,
        0.01,
        ["partially_out_of_frame", "insufficient_boundary_support"],
        {},
    )
    accepted, reasons = _is_replaceable(estimate)
    assert accepted is True
    assert reasons == []


def test_vnext_noise_seed_matches_shared_video_formula():
    assert _frame_seed(100, 0) == 100 * 1_000_003 + 9_176
    assert _frame_seed(100, 4) == 100 * 1_000_003 + 5 * 9_176


class ZeroDegradation:
    def estimate(self, rgb, mask, quad):
        return DegradationEstimate(0.0, 0.0, 0.0, {"method": "zero-test"})


def test_zero_degradation_matches_shared_video_compositor(tmp_path):
    _root, template_path, estimators, _ = _fixture(tmp_path)
    template = load_template(template_path)
    estimators = Estimators(
        geometry=estimators.geometry,
        boundary=estimators.boundary,
        degradation=ZeroDegradation(),
        appearance=estimators.appearance,
        provider_id="zero-degradation-test",
    )
    rgb = np.full((72, 96, 3), 80, np.uint8)
    mask = np.zeros((72, 96), np.uint8)
    mask[26:44, 28:69] = 1
    yy, xx = np.mgrid[:18, :41]
    rgb[26:44, 28:69] = np.stack(
        [100 + xx % 80, 110 + yy % 70, 120 + (xx + yy) % 60], axis=-1
    ).astype(np.uint8)
    quad = np.array([[28, 26], [68, 26], [68, 43], [28, 43]], np.float32)
    geometry = SurfaceGeometry(quad, mask, 1.0, [], {"cut_before": True})
    photometry = TemporalPhotometry(30)
    core = render_surface_pair(
        rgb,
        mask,
        geometry,
        geometry,
        template,
        estimators,
        photometry,
        100 * 1_000_003 + 9_176,
    )
    actual, metadata, support = _render_replacement(
        rgb,
        mask,
        quad,
        template,
        estimators,
        TemporalPhotometry(30),
        100 * 1_000_003 + 9_176,
    )
    assert actual is not None
    assert metadata["blur_sigma_px"] == 0.0
    assert np.array_equal(actual, core[1])
    assert support["support"].any()


def test_manifest_exclusion_overrides_primary_for_both_variants(tmp_path):
    root, template, estimators, _ = _fixture(tmp_path)
    manifest_path = root / "shot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shots"][0]["exclude_ids"] = [0, 8]
    manifest_path.write_text(json.dumps(manifest))
    result = render_vnext(
        root,
        tmp_path / "excluded-primary",
        template_path=template,
        shot_ids=["shot_000"],
        workers=1,
        estimators=estimators,
    )
    row = result["frames"][0]
    assert row["primary_ids"] == [0]
    assert row["selected_ids"] == []
    assert row["exclude_ids"] == [0, 8]
    assert not any(
        action.get("object_id") == 0 and action["action"] in {"replace", "blur_variant"}
        for action in row["actions"]
    )
    decoded = {}
    for variant, path in result["output_paths"].items():
        raw = subprocess.run(
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
        decoded[variant] = np.frombuffer(raw, np.uint8).reshape(120, 288, 3)
    for variant in ("replace", "blur"):
        left = decoded[variant][48:, :96].astype(np.int16)
        right = decoded[variant][48:, 192:288].astype(np.int16)
        assert np.mean(np.abs(left[26:44, 28:69] - right[26:44, 28:69])) < 4
        assert np.mean(np.abs(left[49:59, 74:86] - right[49:59, 74:86])) < 4


def test_tracking_resource_hash_rejects_stale_and_wrong_provenance(tmp_path):
    root, template, estimators, frames = _fixture(tmp_path)
    report_path = root / "tracking" / "shot_000" / "report.json"
    report = json.loads(report_path.read_text())
    report["resource_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="resource hash mismatch"):
        render_vnext(
            root,
            tmp_path / "wrong-report-hash",
            template_path=template,
            shot_ids=["shot_000"],
            workers=1,
            estimators=estimators,
        )

    root, template, estimators, frames = _fixture(tmp_path / "replacement")
    changed = [frame.copy() for frame in frames]
    changed[0][0, 0] = [1, 2, 3]
    _write_video(root / "shots" / "shot_000.mkv", changed)
    with pytest.raises(ValueError, match="resource hash mismatch"):
        render_vnext(
            root,
            tmp_path / "same-size-replacement",
            template_path=template,
            shot_ids=["shot_000"],
            workers=1,
            estimators=estimators,
        )
