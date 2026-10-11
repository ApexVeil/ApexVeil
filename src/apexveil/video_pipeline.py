"""Bounded continuous-shot video experiment using the shared image compositor.

This entry point deliberately accepts at most four seconds. It produces CFR
experiments, frame diagnostics, and original/per-frame/temporal comparisons.
Production VFR/HDR, shot detection and long-job recovery are not implemented.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import load_settings
from .estimation import get_default_estimators
from .imaging.compositor import load_template, redact_surface, replace_surface
from .imaging.offline_geometry import refine_shot_geometry
from .imaging.photometry import TemporalPhotometry
from .imaging.temporal import rectified_visible_reference


def command(args):
    result = subprocess.run(
        [str(x) for x in args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Command failed: {args[0]}\n{result.stderr[-4000:]}")
    return result.stdout


def probe(path):
    return json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                path,
            ]
        )
    )


def timestamp_mapping(frame_count, fps, source_start):
    if frame_count < 0 or fps <= 0 or source_start < 0:
        raise ValueError("Invalid video time mapping")
    return [
        {
            "frame_index": i,
            "output_time_seconds": i / fps,
            "source_sample_time_seconds": source_start + i / fps,
        }
        for i in range(frame_count)
    ]


def mux_audio(silent_video, audio, destination):
    args = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        silent_video,
    ]
    if audio is not None:
        args += [
            "-i",
            audio,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-shortest",
        ]
    else:
        args += ["-map", "0:v:0", "-c", "copy"]
    # No automatic paid/remote service and no silent audio transcoding fallback.
    args += ["-movflags", "+faststart", destination]
    command(args)


def encode_frames(directory, fps, output, audio=None):
    silent = output.with_name(output.stem + "-silent.mp4")
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            directory / "%06d.png",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            silent,
        ]
    )
    mux_audio(silent, audio, output)
    silent.unlink()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_cache_provenance(tracking_root, clip, source_hash, start, duration):
    manifest = tracking_root / "input-provenance.json"
    if not manifest.is_file():
        raise ValueError(
            "Tracking cache lacks input-provenance.json; rerun tracking instead of guessing"
        )
    data = json.loads(manifest.read_text())
    if (
        data.get("clip_sha256") != file_sha256(clip)
        or data.get("source_sha256") != source_hash
    ):
        raise ValueError("Tracking cache content hash does not match source/clip")
    if (
        abs(float(data["start"]) - start) > 1e-6
        or abs(float(data["duration"]) - duration) > 1e-6
    ):
        raise ValueError("Tracking cache interval does not match requested interval")
    return data


def selected_masks(tracking_root, frame_count, target_id):
    report = json.loads((tracking_root / "report.json").read_text())
    frames = {int(f["frame_index"]): f for f in report["frames"]}
    ids = sorted({int(o["object_id"]) for f in frames.values() for o in f["objects"]})
    if target_id is None:
        if len(ids) != 1:
            raise ValueError(f"Choose --target-id; observed track IDs: {ids}")
        target_id = ids[0]
    if target_id not in ids:
        raise ValueError("Selected track ID was not detected")
    masks = {}
    for i in range(frame_count):
        objects = frames.get(i, {}).get("objects", [])
        selected = [o for o in objects if int(o["object_id"]) == target_id]
        if selected:
            masks[i] = tracking_root / f"frame_{i:06d}" / selected[0]["mask"]
    return masks, target_id, report


def audio_packet_hashes(path):
    data = json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_packets",
                "-show_entries",
                "packet=data_hash",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                path,
            ]
        )
    )
    return [packet["data_hash"] for packet in data.get("packets", [])]


def verify_source_audio(source, audio, start, duration):
    data = json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-read_intervals",
                f"{start}%{start + duration + 1}",
                "-select_streams",
                "a:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,duration_time,data_hash",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                source,
            ]
        )
    )
    selected = [
        p
        for p in data.get("packets", [])
        if "pts_time" in p and start <= float(p["pts_time"]) < start + duration
    ]
    copied = audio_packet_hashes(audio)
    if not selected or [p["data_hash"] for p in selected] != copied:
        raise RuntimeError(
            "Extracted audio does not match source packets in the requested absolute interval"
        )
    return {
        "absolute_source_packet_hashes_match": True,
        "packets": len(selected),
        "first_source_pts_seconds": float(selected[0]["pts_time"]),
        "last_source_pts_seconds": float(selected[-1]["pts_time"]),
        "first_packet_offset_ms": (float(selected[0]["pts_time"]) - start) * 1000,
    }


def verify_encoded_output(path, expected_frames, audio=None):
    data = probe(path)
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    count = stream.get("nb_frames")
    if count in {None, "N/A"}:
        count_data = json.loads(
            command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "json",
                    path,
                ]
            )
        )
        count = count_data["streams"][0]["nb_read_frames"]
    if int(count) != expected_frames:
        raise RuntimeError(
            f"Encoded frame count mismatch: {path.name}: {count} != {expected_frames}"
        )
    result = {
        "encoded_frame_count": int(count),
        "video_duration_seconds": float(stream["duration"]),
        "container_duration_seconds": float(data["format"]["duration"]),
    }
    if audio is not None:
        original, copied = audio_packet_hashes(audio), audio_packet_hashes(path)
        offsets = [
            i
            for i in range(min(3, len(original)))
            if original[i : i + len(copied)] == copied
        ]
        if not copied or not offsets:
            raise RuntimeError(
                "Copied audio packet content did not match extracted source track"
            )
        result["audio_packets"] = {
            "content_hashes_match": True,
            "source_packets": len(original),
            "output_packets": len(copied),
            "source_first_packet_index": offsets[0],
        }
    return result


def make_comparison(
    original_paths, per_frame_dir, temporal_dir, frame_reports, output_dir, labels=None
):
    output_dir.mkdir(exist_ok=True)
    labels = labels or [
        "Original / CFR segment",
        "SAM tracked / framewise composition",
        "SAM tracked / temporal composition",
    ]
    if len(labels) != 3:
        raise ValueError("Comparison requires three labels")
    quads = [
        np.asarray(f["temporal_quad"], np.float32)
        for f in frame_reports
        if f["temporal_quad"] is not None
    ]
    if quads:
        joined = np.concatenate(quads)
        center = (joined.min(axis=0) + joined.max(axis=0)) / 2
        width = max(160.0, (joined[:, 0].max() - joined[:, 0].min()) * 1.8)
        height = max(width * 0.56, (joined[:, 1].max() - joined[:, 1].min()) * 2.0)
    else:
        with Image.open(original_paths[0]) as first:
            center = np.array(first.size) / 2
            width, height = first.size
    tiles_to_save = []
    for idx, path in enumerate(original_paths):
        board = Image.new("RGB", (1920, 420), "#14171d")
        for slot, (label, source) in enumerate(
            [
                (labels[0], path),
                (labels[1], per_frame_dir / path.name),
                (labels[2], temporal_dir / path.name),
            ]
        ):
            with Image.open(source) as image:
                x0, y0 = (
                    max(0, int(center[0] - width / 2)),
                    max(0, int(center[1] - height / 2)),
                )
                x1, y1 = (
                    min(image.width, int(center[0] + width / 2)),
                    min(image.height, int(center[1] + height / 2)),
                )
                patch = image.crop((x0, y0, x1, y1))
                patch = ImageOps.contain(patch, (640, 360), Image.Resampling.LANCZOS)
                board.paste(
                    patch,
                    (
                        slot * 640 + (640 - patch.width) // 2,
                        30 + (360 - patch.height) // 2,
                    ),
                )
            ImageDraw.Draw(board).text(
                (slot * 640 + 12, 8),
                label,
                fill="white",
                font=ImageFont.load_default(size=17),
            )
        ImageDraw.Draw(board).text(
            (1300, 398),
            f"Frame {idx} | {frame_reports[idx]['status']}",
            fill="#efaa80" if frame_reports[idx]["flags"] else "#aac4bd",
        )
        board.save(output_dir / path.name, compress_level=2)
        if idx in {0, len(original_paths) // 2, len(original_paths) - 1}:
            tiles_to_save.append(board)
    contact = Image.new("RGB", (1920, 420 * len(tiles_to_save)), "#14171d")
    for i, board in enumerate(tiles_to_save):
        contact.paste(board, (0, i * 420))
    contact.save(output_dir.parent / "comparison-contact.jpg", quality=94)


class LazyRGBFrames:
    """Re-decodable frame sequence; holds one RGB frame rather than a whole shot."""

    def __init__(self, source, frame_count=None):
        self.paths = list(source) if frame_count is None else None
        self.count = len(self.paths) if self.paths is not None else frame_count
        self.capture = None if self.paths is not None else cv2.VideoCapture(str(source))
        self.cached_index = None
        self.cached = None
        self.next_index = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index == self.cached_index:
            return self.cached
        if self.paths is not None:
            with Image.open(self.paths[index]) as image:
                frame = np.array(image.convert("RGB"))
        else:
            if self.next_index != index:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self.capture.read()
            if not ok:
                raise RuntimeError(f"Video decoder ended at frame {index}")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.next_index = index + 1
        self.cached_index, self.cached = index, frame
        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.cached = None


def estimate_geometry_track(frames, masks, aspect, *, estimators=None, identity=None):
    """Use the injected observation provider once, then one offline refinement."""
    estimators = estimators or get_default_estimators()
    if len(frames) != len(masks):
        raise ValueError("Frames and native masks must have equal length")
    observations = []
    for i in range(len(frames)):
        rgb = frames[i]
        if masks[i].shape != rgb.shape[:2]:
            raise ValueError("Cached mask dimensions do not match decoded frames")
        observations.append(estimators.geometry.estimate(rgb, masks[i], aspect))
    refined = refine_shot_geometry(
        frames, masks, aspect, surface_estimates=observations, identity=identity
    )
    for estimate in refined:
        estimate.diagnostics["flags"] = list(estimate.flags)
        estimate.diagnostics["status"] = (
            "needs_review"
            if estimate.quad is None
            else "motion_fused"
            if len(estimate.diagnostics["anchors"]) > 1
            else "observed"
        )
    return observations, refined


def geometry_review_flags(estimate):
    informational = {"offline_motion_refined", "temporally_supported_full_plane"}
    if (
        "temporally_supported_full_plane" in estimate.flags
        and estimate.quad is not None
    ):
        informational |= {"partially_out_of_frame", "insufficient_boundary_support"}
    return [flag for flag in estimate.flags if flag not in informational]


def add_declared_support(full_support, detail):
    if detail:
        x0, y0, x1, y1 = detail["bbox_xyxy"]
        full_support[y0:y1, x0:x1] |= detail["support"]


def render_surface_pair(
    rgb, mask, observation, estimate, template, estimators, photometry, noise_seed
):
    """Shared product renderer for independent and offline geometry comparisons."""
    baseline, fused = rgb.copy(), rgb.copy()
    baseline_meta, temporal_meta = {}, {}
    support_a, support_b = {}, {}
    flags = geometry_review_flags(estimate)
    if not mask.any():
        photometry.reset()
        flags.append("target_not_visible")
        return (
            baseline,
            fused,
            baseline_meta,
            temporal_meta,
            {},
            flags,
            support_a,
            support_b,
        )
    if np.percentile(rgb[mask > 0].mean(axis=1) / 255, 97) < 0.04:
        flags.append("very_dark_surface")
    if estimate.diagnostics.get("cut_before"):
        photometry.reset()
    q = estimate.quad if estimate.quad is not None else observation.quad
    focus = {"sigma_px": 0.35, "reliable": False}
    degradation = None
    if q is not None:
        degradation = estimators.degradation.estimate(rgb, mask, q)
        focus = {**degradation.diagnostics, "sigma_px": float(degradation.sigma_px)}
    if observation.quad is not None:
        baseline, baseline_meta = replace_surface(
            rgb,
            mask,
            observation.quad,
            template,
            focus["sigma_px"],
            estimators=estimators,
            noise_seed=noise_seed,
            edit_support_out=support_a,
        )
    if estimate.quad is not None:
        height, width = template.alpha.shape
        reference, coverage = rectified_visible_reference(
            rgb, mask, estimate.quad, (width, height)
        )
        if reference is not None:
            fused, temporal_meta = replace_surface(
                rgb,
                mask,
                estimate.quad,
                template,
                focus["sigma_px"],
                appearance_reference=reference,
                appearance_parameter_filter=photometry.filter,
                degradation_estimate=degradation,
                estimators=estimators,
                noise_seed=noise_seed,
                edit_support_out=support_b,
            )
            temporal_meta["temporal_appearance"] = {
                "reference_average_applied": False,
                "parameter_filter": "TemporalPhotometry",
                "geometry_filter": "offline_only",
            }
            temporal_meta["reference_visible_fraction"] = coverage
        else:
            flags.append("replacement_skipped_no_appearance")
    else:
        photometry.reset()
        flags.append("replacement_skipped_no_geometry")
    for metadata in (baseline_meta, temporal_meta):
        if metadata.get("boundary", {}).get("needs_review"):
            flags.append("boundary_needs_review")
    return (
        baseline,
        fused,
        baseline_meta,
        temporal_meta,
        focus,
        flags,
        support_a,
        support_b,
    )


def run_video(
    source,
    start,
    duration,
    output,
    settings,
    template_path,
    tracking_cache=None,
    cached_clip=None,
    target_id=None,
    exclude_ids=(),
    *,
    estimators=None,
):
    if not 0 < duration <= 4 or start < 0:
        raise ValueError(
            "This experimental entry point requires 0 < duration <= 4 seconds"
        )
    if bool(tracking_cache) != bool(cached_clip):
        raise ValueError("Cache reuse requires both --tracking-cache and --cached-clip")
    estimators = estimators or get_default_estimators()
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    begun = time.perf_counter()
    source_root = Path(__file__).parent
    implementation_hashes = {
        name: file_sha256(source_root / name)
        for name in (
            "video_pipeline.py",
            "imaging/temporal.py",
            "imaging/compositor.py",
            "imaging/appearance.py",
            "imaging/photometry.py",
            "imaging/geometry.py",
            "imaging/focus.py",
            "imaging/offline_geometry.py",
            "imaging/boundary.py",
            "imaging/degradation.py",
            "estimation.py",
        )
    }
    source_hash = file_sha256(source)
    source_info = probe(source)
    video_stream = next(s for s in source_info["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(video_stream.get("r_frame_rate", "30/1")))
    if not 1 <= fps <= 60:
        raise ValueError("Unsupported source frame rate for bounded experiment")
    expected_frames = int(np.ceil(duration * fps - 1e-6))
    audio_streams = [s for s in source_info["streams"] if s["codec_type"] == "audio"]
    clip = output / "segment-cfr.mp4"
    if cached_clip:
        verify_cache_provenance(
            tracking_cache, cached_clip, source_hash, start, duration
        )
        cached_report = json.loads((tracking_cache / "report.json").read_text())
        if cached_report["resource_name"] != cached_clip.name:
            raise ValueError(
                "Tracking cache does not reference the supplied cached clip"
            )
        shutil.copy2(cached_clip, clip)
    else:
        command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                start,
                "-i",
                source,
                "-t",
                duration,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"fps={fps}",
                "-frames:v",
                str(expected_frames),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                clip,
            ]
        )
    clip_info = probe(clip)
    clip_stream = next(s for s in clip_info["streams"] if s["codec_type"] == "video")
    clip_fps = float(Fraction(clip_stream["avg_frame_rate"]))
    if (
        abs(clip_fps - fps) > 0.001
        or abs(float(clip_info["format"]["duration"]) - duration) > 0.1
    ):
        raise ValueError(
            "Cached/decoded clip timing does not match the requested interval"
        )
    audio = output / "source-audio.mka" if audio_streams else None
    if audio:
        command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source,
                "-ss",
                start,
                "-t",
                duration,
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "copy",
                audio,
            ]
        )
    source_audio_verification = (
        verify_source_audio(source, audio, start, duration) if audio else None
    )
    frames_dir = output / "original-frames"
    frames_dir.mkdir(exist_ok=True)
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            clip,
            "-an",
            "-start_number",
            "0",
            "-threads",
            "2",
            "-frames:v",
            str(expected_frames),
            frames_dir / "%06d.png",
        ]
    )
    paths = sorted(frames_dir.glob("*.png"))
    if not paths or len(paths) > 240:
        raise ValueError("Unexpected bounded frame count")
    if tracking_cache:
        tracking_root = tracking_cache
    else:
        from .backends.sam31 import smoke

        tracking_root = output / "tracking"
        smoke(settings, clip, tracking_root, "license plate", len(paths))
        (tracking_root / "input-provenance.json").write_text(
            json.dumps(
                {
                    "clip_sha256": file_sha256(clip),
                    "source_sha256": source_hash,
                    "start": start,
                    "duration": duration,
                    "frame_count": len(paths),
                },
                indent=2,
            )
        )
    masks, chosen_id, tracking = selected_masks(tracking_root, len(paths), target_id)
    frame_objects = {int(f["frame_index"]): f["objects"] for f in tracking["frames"]}
    template = load_template(template_path)
    width, height = template.alpha.shape[1], template.alpha.shape[0]
    per_dir, temporal_dir = output / "per-frame", output / "temporal"
    per_dir.mkdir(exist_ok=True)
    temporal_dir.mkdir(exist_ok=True)
    frame_sequence = LazyRGBFrames(paths)
    mask_sequence = [
        (np.array(Image.open(masks[i])) > 0).astype(np.uint8)
        if i in masks
        else np.zeros(frame_sequence[i].shape[:2], np.uint8)
        for i in range(len(paths))
    ]
    observations, estimates = estimate_geometry_track(
        frame_sequence,
        mask_sequence,
        width / height,
        estimators=estimators,
        identity=chosen_id,
    )
    frame_sequence.close()
    (output / "frames.jsonl").write_text("")
    photometry = TemporalPhotometry(fps)
    timeline = timestamp_mapping(len(paths), fps, start)
    reports = []
    render_started = time.perf_counter()
    for idx, path in enumerate(paths):
        rgb = np.array(Image.open(path).convert("RGB"))
        mask = mask_sequence[idx]
        observation, estimate = observations[idx], estimates[idx]
        observed, quality = observation.quad, observation.diagnostics
        (
            baseline,
            fused,
            baseline_meta,
            temporal_meta,
            focus,
            flags,
            support_a,
            support_b,
        ) = render_surface_pair(
            rgb,
            mask,
            observation,
            estimate,
            template,
            estimators,
            photometry,
            noise_seed=idx * 1000003 + (chosen_id + 1) * 9176,
        )
        baseline_support = np.zeros(mask.shape, bool)
        temporal_support = np.zeros(mask.shape, bool)
        add_declared_support(baseline_support, support_a)
        add_declared_support(temporal_support, support_b)
        native_support = mask > 0
        redacted_other_ids = []
        for other in frame_objects.get(idx, []):
            other_id = int(other["object_id"])
            if other_id == chosen_id or other_id in exclude_ids:
                continue
            other_mask = (
                np.array(Image.open(tracking_root / f"frame_{idx:06d}" / other["mask"]))
                > 0
            ).astype(np.uint8)
            if other_mask.any():
                baseline, _ = redact_surface(baseline, other_mask, "blur")
                fused, _ = redact_surface(fused, other_mask, "blur")
                baseline_support |= other_mask > 0
                temporal_support |= other_mask > 0
                native_support |= other_mask > 0
                redacted_other_ids.append(other_id)
        for name, image, allowed in [
            ("per-frame", baseline, baseline_support),
            ("temporal", fused, temporal_support),
        ]:
            outside = int(np.count_nonzero(np.any(image != rgb, axis=2) & ~allowed))
            if outside:
                raise RuntimeError(f"{name} changed {outside} protected pixels")
        Image.fromarray(baseline).save(per_dir / path.name, compress_level=2)
        Image.fromarray(fused).save(temporal_dir / path.name, compress_level=2)
        record = {
            **timeline[idx],
            "target_id": chosen_id,
            "status": "needs_review" if flags else "rendered_unreviewed",
            "flags": sorted(set(flags)),
            "visible_pixels": int(mask.sum()),
            "other_redacted_ids": redacted_other_ids,
            "quality": quality,
            "observed_quad": observed.tolist() if observed is not None else None,
            "temporal_quad": estimate.quad.tolist()
            if estimate.quad is not None
            else None,
            "geometry_state": estimate.diagnostics,
            "focus": focus,
            "per_frame_appearance": baseline_meta,
            "temporal_appearance": temporal_meta,
            "outside_visible_masks_changed_pixels": int(
                np.count_nonzero(np.any(fused != rgb, axis=2) & ~native_support)
            ),
            "outside_edit_support_changed_pixels": 0,
            "estimator": estimators.metadata(),
        }
        reports.append(record)
        with (output / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    rendering_seconds = time.perf_counter() - render_started
    make_comparison(paths, per_dir, temporal_dir, reports, output / "comparison-frames")
    encode_frames(frames_dir, fps, output / "original.mp4", audio)
    encode_frames(per_dir, fps, output / "per-frame.mp4", audio)
    encode_frames(temporal_dir, fps, output / "temporal.mp4", audio)
    encode_frames(
        output / "comparison-frames", fps, output / "comparison-detail.mp4", audio
    )
    encoded_verification = {
        name: verify_encoded_output(output / name, len(paths), audio)
        for name in (
            "original.mp4",
            "per-frame.mp4",
            "temporal.mp4",
            "comparison-detail.mp4",
        )
    }
    final_info = probe(output / "temporal.mp4")
    motion_records = [
        f["geometry_state"]
        for f in reports
        if "observation_motion_residual_px" in f["geometry_state"]
    ]
    result = {
        "kind": "bounded_video_compositing_experiment",
        "estimator": estimators.metadata(),
        "geometry_policy": "per_identity_offline_bidirectional_consensus",
        "source_name": source.name,
        "source_sha256": source_hash,
        "source_interval_seconds": [start, start + duration],
        "frame_count": len(paths),
        "selected_target_id": chosen_id,
        "excluded_track_ids": list(exclude_ids),
        "other_track_policy": "blur",
        "fps": fps,
        "source_avg_frame_rate": video_stream.get("avg_frame_rate"),
        "source_nominal_frame_rate": video_stream.get("r_frame_rate"),
        "timing_policy": "CFR resampling; source sample time = requested start + index/fps",
        "native_variable_frame_timestamps_preserved": False,
        "source_video_reencoded_for_experiment": True,
        "cache_reused": bool(tracking_cache),
        "tracking_ran_this_job": not bool(tracking_cache),
        "implementation_files_sha256": implementation_hashes,
        "cache_clip_sha256": file_sha256(clip),
        "audio": {
            "source_first_track_copied": bool(audio),
            "source_packet_verification": source_audio_verification,
            "source_codec": audio_streams[0]["codec_name"] if audio_streams else None,
            "output_codec": next(
                (
                    s["codec_name"]
                    for s in final_info["streams"]
                    if s["codec_type"] == "audio"
                ),
                None,
            ),
            "packet_boundary_precision": "AAC/other packet boundaries; not sample-accurate editing",
        },
        "tracking_inference_seconds": tracking["inference_seconds"],
        "render_and_png_seconds": rendering_seconds,
        "total_seconds": time.perf_counter() - begun,
        "needs_review_frames": sum(bool(r["flags"]) for r in reports),
        "rendered_frames": sum(bool(r["temporal_appearance"]) for r in reports),
        "motion_fused_frames": sum(
            r["geometry_state"]["status"] == "motion_fused" for r in reports
        ),
        "motion_residual_diagnostic": {
            "not_ground_truth_quality": True,
            "observation_median_px": float(
                np.median([r["observation_motion_residual_px"] for r in motion_records])
            )
            if motion_records
            else None,
            "fused_median_px": float(
                np.median([r["fused_motion_residual_px"] for r in motion_records])
            )
            if motion_records
            else None,
        },
        "encoded_verification": encoded_verification,
        "ai_review": "not_run",
        "frame_reports": "frames.jsonl",
        "public_redistribution": False,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--tracking-cache", type=Path)
    parser.add_argument("--cached-clip", type=Path)
    parser.add_argument("--target-id", type=int)
    parser.add_argument("--exclude-ids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if (args.output / "frames.jsonl").exists():
        parser.error("Output already contains frame reports; choose a fresh directory")
    result = run_video(
        args.source,
        args.start,
        args.duration,
        args.output,
        load_settings(args.config),
        args.template,
        args.tracking_cache,
        args.cached_clip,
        args.target_id,
        args.exclude_ids,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
