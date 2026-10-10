"""Shot-aware, full-timeline comparison export using the shared ApexVeil core.

A reviewed manifest defines true shot boundaries. Native SAM text/tracking resets
only at those boundaries. Lossless shot inputs and streamed H.264 outputs avoid
per-frame RGB image dumps. Unreliable frames and native errors retain their time.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import load_settings
from .estimation import get_default_estimators
from .imaging.compositor import load_template, redact_surface
from .imaging.geometry import SurfaceGeometry
from .imaging.photometry import TemporalPhotometry
from .video_pipeline import (
    LazyRGBFrames,
    add_declared_support,
    command,
    estimate_geometry_track,
    file_sha256,
    mux_audio,
    probe,
    render_surface_pair,
    verify_encoded_output,
    verify_source_audio,
)

PASSTHROUGH = {"no_plate", "black", "transition"}


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def validate_manifest(manifest):
    cursor = 0
    ids = set()
    for shot in manifest["shots"]:
        if shot["start_frame"] != cursor or shot["end_frame"] <= cursor:
            raise ValueError("Shot intervals must be ordered, contiguous and nonempty")
        if shot["frame_count"] != shot["end_frame"] - cursor or shot["id"] in ids:
            raise ValueError("Invalid shot length or duplicate ID")
        if shot["kind"] not in PASSTHROUGH | {"shot", "background"}:
            raise ValueError("Unknown shot kind")
        ids.add(shot["id"])
        cursor = shot["end_frame"]
    if cursor != manifest["frame_count"]:
        raise ValueError("Manifest does not cover the complete timeline")
    if manifest["fps"] != 30:
        raise ValueError("This deliverable contract uses explicit 30 fps CFR")
    return True


def raw_writer(path, width, height, fps, codec="ffv1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
    ]
    if codec == "ffv1":
        args += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-threads", "2"]
    else:
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "60",
            "-keyint_min",
            "60",
            "-sc_threshold",
            "0",
            "-vf",
            "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
        ]
    args += [str(path)]
    return subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


def finish_writer(process):
    process.stdin.close()
    error = process.stderr.read().decode(errors="replace")
    if process.wait():
        raise RuntimeError(f"FFmpeg stream encoding failed: {error[-3000:]}")


def prepare_shots(root, manifest):
    validate_manifest(manifest)
    capture = cv2.VideoCapture(str(root / manifest["master"]))
    cursor = 0
    for shot in manifest["shots"]:
        path = root / "shots" / (shot["id"] + ".mkv")
        ready = path.with_suffix(".ready.json")
        writer = (
            None if ready.exists() else raw_writer(path, 1280, 720, manifest["fps"])
        )
        for _ in range(shot["frame_count"]):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Master ended before frame {cursor}")
            if writer:
                writer.stdin.write(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes())
            cursor += 1
        if writer:
            finish_writer(writer)
            save_json(
                ready, {"frame_count": shot["frame_count"], "sha256": file_sha256(path)}
            )
        print(
            json.dumps({"stage": "prepare", "shot": shot["id"], "end_frame": cursor}),
            flush=True,
        )
    ok, _ = capture.read()
    capture.release()
    if ok or cursor != manifest["frame_count"]:
        raise RuntimeError("Master frame count differs from the reviewed manifest")


def candidate_sheet(root, shot, report):
    records = {f["frame_index"]: f for f in report["frames"]}
    representatives = {}
    for idx, frame in records.items():
        for obj in frame["objects"]:
            oid = obj["object_id"]
            if (
                oid not in representatives
                or obj["visible_pixels"] > representatives[oid][1]["visible_pixels"]
            ):
                representatives[oid] = (idx, obj)
    cap = cv2.VideoCapture(str(root / "shots" / (shot["id"] + ".mkv")))
    cols = 2
    rows = max(1, (len(representatives) + 1) // 2)
    board = Image.new("RGB", (1000, 325 * rows), "#14171d")
    for slot, (oid, (idx, obj)) in enumerate(sorted(representatives.items())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = (
            np.array(
                Image.open(
                    root / "tracking" / shot["id"] / f"frame_{idx:06d}" / obj["mask"]
                )
            )
            > 0
        )
        rgb[mask] = (rgb[mask] * 0.55 + np.array([255, 80, 40]) * 0.45).astype(np.uint8)
        image = Image.fromarray(rgb)
        image.thumbnail((500, 281))
        x = slot % cols * 500
        y = slot // cols * 325
        board.paste(image, (x, y + 35))
        ImageDraw.Draw(board).text(
            (x + 10, y + 8),
            f"{shot['id']} / ID {oid} / frame {idx} / max area {obj['visible_pixels']}",
            fill="white",
        )
    cap.release()
    folder = root / "candidate-sheets"
    folder.mkdir(exist_ok=True)
    board.save(folder / (shot["id"] + ".jpg"), quality=93)


def full_shot_request(session_id, frame_count, confidence_threshold=0.5):
    if frame_count < 1:
        raise ValueError("A shot must contain at least one frame")
    # Upstream's processing order clamps an inclusive end, while its batched
    # detector interprets this field as an exclusive frame count. Passing T-1
    # makes the final batch empty; full-shot propagation must receive T.
    return {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
        "start_frame_index": 0,
        "max_frame_num_to_track": frame_count,
        "output_prob_thresh": confidence_threshold,
    }


def track_shots(root, manifest, settings):
    import torch

    from .backends.sam31 import build_predictor, summarize_output

    torch.set_num_threads(8)
    predictor = None
    for shot in manifest["shots"]:
        if shot["kind"] in PASSTHROUGH:
            continue
        destination = root / "tracking" / shot["id"]
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "report.json").exists():
            continue
        if predictor is None:
            predictor = build_predictor(settings)
        ready = root / "shots" / (shot["id"] + ".ready.json")
        deadline = time.monotonic() + 1800
        while not ready.exists():
            if time.monotonic() > deadline or (root / "prepare.error").exists():
                raise RuntimeError("Shot preparation failed or timed out")
            time.sleep(0.3)
        sid = None
        frames = {}
        started = time.perf_counter()
        error = None
        try:
            session = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(
                        (root / "shots" / (shot["id"] + ".mkv")).resolve()
                    ),
                    "offload_video_to_cpu": True,
                }
            )
            sid = session["session_id"]
            initial = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": "license plate",
                    "output_prob_thresh": settings.confidence_threshold,
                }
            )
            frames[0] = summarize_output(initial, destination / "frame_000000")
            request = full_shot_request(
                sid, shot["frame_count"], settings.confidence_threshold
            )
            for response in predictor.handle_stream_request(request):
                idx = int(response["frame_index"])
                if not 0 <= idx < shot["frame_count"]:
                    raise RuntimeError("Native response escaped shot frame range")
                frames[idx] = summarize_output(
                    response, destination / f"frame_{idx:06d}"
                )
            if len(frames) != shot["frame_count"]:
                raise RuntimeError(
                    f"Native result contains {len(frames)} frames, expected {shot['frame_count']}"
                )
        except Exception as exception:  # noqa: BLE001 -- retain partial native output
            error = f"{type(exception).__name__}: {exception}"
            (destination / "error-trace.txt").write_text(traceback.format_exc())
        finally:
            if sid:
                predictor.handle_request({"type": "close_session", "session_id": sid})
            torch.cuda.empty_cache()
        for i in range(shot["frame_count"]):
            frames.setdefault(
                i, {"frame_index": i, "objects": [], "native_output_missing": True}
            )
        stats = {}
        for frame in frames.values():
            for obj in frame["objects"]:
                stats.setdefault(str(obj["object_id"]), []).append(
                    obj["visible_pixels"]
                )
        report = {
            "shot_id": shot["id"],
            "kind": "sam31_text_frame0_native_forward",
            "point_prompts": False,
            "resource_name": shot["id"] + ".mkv",
            "resource_sha256": file_sha256(root / "shots" / (shot["id"] + ".mkv")),
            "inference_seconds": time.perf_counter() - started,
            "frame_count": shot["frame_count"],
            "frames": [frames[i] for i in sorted(frames)],
            "native_error": error,
            "candidates": {
                k: {
                    "frames": len(a),
                    "median_area": float(np.median(a)),
                    "max_area": int(max(a)),
                }
                for k, a in stats.items()
            },
        }
        save_json(destination / "report.json", report)
        candidate_sheet(root, shot, report)
        print(
            json.dumps(
                {
                    "stage": "tracking",
                    "shot": shot["id"],
                    "candidates": report["candidates"],
                    "error": error,
                }
            ),
            flush=True,
        )


def title_pixels(width=1280, height=48):
    image = Image.new("RGB", (width * 3, height), "#11151b")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    for i, label in enumerate(["RGB", "SAM3.1 + Independent", "ApexVeil"]):
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (
                i * width + (width - (bbox[2] - bbox[0])) / 2,
                (height - (bbox[3] - bbox[1])) / 2 - 3,
            ),
            label,
            font=font,
            fill="#eef1f4",
        )
    return np.array(image)


def select_primary_alias(masks, primary_ids, previous_id=None):
    available = [oid for oid in primary_ids if oid in masks and masks[oid].any()]
    if previous_id in available:
        return previous_id
    return available[0] if available else None


def render_shot(root_string, shot, template_string, *, estimators=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cv2.setNumThreads(2)
    injected_provider = estimators is not None
    estimators = estimators or get_default_estimators()
    root = Path(root_string)
    template = load_template(Path(template_string))
    th, tw = template.alpha.shape
    folder = root / "render"
    folder.mkdir(exist_ok=True)
    video = folder / (shot["id"] + ".mp4")
    report_path = folder / (shot["id"] + ".json")
    tracking_path = root / "tracking" / shot["id"] / "report.json"
    signature = {
        "estimator": estimators.metadata(),
        "provider_injected": injected_provider,
        "geometry_policy": "per_identity_offline_bidirectional_consensus",
        "selection": {
            k: shot.get(k) for k in ["kind", "target_id", "primary_ids", "exclude_ids"]
        },
        "template_sha256": template.metadata["source_sha256"],
        "renderer_sha256": file_sha256(Path(__file__)),
        "tracking_report_sha256": file_sha256(tracking_path)
        if tracking_path.exists()
        else None,
        "core_hashes": {
            n: file_sha256(Path(__file__).parent / n)
            for n in [
                "imaging/geometry.py",
                "imaging/appearance.py",
                "imaging/compositor.py",
                "imaging/temporal.py",
                "imaging/photometry.py",
                "imaging/offline_geometry.py",
                "imaging/boundary.py",
                "imaging/degradation.py",
                "estimation.py",
            ]
        },
    }
    if not injected_provider and video.exists() and report_path.exists():
        cached = json.loads(report_path.read_text())
        if cached.get("signature") == signature:
            return cached
    tracking = (
        json.loads(tracking_path.read_text())
        if tracking_path.exists()
        else {"frames": [], "native_error": None}
    )
    records = {f["frame_index"]: f for f in tracking["frames"]}
    primary_ids_for_shot = shot.get(
        "primary_ids", [] if shot.get("target_id") is None else [shot["target_id"]]
    )
    excluded_for_shot = set(shot.get("exclude_ids", []))
    geometry_by_id = {}
    native_masks_by_id = {}
    if shot["kind"] == "shot" and primary_ids_for_shot:
        frame_sequence = LazyRGBFrames(
            root / "shots" / (shot["id"] + ".mkv"), shot["frame_count"]
        )
        try:
            shape = frame_sequence[0].shape[:2]
            for identity in primary_ids_for_shot:
                if identity in excluded_for_shot:
                    continue
                native_masks = []
                for i in range(shot["frame_count"]):
                    selected = [
                        o
                        for o in records.get(i, {}).get("objects", [])
                        if int(o["object_id"]) == identity
                    ]
                    if selected:
                        path = (
                            root
                            / "tracking"
                            / shot["id"]
                            / f"frame_{i:06d}"
                            / selected[0]["mask"]
                        )
                        native_masks.append(
                            (np.array(Image.open(path)) > 0).astype(np.uint8)
                        )
                    else:
                        native_masks.append(np.zeros(shape, np.uint8))
                native_masks_by_id[identity] = native_masks
                geometry_by_id[identity] = estimate_geometry_track(
                    frame_sequence,
                    native_masks,
                    tw / th,
                    estimators=estimators,
                    identity=identity,
                )
        finally:
            frame_sequence.close()
    cap = cv2.VideoCapture(str(root / "shots" / (shot["id"] + ".mkv")))
    writer = raw_writer(video, 3840, 768, 30, "h264")
    header = title_pixels()
    canvas = np.zeros((768, 3840, 3), np.uint8)
    canvas[:48] = header
    photometry_by_id = {identity: TemporalPhotometry(30) for identity in geometry_by_id}
    counts = Counter()
    flag_counts = Counter()
    started = time.perf_counter()
    previous_primary = None
    journal = folder / (shot["id"] + ".jsonl")
    contacts = []
    try:
        with journal.open("w") as stream:
            for i in range(shot["frame_count"]):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Shot decoder ended at {i}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                baseline = rgb.copy()
                fused = rgb.copy()
                flags = []
                observed = None
                quality = {}
                estimate = None
                focus = {}
                baseline_meta = {}
                temporal_meta = {}
                frame = records.get(i, {"frame_index": i, "objects": []})
                declared_primary = shot.get("target_id")
                primary_ids = shot.get(
                    "primary_ids",
                    [] if declared_primary is None else [declared_primary],
                )
                primary = declared_primary
                excluded = set(shot.get("exclude_ids", []))
                support = np.zeros(rgb.shape[:2], bool)
                baseline_support = np.zeros(rgb.shape[:2], bool)
                temporal_support = np.zeros(rgb.shape[:2], bool)
                alias_switched = False
                masks = {}
                if shot["kind"] not in PASSTHROUGH:
                    for obj in frame["objects"]:
                        oid = int(obj["object_id"])
                        if oid not in excluded:
                            if oid in native_masks_by_id:
                                masks[oid] = native_masks_by_id[oid][i]
                                continue
                            masks[oid] = (
                                np.array(
                                    Image.open(
                                        root
                                        / "tracking"
                                        / shot["id"]
                                        / f"frame_{i:06d}"
                                        / obj["mask"]
                                    )
                                )
                                > 0
                            ).astype(np.uint8)
                    if tracking.get("native_error"):
                        flags.append("native_tracking_error_partial_or_missing")
                    if frame.get("native_output_missing"):
                        flags.append("native_output_missing")
                    primary = select_primary_alias(masks, primary_ids, previous_primary)
                    alias_switched = (
                        primary is not None
                        and previous_primary is not None
                        and primary != previous_primary
                    )
                    if primary is not None:
                        previous_primary = primary
                    mask = masks.get(primary, np.zeros(rgb.shape[:2], np.uint8))
                    if shot["kind"] == "shot":
                        if primary in geometry_by_id:
                            observations, estimates = geometry_by_id[primary]
                            observation, estimate = observations[i], estimates[i]
                            observed = observation.quad
                            quality = observation.diagnostics
                            photometry = photometry_by_id[primary]
                            if alias_switched:
                                photometry.reset()
                            (
                                baseline,
                                fused,
                                baseline_meta,
                                temporal_meta,
                                focus,
                                render_flags,
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
                                noise_seed=(shot["start_frame"] + i) * 1000003
                                + (primary + 1) * 9176,
                            )
                            flags.extend(render_flags)
                            add_declared_support(baseline_support, support_a)
                            add_declared_support(temporal_support, support_b)
                        else:
                            missing = "target_not_visible_or_not_detected"
                            estimate = SurfaceGeometry(
                                None,
                                mask,
                                0.0,
                                [missing],
                                {"status": "needs_review", "flags": [missing]},
                            )
                            flags.extend([missing, "replacement_skipped_no_geometry"])
                            for state in photometry_by_id.values():
                                state.reset()
                    other_ids = []
                    for oid, mask_other in masks.items():
                        support |= mask_other > 0
                        if oid not in primary_ids:
                            baseline, _ = redact_surface(baseline, mask_other, "blur")
                            fused, _ = redact_surface(fused, mask_other, "blur")
                            other_ids.append(oid)
                            baseline_support |= mask_other > 0
                            temporal_support |= mask_other > 0
                    outside = int(
                        np.count_nonzero(
                            np.any(baseline != rgb, axis=2) & ~baseline_support
                        )
                    ) + int(
                        np.count_nonzero(
                            np.any(fused != rgb, axis=2) & ~temporal_support
                        )
                    )
                    if outside:
                        raise RuntimeError(
                            "Compositor changed pixels outside declared edit support"
                        )
                    status = (
                        "needs_review"
                        if flags
                        else (
                            "rendered_unreviewed"
                            if temporal_meta or other_ids
                            else "no_detected_plate"
                        )
                    )
                else:
                    other_ids = []
                    status = "passthrough_" + shot["kind"]
                    if shot["kind"] == "transition":
                        flags.append("dissolve_not_modeled")
                canvas[48:, :1280] = rgb
                canvas[48:, 1280:2560] = baseline
                canvas[48:, 2560:] = fused
                writer.stdin.write(canvas.tobytes())
                global_idx = shot["start_frame"] + i
                row = {
                    "global_frame_index": global_idx,
                    "shot_id": shot["id"],
                    "shot_frame_index": i,
                    "output_time_seconds": global_idx / 30,
                    "source_sample_time_seconds": 240 + global_idx / 30,
                    "status": status,
                    "flags": sorted(set(flags)),
                    "target_id": primary,
                    "primary_ids": primary_ids,
                    "primary_alias_switched": alias_switched,
                    "other_redacted_ids": other_ids,
                    "independent_replaced": bool(baseline_meta),
                    "apexveil_replaced": bool(temporal_meta),
                    "observed_quad": observed.tolist()
                    if observed is not None
                    else None,
                    "temporal_quad": estimate.quad.tolist()
                    if estimate is not None and estimate.quad is not None
                    else None,
                    "geometry_state": estimate.diagnostics
                    if estimate is not None
                    else None,
                    "quality": quality,
                    "focus": focus,
                    "independent_appearance": baseline_meta,
                    "apexveil_appearance": temporal_meta,
                    "outside_edit_support_changed_pixels": 0,
                    "estimator": estimators.metadata(),
                }
                stream.write(json.dumps(row) + "\n")
                counts[status] += 1
                counts["independent_replaced"] += bool(baseline_meta)
                counts["apexveil_replaced"] += bool(temporal_meta)
                flag_counts.update(set(flags))
                if i in {0, shot["frame_count"] // 2, shot["frame_count"] - 1}:
                    preview = Image.fromarray(canvas)
                    preview.thumbnail((1920, 384))
                    contacts.append(preview.copy())
        finish_writer(writer)
    except BaseException:
        writer.kill()
        writer.wait()
        raise
    finally:
        cap.release()
    verified = verify_encoded_output(video, shot["frame_count"])
    contact = Image.new("RGB", (1920, 384 * len(contacts)), "#10141b")
    for i, im in enumerate(contacts):
        contact.paste(im, (0, 384 * i))
    contact.save(folder / (shot["id"] + "-contact.jpg"), quality=92)
    report = {
        "shot_id": shot["id"],
        "frame_count": shot["frame_count"],
        "signature": signature,
        "counts": dict(counts),
        "flags": dict(flag_counts),
        "seconds": time.perf_counter() - started,
        "encoded_verification": verified,
        "native_error": tracking.get("native_error"),
        "target_id": shot.get("target_id"),
        "exclude_ids": shot.get("exclude_ids", []),
    }
    save_json(report_path, report)
    return report


def render_all(
    root, manifest, template, workers=8, only_ready=False, *, estimators=None
):
    validate_manifest(manifest)
    available = []
    for shot in manifest["shots"]:
        if shot["kind"] not in PASSTHROUGH and shot.get("selection_status") not in {
            "reviewed",
            "none_detected",
        }:
            if only_ready:
                continue
            raise ValueError(f"Candidate selection not reviewed: {shot['id']}")
        available.append(shot)
    if estimators is not None:
        results = [
            render_shot(
                str(root.resolve()),
                shot,
                str(template.resolve()),
                estimators=estimators,
            )
            for shot in available
        ]
        save_json(
            root / "render-progress.json",
            {
                "completed_shots": len(results),
                "total_shots": len(manifest["shots"]),
                "results": results,
                "execution": "injected_provider_sequential",
            },
        )
        if not only_ready:
            (root / "renderer-source-snapshot.py").write_bytes(
                Path(__file__).read_bytes()
            )
        return results
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(
                render_shot, str(root.resolve()), shot, str(template.resolve())
            ): shot["id"]
            for shot in available
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save_json(
                root / "render-progress.json",
                {
                    "completed_shots": len(results),
                    "total_shots": len(manifest["shots"]),
                    "results": results,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "render",
                        "shot": result["shot_id"],
                        "counts": result["counts"],
                    }
                ),
                flush=True,
            )
    if not only_ready:
        renderer_hash = file_sha256(Path(__file__))
        if {r["signature"]["renderer_sha256"] for r in results} != {renderer_hash}:
            raise RuntimeError("Renderer changed during the job; do not mix versions")
        (root / "renderer-source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    return results


def decoded_timing(path, fps, expected_frames):
    data = json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                path,
            ]
        )
    )
    pts = np.array(
        [float(f["best_effort_timestamp_time"]) for f in data["frames"]], np.float64
    )
    if len(pts) != expected_frames:
        raise RuntimeError(
            f"Decoded frame count {len(pts)} differs from {expected_frames}"
        )
    expected = np.arange(expected_frames, dtype=np.float64) / fps
    return {
        "decoded_frames": len(pts),
        "first_pts_seconds": float(pts[0]),
        "last_pts_seconds": float(pts[-1]),
        "max_grid_error_seconds": float(np.max(np.abs(pts - expected))),
        "min_frame_step_seconds": float(np.diff(pts).min()),
        "max_frame_step_seconds": float(np.diff(pts).max()),
    }


def finalize(root, manifest, template_path=None):
    validate_manifest(manifest)
    shots = manifest["shots"]
    if template_path is None:
        raise ValueError(
            "Finalization requires the approved --template for signature verification"
        )
    expected_core = {
        n: file_sha256(Path(__file__).parent / n)
        for n in [
            "imaging/geometry.py",
            "imaging/appearance.py",
            "imaging/compositor.py",
            "imaging/temporal.py",
            "imaging/photometry.py",
        ]
    }
    expected_template = file_sha256(template_path)
    expected_renderer = file_sha256(root / "renderer-source-snapshot.py")
    reports = []
    global_rows = []
    for shot in shots:
        report = json.loads((root / "render" / (shot["id"] + ".json")).read_text())
        signature = report["signature"]
        selection = {
            k: shot.get(k) for k in ["kind", "target_id", "primary_ids", "exclude_ids"]
        }
        tracking_path = root / "tracking" / shot["id"] / "report.json"
        tracking_hash = file_sha256(tracking_path) if tracking_path.exists() else None
        if (
            signature["selection"] != selection
            or signature["core_hashes"] != expected_core
            or signature["template_sha256"] != expected_template
            or signature["tracking_report_sha256"] != tracking_hash
            or signature["renderer_sha256"] != expected_renderer
        ):
            raise RuntimeError(f"Stale render signature for {shot['id']}")
        reports.append(report)
        rows = [
            json.loads(x)
            for x in (root / "render" / (shot["id"] + ".jsonl"))
            .read_text()
            .splitlines()
        ]
        if [x["global_frame_index"] for x in rows] != list(
            range(shot["start_frame"], shot["end_frame"])
        ):
            raise RuntimeError("Frame journal timeline mismatch")
        global_rows.extend(rows)
    if len(global_rows) != 2100:
        raise RuntimeError("Expected all 2100 output frames")
    concat = root / "concat.txt"
    concat.write_text(
        "".join(
            "file '"
            + str((root / "render" / (s["id"] + ".mp4")).resolve()).replace(
                "'", "'\\''"
            )
            + "'\n"
            for s in shots
        )
    )
    silent = root / "comparison-silent.mp4"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            silent,
        ]
    )
    concat_timing = decoded_timing(silent, 30, 2100)
    normalized = False
    if concat_timing["max_grid_error_seconds"] > 1e-5:
        normalized_path = root / "comparison-cfr-silent.mp4"
        command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                silent,
                "-vf",
                "setpts=N/(30*TB)",
                "-r",
                "30",
                "-fps_mode",
                "cfr",
                "-frames:v",
                "2100",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-threads",
                "8",
                "-an",
                "-movflags",
                "+faststart",
                normalized_path,
            ]
        )
        silent = normalized_path
        normalized = True
    source = Path(manifest["source_path"])
    audio = root / "source-audio.mka"
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
            "240",
            "-t",
            "70",
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "copy",
            audio,
        ]
    )
    audio_check = verify_source_audio(source, audio, 240, 70)
    final = root / "ApexVeil-0400-0510-comparison.mp4"
    mux_audio(silent, audio, final)
    encoded = verify_encoded_output(final, 2100, audio)
    if abs(encoded["video_duration_seconds"] - 70) > 0.0001:
        raise RuntimeError("Final video is not exactly70 seconds")
    final_timing = decoded_timing(final, 30, 2100)
    if final_timing["max_grid_error_seconds"] > 1e-5:
        raise RuntimeError("Final decoded PTS are not on the exact 30 fps timeline")
    info = probe(final)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    if (stream["width"], stream["height"]) != (3840, 768):
        raise RuntimeError("Three native-resolution panels were not retained")
    with (root / "frames.jsonl").open("w") as output:
        for row in global_rows:
            output.write(json.dumps(row) + "\n")
    counts = Counter(r["status"] for r in global_rows)
    flags = Counter(f for r in global_rows for f in r["flags"])
    core_files = [
        "imaging/geometry.py",
        "imaging/appearance.py",
        "imaging/compositor.py",
        "imaging/temporal.py",
        "imaging/photometry.py",
        "backends/sam31.py",
    ]
    report = {
        "kind": "complete_70_second_shot_aware_comparison",
        "source_interval_seconds": [240, 310],
        "source_sha256": manifest["source_sha256"],
        "frame_count": 2100,
        "fps": 30,
        "duration_seconds": 70,
        "output_dimensions": [3840, 768],
        "native_panel_dimensions": [1280, 720],
        "title_height": 48,
        "columns": ["RGB", "SAM3.1 + Independent", "ApexVeil"],
        "independent_definition": "Same per-shot SAM3.1 native masks, independent frame geometry/appearance composition",
        "apexveil_definition": "Latest shared core with temporal geometry and fitted photometric state",
        "base_commit": manifest["base_commit"],
        "core_hashes": {
            name: file_sha256(Path(__file__).parent / name) for name in core_files
        },
        "shot_count": len(shots),
        "tracked_shots": sum(s["kind"] not in PASSTHROUGH for s in shots),
        "counts": dict(counts),
        "flags": dict(flags),
        "independent_replaced_frames": sum(
            r["independent_replaced"] for r in global_rows
        ),
        "apexveil_replaced_frames": sum(r["apexveil_replaced"] for r in global_rows),
        "dropped_frames": 0,
        "native_tracking_errors": {
            r["shot_id"]: r["native_error"] for r in reports if r["native_error"]
        },
        "audio_source_verification": audio_check,
        "encoded_verification": encoded,
        "output_sha256": file_sha256(final),
        "concat_timing_before_normalization": concat_timing,
        "final_decoded_timing": final_timing,
        "final_cfr_normalization_applied": normalized,
        "render_signature_verification": "all shots match reviewed selection, approved template, tracking and core hashes",
        "renderer_source_snapshot_sha256": expected_renderer,
        "frame_journal": "frames.jsonl",
        "shot_manifest": "shot_manifest.json",
        "privacy_or_photorealism_certified": False,
        "ai_review": "not_run",
        "public_redistribution": False,
        "timing_policy": "Full fixed 2100-frame CFR sampling; source timestamp = 240 + frame/30; AAC packet-level preservation",
    }
    save_json(root / "report.json", report)
    save_json(
        root / "latest.json",
        {
            "core_commit": manifest["base_commit"],
            "source_sha256": manifest["source_sha256"],
            "output_sha256": report["output_sha256"],
            "video": final.name,
            "report": "report.json",
            "manifest": "shot_manifest.json",
            "columns": report["columns"],
            "complete_timeline": True,
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "track", "render", "finalize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only-ready", action="store_true")
    args = parser.parse_args()
    if args.stage in {"render", "finalize"} and args.template is None:
        parser.error("--template is required for render/finalize")
    manifest = json.loads((args.root / "shot_manifest.json").read_text())
    if args.stage == "prepare":
        prepare_shots(args.root, manifest)
    elif args.stage == "track":
        track_shots(args.root, manifest, load_settings(args.config))
    elif args.stage == "render":
        render_all(args.root, manifest, args.template, args.workers, args.only_ready)
    else:
        print(json.dumps(finalize(args.root, manifest, args.template), indent=2))


if __name__ == "__main__":
    main()
