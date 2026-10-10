"""Offline full-timeline ApexVeil evaluation from cached native SAM masks.

This script never runs segmentation. A reviewed shot manifest defines identity
selection and timeline boundaries. The center diagnostic panel always shows all
saved native masks, including identities excluded from editing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Environment variables must be set before importing NumPy/OpenCV so backend
# libraries cannot create nested worker pools.
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from apexveil.diagnostic_export import color_native_masks, stable_color
from apexveil.estimation import Estimators, get_default_estimators
from apexveil.imaging.compositor import (
    load_template,
    redact_surface,
    replace_surface,
)
from apexveil.imaging.offline_geometry import SurfaceGeometry
from apexveil.imaging.photometry import TemporalPhotometry
from apexveil.imaging.temporal import rectified_visible_reference, valid_quad
from apexveil.montage_pipeline import validate_manifest
from apexveil.video_pipeline import (
    file_sha256,
    geometry_review_flags,
    mux_audio,
    verify_encoded_output,
    verify_source_audio,
)

HEADER_HEIGHT = 48
MAX_WORKERS = 4
PASSTHROUGH_KINDS = {"no_plate", "black", "transition"}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _decode_shot(path: Path, expected_count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open shot video: {path}")
    frames = []
    try:
        for index in range(expected_count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"{path} ended at frame {index}/{expected_count}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ok, _ = capture.read()
        if ok:
            raise RuntimeError(f"{path} contains more than {expected_count} frames")
    finally:
        capture.release()
    return frames


def _load_tracking(root: Path, shot: dict, frame_count: int, shape: tuple[int, int]):
    report_path = root / "tracking" / shot["id"] / "report.json"
    if not report_path.is_file():
        return (
            [{"frame_index": i, "objects": []} for i in range(frame_count)],
            [{} for _ in range(frame_count)],
            [
                {"frame_index": i, "reason": "tracking_report_missing"}
                for i in range(frame_count)
            ],
            {"path": str(report_path), "status": "missing"},
        )

    report = json.loads(report_path.read_text())
    records = report.get("frames", [])
    records_by_index: dict[int, dict] = {}
    duplicate_indexes = set()
    for record in records:
        index = int(record.get("frame_index", -1))
        if index in records_by_index:
            duplicate_indexes.add(index)
        else:
            records_by_index[index] = record

    normalized_records = []
    frame_masks = []
    errors = []
    for index in range(frame_count):
        record = records_by_index.get(index)
        if index in duplicate_indexes:
            errors.append(
                {"frame_index": index, "reason": "duplicate_tracking_frame_record"}
            )
            record = None
        if record is None:
            errors.append(
                {"frame_index": index, "reason": "tracking_frame_record_missing"}
            )
            record = {
                "frame_index": index,
                "objects": [],
                "native_output_missing": True,
            }
        normalized_records.append(record)
        masks = {}
        if record.get("native_output_missing"):
            errors.append({"frame_index": index, "reason": "native_output_missing"})
        folder = root / "tracking" / shot["id"] / f"frame_{index:06d}"
        for obj in record.get("objects", []):
            object_id = int(obj["object_id"])
            path = folder / obj["mask"]
            if not path.is_file():
                errors.append(
                    {
                        "frame_index": index,
                        "object_id": object_id,
                        "reason": "mask_file_missing",
                        "path": str(path),
                    }
                )
                continue
            try:
                with Image.open(path) as image:
                    mask = np.asarray(image.convert("L")) > 0
            except OSError as error:
                errors.append(
                    {
                        "frame_index": index,
                        "object_id": object_id,
                        "reason": "mask_decode_failed",
                        "error": str(error),
                    }
                )
                continue
            if mask.shape != shape:
                errors.append(
                    {
                        "frame_index": index,
                        "object_id": object_id,
                        "reason": "mask_dimensions_mismatch",
                        "actual": list(mask.shape),
                        "expected": list(shape),
                    }
                )
                continue
            masks[object_id] = mask.astype(np.uint8)
        frame_masks.append(masks)
    status = {
        "path": str(report_path),
        "status": "loaded",
        "reported_frame_count": report.get("frame_count"),
        "expected_frame_count": frame_count,
        "native_error": report.get("native_error"),
        "kind": report.get("kind"),
        "resource_sha256": report.get("resource_sha256"),
        "point_prompts": report.get("point_prompts"),
        "report_sha256": file_sha256(report_path),
    }
    if report.get("frame_count") != frame_count:
        errors.append(
            {
                "reason": "tracking_report_frame_count_mismatch",
                "reported": report.get("frame_count"),
                "expected": frame_count,
            }
        )
    return normalized_records, frame_masks, errors, status


def _load_identity_masks(
    frame_masks: Sequence[dict[int, np.ndarray]], object_id: int, shape
):
    return [
        frame_masks[index].get(object_id, np.zeros(shape, np.uint8))
        for index in range(len(frame_masks))
    ]


def _estimate_identity(
    frames,
    masks,
    aspect: float,
    identity: int,
    estimators: Estimators,
):
    from apexveil.video_pipeline import estimate_geometry_track

    observations, refined = estimate_geometry_track(
        frames, masks, aspect, estimators=estimators, identity=identity
    )
    return identity, observations, refined


def _is_replaceable(estimate: SurfaceGeometry | None) -> tuple[bool, list[str]]:
    if estimate is None or estimate.quad is None:
        return False, ["quadrilateral_missing"]
    if not valid_quad(np.asarray(estimate.quad, dtype=np.float32)):
        return False, ["quadrilateral_invalid"]
    return True, []


def _frame_seed(global_index: int, identity: int) -> int:
    # Match the shared video renderer's stable per-frame/per-ID seed.
    return int((global_index * 1_000_003 + (identity + 1) * 9_176) % (2**32))


def _blend_supported(
    destination: np.ndarray, generated: np.ndarray, detail: dict
) -> None:
    if not detail:
        return
    x0, y0, x1, y1 = detail["bbox_xyxy"]
    local_support = np.asarray(detail["support"], dtype=bool)
    current = destination[y0:y1, x0:x1]
    source = generated[y0:y1, x0:x1]
    if current.shape[:2] != local_support.shape:
        raise ValueError("Compositor support dimensions do not match its bounds")
    current[local_support] = source[local_support]


def _render_replacement(
    rgb,
    mask,
    quad,
    template,
    estimators,
    photometry,
    seed,
):
    height, width = template.alpha.shape
    reference, coverage = rectified_visible_reference(rgb, mask, quad, (width, height))
    if reference is None:
        return (
            None,
            {
                "status": "replacement_rejected",
                "reason": "insufficient_visible_appearance",
                "reference_visible_fraction": coverage,
            },
            {},
        )
    sigma = 0.35
    support = {}
    generated, metadata = replace_surface(
        rgb,
        mask,
        quad,
        template,
        sigma,
        appearance_reference=reference,
        appearance_parameter_filter=photometry.filter,
        noise_seed=seed,
        edit_support_out=support,
        estimators=estimators,
    )
    metadata["reference_visible_fraction"] = coverage
    metadata["appearance_temporal_filter"] = "TemporalPhotometry.filter"
    return generated, metadata, support


def _paint_header(
    panel: np.ndarray, shot: dict, global_index: int, native_ids, mode: str, status: str
):
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    width = panel.shape[1] // 3
    draw.rectangle((0, 0, panel.shape[1], HEADER_HEIGHT), fill="#15191e")
    draw.text(
        (12, 8), f"RGB · {shot['id']} · frame {global_index}", font=font, fill="white"
    )
    draw.text(
        (width + 12, 8),
        "SAM 3.1 native raw masks · all returned IDs",
        font=font,
        fill="white",
    )
    x = width + 12
    for object_id in native_ids:
        color = stable_color(object_id, shot["id"])
        draw.rectangle((x, 34, x + 12, 46), fill=color)
        draw.text((x + 17, 31), f"ID {object_id}", font=font, fill="#d7dbe0")
        x += 68
        if x > width * 2 - 120:
            break
    draw.text(
        (2 * width + 12, 8),
        f"ApexVeil · {mode.title()}",
        font=font,
        fill="white",
    )
    draw.text(
        (2 * width + 12, 34),
        status[:90],
        font=font,
        fill="#ffcf8b" if status else "#bfc7ce",
    )
    return np.asarray(image)


def _diagnostic_frame(rgb, frame_masks, shot, global_index, right, mode, status):
    raw_panel, _ = color_native_masks(
        rgb,
        frame_masks,
        shot_id=shot["id"],
        target_id=None,
        primary_ids=shot.get("primary_ids") or [],
        exclude_ids=shot.get("exclude_ids") or [],
    )
    height, width = rgb.shape[:2]
    panel = np.empty((height + HEADER_HEIGHT, width * 3, 3), dtype=np.uint8)
    panel[HEADER_HEIGHT:, :width] = rgb
    panel[HEADER_HEIGHT:, width : 2 * width] = raw_panel
    panel[HEADER_HEIGHT:, 2 * width :] = right
    return _paint_header(panel, shot, global_index, sorted(frame_masks), mode, status)


def _process_shot(
    root: Path,
    shot: dict,
    template,
    estimators: Estimators,
    workers: int,
    fps: int,
    source_start: float,
    frame_callback,
):
    source_path = root / "shots" / f"{shot['id']}.mkv"
    frames = _decode_shot(source_path, int(shot["frame_count"]))
    height, width = frames[0].shape[:2]
    if any(frame.shape != (height, width, 3) for frame in frames):
        raise ValueError(f"Shot {shot['id']} changes dimensions within its interval")
    records, frame_masks, mask_errors, tracking_status = _load_tracking(
        root, shot, len(frames), (height, width)
    )
    native_ids = sorted({oid for masks in frame_masks for oid in masks})
    # Include IDs explicitly returned in report rows even if every mask file is missing.
    native_ids = sorted(
        set(native_ids)
        | {
            int(obj["object_id"])
            for record in records
            for obj in record.get("objects", [])
        }
    )
    selected_ids = sorted(set(map(int, shot.get("primary_ids") or [])))
    excluded = set(map(int, shot.get("exclude_ids") or []))
    primary_masks = {
        identity: _load_identity_masks(frame_masks, identity, (height, width))
        for identity in selected_ids
    }
    aspect = template.rgb.shape[1] / template.rgb.shape[0]
    estimates = {}
    estimator_errors = {}
    if shot["kind"] == "shot":
        with ThreadPoolExecutor(max_workers=min(workers, 2)) as pool:
            futures = {
                pool.submit(
                    _estimate_identity,
                    frames,
                    masks,
                    aspect,
                    identity,
                    estimators,
                ): identity
                for identity, masks in primary_masks.items()
            }
            for future, identity in ((future, futures[future]) for future in futures):
                try:
                    _, observations, refined = future.result()
                    estimates[identity] = (observations, refined)
                except Exception as error:  # noqa: BLE001 - report estimator failures instead of hiding them
                    estimator_errors[identity] = f"{type(error).__name__}: {error}"
                    estimates[identity] = ([None] * len(frames), [None] * len(frames))

    photometry = {identity: TemporalPhotometry(float(fps)) for identity in selected_ids}
    rows = []
    for local_index, (rgb, record, masks) in enumerate(
        zip(frames, records, frame_masks)
    ):
        global_index = int(shot["start_frame"]) + local_index
        replace = rgb.copy()
        blur = rgb.copy()
        declared_support = np.zeros((height, width), dtype=bool)
        flags = []
        actions = []
        missing_ids = sorted(
            {
                int(obj["object_id"])
                for obj in record.get("objects", [])
                if int(obj["object_id"]) not in masks
            }
        )
        if missing_ids:
            flags.append("native_masks_missing")
        if record.get("native_output_missing"):
            flags.append("native_output_missing")
        if shot["kind"] in PASSTHROUGH_KINDS:
            frame_status = f"passthrough_{shot['kind']}"
        else:
            frame_status = "rendered_unreviewed"
            if shot["kind"] == "shot" and not selected_ids:
                flags.append("primary_ids_empty")
            # Blur every non-excluded native mask in both output variants.
            # Background shots have no primary selection, so traffic plates only blur.
            for object_id in sorted(masks):
                mask = masks[object_id]
                if object_id in excluded or not mask.any():
                    continue
                if shot["kind"] == "shot" and object_id in selected_ids:
                    continue
                replace, blur_meta = redact_surface(replace, mask, "blur")
                actions.append(
                    {
                        "object_id": object_id,
                        "action": "blur_other",
                        "details": blur_meta,
                    }
                )
            for object_id in sorted(masks):
                mask = masks[object_id]
                if object_id in excluded or not mask.any():
                    continue
                blur, blur_meta = redact_surface(blur, mask, "blur")
                actions.append(
                    {
                        "object_id": object_id,
                        "action": "blur_variant",
                        "details": blur_meta,
                    }
                )
            if shot["kind"] == "shot":
                for identity in selected_ids:
                    mask = primary_masks[identity][local_index]
                    observation, estimate = (None, None)
                    if identity in estimates:
                        observation, estimate = (
                            estimates[identity][0][local_index],
                            estimates[identity][1][local_index],
                        )
                    accepted, rejected = _is_replaceable(estimate)
                    item = {
                        "object_id": identity,
                        "visible_pixels": int(mask.sum()),
                        "quad_accepted": bool(accepted),
                        "quad_rejection_reasons": rejected,
                        "quad": estimate.quad.tolist()
                        if estimate is not None and estimate.quad is not None
                        else None,
                        "confidence": estimate.confidence
                        if estimate is not None
                        else 0.0,
                        "geometry_flags": list(estimate.flags)
                        if estimate is not None
                        else [],
                        "geometry_review_flags": geometry_review_flags(estimate)
                        if estimate is not None
                        else [],
                        "geometry_diagnostics": estimate.diagnostics
                        if estimate is not None
                        else {},
                        "observation_quad": observation.quad.tolist()
                        if observation is not None and observation.quad is not None
                        else None,
                        "provider_error": estimator_errors.get(identity),
                        "compositor": None,
                        "action": "none",
                    }
                    if not mask.any():
                        item["action"] = "mask_missing_or_empty"
                        flags.append("primary_mask_missing_or_empty")
                        photometry[identity].reset()
                    elif accepted:
                        if estimate.diagnostics.get("cut_before"):
                            photometry[identity].reset()
                        try:
                            generated, comp_meta, support = _render_replacement(
                                rgb,
                                mask,
                                estimate.quad,
                                template,
                                estimators,
                                photometry[identity],
                                _frame_seed(global_index, identity),
                            )
                            if generated is None:
                                raise ValueError(comp_meta["reason"])
                            _blend_supported(replace, generated, support)
                            if support:
                                x0, y0, x1, y1 = support["bbox_xyxy"]
                                declared_support[y0:y1, x0:x1] |= support["support"]
                                item["declared_support"] = {
                                    "bbox_xyxy": support["bbox_xyxy"],
                                    "pixels": int(np.count_nonzero(support["support"])),
                                }
                            item["action"] = "replace"
                            item["compositor"] = comp_meta
                            if geometry_review_flags(estimate) or comp_meta.get(
                                "boundary", {}
                            ).get("needs_review"):
                                flags.append("geometry_or_boundary_needs_review")
                        except Exception as error:  # noqa: BLE001 - record compositor rejection
                            photometry[identity].reset()
                            item["action"] = "replacement_rejected_compositor_error"
                            item["compositor_error"] = (
                                f"{type(error).__name__}: {error}"
                            )
                            flags.append("replacement_rejected")
                    else:
                        photometry[identity].reset()
                        if estimate is None:
                            flags.append("geometry_estimation_failed")
                        else:
                            flags.append("geometry_quad_rejected")
                        item["action"] = "replacement_rejected_geometry"
                    actions.append(item)
            if missing_ids:
                frame_status = "needs_review_missing_masks"
            elif any(
                action.get("action", "").startswith("replacement_rejected")
                for action in actions
            ):
                frame_status = "needs_review_replacement_rejected"
            elif flags:
                frame_status = "needs_review"
        # Explicit safety checks constrain all edits to raw visible masks.
        allowed = np.zeros((height, width), bool)
        if shot["kind"] not in PASSTHROUGH_KINDS:
            for object_id, mask in masks.items():
                if object_id not in excluded:
                    allowed |= mask > 0
            allowed |= declared_support
        for label, output in (("replace", replace), ("blur", blur)):
            outside = int(np.count_nonzero(np.any(output != rgb, axis=2) & ~allowed))
            if outside:
                raise RuntimeError(
                    f"{shot['id']} frame {local_index} {label} changed {outside} pixels outside native masks"
                )
        raw_ids = sorted({int(obj["object_id"]) for obj in record.get("objects", [])})
        rows.append(
            {
                "global_frame_index": global_index,
                "output_time_seconds": global_index / float(fps),
                "source_sample_time_seconds": source_start + global_index / float(fps),
                "shot_id": shot["id"],
                "shot_frame_index": local_index,
                "kind": shot["kind"],
                "status": frame_status,
                "flags": sorted(set(flags)),
                "primary_ids": selected_ids,
                "exclude_ids": sorted(excluded),
                "native_returned_ids": raw_ids,
                "missing_mask_ids": missing_ids,
                "native_output_missing": bool(record.get("native_output_missing")),
                "actions": actions,
            }
        )
        frame_callback(
            rows[-1],
            _diagnostic_frame(
                rgb, masks, shot, global_index, replace, "Replace", frame_status
            ),
            _diagnostic_frame(
                rgb, masks, shot, global_index, blur, "Blur", frame_status
            ),
        )
    return {
        "shot_id": shot["id"],
        "kind": shot["kind"],
        "frame_count": len(frames),
        "source_video": str(source_path),
        "source_video_sha256": file_sha256(source_path),
        "tracking": tracking_status,
        "mask_errors": mask_errors,
        "native_ids": native_ids,
        "primary_ids": selected_ids,
        "exclude_ids": sorted(excluded),
        "estimator_errors": estimator_errors,
        "frames": rows,
    }


def _start_encoders(output_dir: Path, variant: str, width: int, height: int, fps: int):
    path = output_dir / f".{variant}-partial.mp4"
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
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
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
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    return path, process


def _stop_encoder(process, terminate=False):
    if process is None:
        return
    if terminate:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stderr:
            process.stderr.close()
        return
    if process.stdin and not process.stdin.closed:
        process.stdin.close()
    error = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.wait():
        raise RuntimeError(f"Diagnostic video encoding failed: {error[-3000:]}")


def _finalize_video(silent_path, output_path, audio_source, expected_frames, full_run):
    pending = output_path.with_name(f".{output_path.name}.pending.mp4")
    pending.unlink(missing_ok=True)
    try:
        if full_run:
            mux_audio(silent_path, audio_source, pending)
            verification = verify_encoded_output(pending, expected_frames, audio_source)
        else:
            verification = verify_encoded_output(silent_path, expected_frames)
            silent_path.replace(pending)
        pending.replace(output_path)
        if full_run:
            silent_path.unlink(missing_ok=True)
        return verification
    except Exception:
        pending.unlink(missing_ok=True)
        raise


def render_vnext(
    root,
    output,
    *,
    template_path=None,
    shot_ids=None,
    workers=4,
    estimators=None,
):
    """Render selected shots or the full reviewed timeline from cached masks."""
    root, output = Path(root), Path(output)
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    manifest_path = root / "shot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    shots = manifest["shots"]
    all_ids = {shot["id"] for shot in shots}
    selected = set(shot_ids) if shot_ids else all_ids
    unknown = selected - all_ids
    if unknown:
        raise ValueError(f"Unknown shot IDs: {sorted(unknown)}")
    ordered_shots = [shot for shot in shots if shot["id"] in selected]
    full_run = shot_ids is None
    if template_path is None:
        template_path = (
            root.parent.parent
            / "data"
            / "private"
            / "templates"
            / "P7plus-flat-approved.png"
        )
    template_path = Path(template_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"Approved template not found: {template_path}")
    template = load_template(template_path)
    estimators = estimators or get_default_estimators()
    output.mkdir(parents=True, exist_ok=True)
    panel_width, panel_height = manifest.get("native_panel_dimensions", [1280, 720])
    width, height = int(panel_width) * 3, int(panel_height) + HEADER_HEIGHT
    fps = int(manifest["fps"])
    stem = "ApexVeil-vnext" if full_run else "ApexVeil-vnext-subset"
    final_paths = {
        "replace": output / f"{stem}-replace-diagnostics.mp4",
        "blur": output / f"{stem}-blur-diagnostics.mp4",
    }
    audio_source = root / "source-audio.mka"
    source = Path(manifest["source_path"]) if full_run else None
    if full_run and (
        source is None or not source.is_file() or not audio_source.is_file()
    ):
        raise FileNotFoundError(
            "Full run requires original source and extracted source-audio.mka"
        )
    report = {
        "schema_version": 1,
        "status": "rendering",
        "complete_timeline": False,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "source_interval_seconds": manifest["source_interval_seconds"],
        "fps": fps,
        "expected_frame_count": int(manifest["frame_count"])
        if full_run
        else sum(s["frame_count"] for s in ordered_shots),
        "full_timeline": full_run,
        "shot_ids": [s["id"] for s in ordered_shots],
        "layout": ["RGB", "SAM3.1 colored raw masks", "ApexVeil Replace or Blur"],
        "output_dimensions": [width, height],
        "title_height": HEADER_HEIGHT,
        "template": str(template_path),
        "template_sha256": template.metadata["source_sha256"],
        "estimators": estimators.metadata(),
        "cpu_workers": int(workers),
        "geometry_workers": min(int(workers), 2),
        "encoder_threads_per_variant": 1,
        "gpu_used": False,
        "sam_inference_run": False,
        "sam_raw_masks_are_ground_truth": False,
        "privacy_certified": False,
        "product_quality_acceptance": "not_run",
        "full_plane_completeness": "unverified_semantic_hypothesis",
        "noise_seed_policy": "global_frame_index*1000003 + (object_id+1)*9176 modulo 2^32",
        "source_audio": str(root / "source-audio.mka") if full_run else None,
        "source_audio_verification": None,
        "encoded_verification": {},
        "shots": [],
        "frames": [],
    }
    if full_run:
        report["source_audio_verification"] = verify_source_audio(
            source,
            audio_source,
            float(manifest["source_interval_seconds"][0]),
            float(manifest["duration_seconds"]),
        )
    silent_paths, encoders = {}, {}
    frame_cursor = 0
    try:
        for variant in ("replace", "blur"):
            silent_paths[variant], encoders[variant] = _start_encoders(
                output, variant, width, height, fps
            )
        cv2.setNumThreads(1)
        for shot in ordered_shots:
            shot_rows = []

            def consume_frame(row, replace_panel, blur_panel, shot_rows=shot_rows):
                nonlocal frame_cursor
                encoders["replace"].stdin.write(replace_panel.tobytes())
                encoders["blur"].stdin.write(blur_panel.tobytes())
                report["frames"].append(row)
                shot_rows.append(row)
                frame_cursor += 1

            processed = _process_shot(
                root,
                shot,
                template,
                estimators,
                int(workers),
                fps,
                float(manifest["source_interval_seconds"][0]),
                consume_frame,
            )
            processed["frames"] = shot_rows
            report["shots"].append(
                {k: v for k, v in processed.items() if k != "frames"}
            )
            _write_json(output / f"{shot['id']}-vnext-report.json", processed)
            del processed
        for process in encoders.values():
            _stop_encoder(process)
        count = report["expected_frame_count"]
        if frame_cursor != count:
            raise RuntimeError(f"Rendered {frame_cursor} frames; expected {count}")
        report["encoded_verification"] = {}
        for variant in ("replace", "blur"):
            report["encoded_verification"][variant] = _finalize_video(
                silent_paths[variant],
                final_paths[variant],
                audio_source if full_run else None,
                count,
                full_run,
            )
        report["output_paths"] = {key: str(path) for key, path in final_paths.items()}
        report["status"] = (
            "completed_unreviewed"
            if full_run
            and not any(
                shot["mask_errors"]
                or shot["estimator_errors"]
                or shot["tracking"].get("native_error")
                for shot in report["shots"]
            )
            and not any(frame["flags"] for frame in report["frames"])
            else "needs_review"
            if full_run
            else "subset_preview"
        )
        report["complete_timeline"] = full_run
    except Exception as error:
        for process in encoders.values():
            _stop_encoder(process, terminate=True)
        for path in silent_paths.values():
            path.unlink(missing_ok=True)
        for path in output.glob(".*.pending.mp4"):
            path.unlink(missing_ok=True)
        report["status"] = "failed"
        report["failure"] = f"{type(error).__name__}: {error}"
        _write_json(output / "vnext-report.json", report)
        raise
    _write_json(output / "vnext-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="deliverable-70s cache root"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument(
        "--shots", help="comma-separated manifest IDs for a silent subset preview"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = (
        None
        if not args.shots
        else [item.strip() for item in args.shots.split(",") if item.strip()]
    )
    result = render_vnext(
        args.root,
        args.output,
        template_path=args.template,
        shot_ids=selected,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "complete_timeline": result["complete_timeline"],
                "frame_count": len(result["frames"]),
                "output_paths": result.get("output_paths", {}),
                "report": str(args.output / "vnext-report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
