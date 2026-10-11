"""Uploaded-image workflow using a caller-owned persistent segmenter."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..backends.codex_advisor import (
    PROFILES,
    CodexSceneAdvisor,
    to_source_geometry,
    validate_guidance,
)
from ..diagnostic_export import color_native_masks, stable_color
from ..imaging.compositor import load_template, local_bounds, redact_surface
from ..scene_pipeline import solve_scene
from .evidence import build_scene_evidence, json_value, save_preview, scene_crop

SCOPE = "image-workflow"


def _progress(callback, message, fraction=None):
    if callback is not None:
        callback(message, fraction)


def prepare_image(source_path, output_dir, segmenter, *, progress=None):
    start = time.perf_counter()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path).resolve()
    _progress(progress, "读取图片", 0.05)
    with Image.open(source_path) as source:
        icc = source.info.get("icc_profile")
        rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    decoded = time.perf_counter()
    _progress(progress, "正在识别车牌", 0.2)
    detections, metadata = segmenter.detect(
        Image.fromarray(rgb), prompt="license plate", refine=True
    )
    detected = time.perf_counter()
    masks = {}
    targets = []
    native = out / "native-masks"
    native.mkdir(exist_ok=True)
    for detection in detections:
        oid = int(detection.object_id)
        mask = np.asarray(detection.visible_mask).copy()
        if mask.shape != rgb.shape[:2]:
            raise ValueError("Native mask dimensions do not match oriented source")
        if oid in masks:
            raise ValueError("Duplicate native object ID")
        if not np.any(mask):
            continue
        masks[oid] = mask
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(
            native / f"plate-{oid}.png"
        )
        targets.append(
            {
                "id": oid,
                "area": int(np.count_nonzero(mask)),
                "bbox": local_bounds(mask),
                "score": float(detection.score),
                "color_rgb": list(stable_color(oid, SCOPE)),
            }
        )
    source_preview = save_preview(rgb, out / "source-preview.jpg")
    overlay, _ = color_native_masks(rgb, masks, shot_id=SCOPE)
    draw = ImageDraw.Draw(image := Image.fromarray(overlay))
    for target in targets:
        draw.rectangle(target["bbox"], outline=tuple(target["color_rgb"]), width=3)
        draw.text(tuple(target["bbox"][:2]), str(target["id"]), fill="white")
    image.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
    mask_preview = out / "mask-preview.jpg"
    image.save(mask_preview, quality=95)
    public = {
        "source_preview_path": source_preview,
        "mask_preview_path": str(mask_preview),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "targets": targets,
        "mask_ground_truth": False,
    }
    timings = {
        "decode_seconds": decoded - start,
        "detect_seconds": detected - decoded,
        "model_metadata_load_seconds": float(
            getattr(segmenter, "metadata", {}).get("load_seconds", 0)
        ),
        "prepare_wall_seconds": time.perf_counter() - start,
    }
    prepared = {
        "public": public,
        "private": {
            "rgb": rgb,
            "masks": masks,
            "segmenter": segmenter,
            "detection_metadata": json_value(metadata),
            "model_metadata": json_value(getattr(segmenter, "metadata", {})),
            "source_name": source_path.name,
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "icc_profile": icc,
        },
        "timings": timings,
        "warnings": [] if targets else ["未找到可选车牌，请尝试另一张图片。"],
    }
    timings["prepare_wall_seconds"] = time.perf_counter() - start
    (out / "prepare-report.json").write_text(
        json.dumps(
            {
                "public": public,
                "timings": timings,
                "source_sha256": prepared["private"]["source_sha256"],
                "detection": json_value(metadata),
            },
            indent=2,
        )
    )
    _progress(progress, "请选择需要处理的车牌", 1)
    return prepared


def _foreground(private, target_mask, out, progress):
    _progress(progress, "正在识别人物前景", 0.15)
    start = time.perf_counter()
    detections, metadata = private["segmenter"].detect(
        Image.fromarray(private["rgb"]), prompt="person", refine=True
    )
    directory = out / "native-person-masks"
    directory.mkdir(exist_ok=True)
    h, w = target_mask.shape
    x0, y0, x1, y1 = local_bounds(target_mask)
    pad = max(24, round(max(x1 - x0, y1 - y0) * 0.65))
    box = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
    union = np.zeros((h, w), bool)
    records = []
    for item in detections:
        mask = np.asarray(item.visible_mask)
        if mask.shape != (h, w):
            raise ValueError("Person mask dimensions do not match source")
        path = directory / f"person-{int(item.object_id)}.png"
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
        adjacent = bool(np.any(mask[box[1] : box[3], box[0] : box[2]]))
        if adjacent:
            union |= mask > 0
        records.append(
            {
                "id": int(item.object_id),
                "score": float(item.score),
                "raw_mask_path": str(path),
                "selected_adjacent": adjacent,
                "native_plate_overlap_pixels": int(
                    np.count_nonzero((mask > 0) & (target_mask > 0))
                ),
                "diagnostics": json_value(getattr(item, "diagnostics", {})),
            }
        )
    selected = any(record["selected_adjacent"] for record in records)
    if selected:
        Image.fromarray(union.astype(np.uint8) * 255).save(
            out / "foreground-estimate.png"
        )
    return (
        union if selected else None,
        {
            "enabled": True,
            "ground_truth": False,
            "selection": "person masks adjacent to target bounding box",
            "adjacency_bbox": box,
            "instances": records,
            "detection": json_value(metadata),
        },
        time.perf_counter() - start,
    )


def _diagnostics(source, result, masks, target_id, out):
    overlay, _ = color_native_masks(source, masks, shot_id=SCOPE, target_id=target_id)
    images = [source, overlay, result]
    labels = [
        "Source RGB",
        "识别范围 / 所选目标ID" + str(target_id),
        "Result (not quality certified)",
    ]

    def board(views, path):
        width = 640
        height = round(views[0].shape[0] * width / views[0].shape[1])
        sheet = Image.new("RGB", (width * 3, height + 32), (25, 28, 33))
        draw = ImageDraw.Draw(sheet)
        for i, (view, label) in enumerate(zip(views, labels)):
            sheet.paste(
                Image.fromarray(view).resize((width, height), Image.Resampling.LANCZOS),
                (i * width, 32),
            )
            draw.text((i * width + 8, 8), label, fill="white")
        sheet.save(path, quality=95)

    full = out / "diagnostic-full.jpg"
    board(images, full)
    x0, y0, x1, y1 = scene_crop(masks[target_id])
    detail = out / "diagnostic-detail.jpg"
    board([im[y0:y1, x0:x1] for im in images], detail)
    return str(full), str(detail)


def render_image(
    prepared,
    template_path,
    output_dir,
    *,
    target_id,
    mode="replace",
    advisor_profile=None,
    proxy_url=None,
    protect_foreground=True,
    progress=None,
):
    start = time.perf_counter()
    private = prepared["private"]
    rgb = private["rgb"]
    masks = private["masks"]
    target_id = int(target_id)
    if target_id not in masks:
        raise ValueError("Selected target is not a prepared native detection")
    if mode not in {"replace", "blur", "mosaic"}:
        raise ValueError("Unsupported image mode")
    if advisor_profile is not None and advisor_profile not in PROFILES:
        raise ValueError("Unknown advisor model profile")
    if mode == "replace" and template_path is None:
        raise ValueError("Replacement requires an approved/uploaded template")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    warnings = list(prepared.get("warnings", []))
    timings = dict(prepared.get("timings", {}))
    foreground = None
    foreground_report = {"enabled": False, "ground_truth": False}
    timings["person_detect_seconds"] = 0.0
    timings["ai_seconds"] = 0.0
    timings["evidence_export_seconds"] = 0.0
    calls = 0
    advisor_record = None
    evidence = None
    advice = None
    run_token = uuid.uuid4().hex[:12]
    if mode == "replace" and protect_foreground:
        foreground, foreground_report, timings["person_detect_seconds"] = _foreground(
            private, masks[target_id], out, progress
        )
    _progress(progress, "正在生成图片", 0.3)
    render_started = time.perf_counter()
    timings["appearance_render_seconds"] = 0.0
    if mode == "replace":
        template = load_template(Path(template_path))
        target_result, scene, _ = solve_scene(
            rgb, masks[target_id], template, foreground_mask=foreground
        )
        timings["appearance_render_seconds"] += time.perf_counter() - render_started
        if advisor_profile is not None:
            evidence_started = time.perf_counter()
            evidence = build_scene_evidence(
                rgb,
                masks[target_id],
                template_path,
                out / ("evidence-" + run_token),
                baseline_rgb=target_result,
                scene_report=scene,
                target_id=target_id,
                foreground_mask=foreground,
                context={
                    "domain": "image",
                    "palette_scope": SCOPE,
                    "source_sha256": private["source_sha256"],
                },
            )
            timings["evidence_export_seconds"] = time.perf_counter() - evidence_started
            _progress(progress, "AI正在分析场景", 0.5)
            ai_start = time.perf_counter()
            calls = 1
            advice, advisor_record = CodexSceneAdvisor(
                profile=advisor_profile, proxy_url=proxy_url
            ).run(
                out / ("advisor-" + run_token),
                evidence["prompt"],
                evidence["image_paths"],
            )
            timings["ai_seconds"] = time.perf_counter() - ai_start
            if advice is not None:
                validate_guidance(advice)
                if not advice["plate_confirmed"]:
                    raise ValueError(
                        "Advisor did not confirm the selected license-plate candidate"
                    )
                geometry = to_source_geometry(
                    advice,
                    evidence["crop_xyxy"],
                    source_size=(rgb.shape[1], rgb.shape[0]),
                )
                appearance = {
                    **advice["appearance"],
                    "provenance": "one_shot_codex_scene_guidance",
                }
                _progress(
                    progress,
                    "正在应用AI指导并生成图片",
                    0.8,
                )
                guided_start = time.perf_counter()
                target_result, scene, _ = solve_scene(
                    rgb,
                    masks[target_id],
                    template,
                    foreground_mask=foreground,
                    geometry_guidance=geometry,
                    appearance_guidance=appearance,
                )
                timings["appearance_render_seconds"] += (
                    time.perf_counter() - guided_start
                )
            else:
                warnings.append("AI指导未完成，本次保留自动合成结果。")
        if not scene.get("rendered"):
            raise ValueError(
                "Replacement unresolved: "
                + str(scene.get("unresolved_reason", "no supported scene geometry"))
            )
    else:
        target_result, scene = redact_surface(rgb, masks[target_id], mode)
        scene = {"rendered": True, "status": "rendered_unreviewed", "compositor": scene}
        if advisor_profile is not None:
            warnings.append("模糊和马赛克无需AI指导，本次未调用AI。")
    auxiliary_started = time.perf_counter()
    # Redact all non-targets, then give the selected target priority in overlap.
    result = rgb.copy()
    others = []
    for oid, mask in masks.items():
        if oid != target_id:
            safe = ((mask > 0) & ~(masks[target_id] > 0)).astype(np.uint8)
            if safe.any():
                result, _ = redact_surface(result, safe, "blur")
                others.append(oid)
    target_changed = np.any(target_result != rgb, axis=2)
    result[target_changed] = target_result[target_changed]
    if foreground is not None:
        result[foreground] = rgb[foreground]
    if mode != "replace":
        timings["appearance_render_seconds"] = time.perf_counter() - render_started
    timings["other_redaction_and_merge_seconds"] = (
        time.perf_counter() - auxiliary_started
    )
    output = out / "result.png"
    export_started = time.perf_counter()
    # Lossless RGB and ICC are preserved; level 3 reduces offline export time.
    Image.fromarray(result).save(
        output, compress_level=3, icc_profile=private.get("icc_profile")
    )
    timings["png_export_seconds"] = time.perf_counter() - export_started
    diagnostic_started = time.perf_counter()
    diagnostic, detail = _diagnostics(rgb, result, masks, target_id, out)
    timings["diagnostic_export_seconds"] = time.perf_counter() - diagnostic_started
    if scene.get("status") == "rendered_needs_review":
        warnings.append("光照或边缘仍有估计误差，请检查结果画面。")
    timings["render_wall_seconds"] = time.perf_counter() - start
    timings["processing_wall_seconds"] = (
        timings.get("prepare_wall_seconds", 0) + timings["render_wall_seconds"]
    )
    report = {
        "kind": "image_workflow",
        "success": True,
        "source_name": private["source_name"],
        "source_sha256": private["source_sha256"],
        "target_id": target_id,
        "mode": mode,
        "other_redacted_ids": others,
        "scene": json_value(scene),
        "foreground": foreground_report,
        "advisor": advisor_record,
        "model_calls": calls,
        "advisor_assessment_scope": "shown_baseline_only",
        "final_quality_accepted": None,
        "guidance_applied": advice is not None,
        "evidence_manifest": None if evidence is None else evidence["manifest_path"],
        "timings": timings,
        "warnings": warnings,
        "model_metadata": private.get("model_metadata", {}),
        "model_loaded_by_workflow": False,
        "output_encoding": {"format": "PNG", "lossless": True, "compress_level": 3},
        "outputs": {
            "output_path": str(output),
            "diagnostic_path": diagnostic,
            "detail_path": detail,
        },
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2))
    _progress(progress, "图片生成完成", 1)
    return {
        "success": True,
        "output_path": str(output),
        "diagnostic_path": diagnostic,
        "detail_diagnostic_path": detail,
        "report_path": str(report_path),
        "timings": timings,
        "warnings": warnings,
        "media_type": "image",
        "model_calls": calls,
    }
