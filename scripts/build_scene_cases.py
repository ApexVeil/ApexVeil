"""Build private, source-bound evidence packs for selected image/video cases."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from apexveil.diagnostic_export import color_native_masks
from apexveil.video_pipeline import file_sha256

IMAGE_CASES = ("IMG_0205", "IMG_0287", "IMG_9998")
VIDEO_GLOBAL_FRAMES = (520, 1205, 1931)
DEFAULT_OUTPUT = Path("outputs/scene-solver-20261010/cases")
TEMPLATE_RELATIVE = Path("data/private/templates/P7plus-flat-approved.png")
VIDEO_CACHE = Path("outputs/deliverable-70s-20261010")
IMAGE_AUTOMATIC = Path("outputs/deliverable-images-20261010/automatic")
IMAGE_PACKAGE = Path("outputs/deliverable-20261010/images")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _mask_bbox(masks: list[np.ndarray], image_shape: tuple[int, int], pad: int):
    union = np.zeros(image_shape, dtype=bool)
    for mask in masks:
        if mask.shape != image_shape:
            raise ValueError("Mask and source frame dimensions do not match")
        union |= mask > 0
    ys, xs = np.where(union)
    height, width = image_shape
    if not len(xs):
        return (0, 0, width, height)
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + pad + 1),
        min(height, int(ys.max()) + pad + 1),
    )


def _crop_for_masks(masks, shape, *, padding_ratio: float, minimum_padding: int):
    ys, xs = np.where(np.logical_or.reduce(masks) if masks else np.zeros(shape, bool))
    height, width = shape
    if not len(xs):
        return (0, 0, width, height)
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    pad = max(minimum_padding, math.ceil(span * padding_ratio))
    return _mask_bbox(masks, shape, pad)


def _expand_crop_minimum(crop_xyxy, image_shape, minimum_size):
    x0, y0, x1, y1 = crop_xyxy
    height, width = image_shape
    min_width, min_height = minimum_size
    wanted_width = min(width, max(x1 - x0, min_width))
    wanted_height = min(height, max(y1 - y0, min_height))
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    new_x0 = max(0, min(width - wanted_width, center_x - wanted_width // 2))
    new_y0 = max(0, min(height - wanted_height, center_y - wanted_height // 2))
    return new_x0, new_y0, new_x0 + wanted_width, new_y0 + wanted_height


def _tight_crop_for_target(mask, image_shape, padding_factor=1.0):
    ys, xs = np.where(mask > 0)
    height, width = image_shape
    if not len(xs):
        return 0, 0, width, height
    target_height = int(ys.max() - ys.min() + 1)
    pad = max(1, math.ceil(target_height * padding_factor))
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + pad + 1),
        min(height, int(ys.max()) + pad + 1),
    )


def _save_crop(rgb, crop_xyxy, path):
    x0, y0, x1, y1 = crop_xyxy
    Image.fromarray(rgb[y0:y1, x0:x1]).save(path, compress_level=2)


def _crop_mask_overlay(rgb, masks, crop_xyxy, scope):
    x0, y0, x1, y1 = crop_xyxy
    cropped = {object_id: mask[y0:y1, x0:x1] for object_id, mask in masks.items()}
    return _overlay(rgb[y0:y1, x0:x1], cropped, scope)


def _save_ruler_crop(rgb: np.ndarray, crop_xyxy, destination: Path, tick=200):
    x0, y0, x1, y1 = crop_xyxy
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    margin = 56
    canvas = Image.new(
        "RGB", (crop.width + margin * 2, crop.height + margin * 2), "#e8e8e8"
    )
    canvas.paste(crop, (margin, margin))
    grid = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid)
    grid_step = 50
    for local_x in range(0, crop.width + 1, grid_step):
        px = margin + local_x
        grid_draw.line(
            (px, margin, px, margin + crop.height),
            fill=(255, 255, 255, 72),
            width=1,
        )
    for local_y in range(0, crop.height + 1, grid_step):
        py = margin + local_y
        grid_draw.line(
            (margin, py, margin + crop.width, py),
            fill=(255, 255, 255, 72),
            width=1,
        )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), grid).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    draw.text((margin, 5), "crop-local x (px) · grid 50 px", fill="#111111", font=font)
    draw.text((4, margin - 17), "y", fill="#111111", font=font)
    for local_x in range(0, crop.width + 1, tick):
        px = margin + local_x
        draw.line((px, margin - 6, px, margin), fill="#111111", width=1)
        draw.text((px + 2, margin - 20), str(local_x), fill="#111111", font=font)
    for local_y in range(0, crop.height + 1, tick):
        py = margin + local_y
        draw.line((margin - 6, py, margin, py), fill="#111111", width=1)
        draw.text((3, py + 2), str(local_y), fill="#111111", font=font)
    canvas.save(destination)
    return {
        "crop_xyxy_original": [x0, y0, x1, y1],
        "crop_size": [crop.width, crop.height],
        "ruler_border_px": margin,
        "ruler_content_origin_xy": [margin, margin],
        "grid_spacing_px": grid_step,
        "grid_overlay_applied_to_ruler_image_only": True,
        "ruler_pixel_to_crop_local": "u=x-ruler_content_origin_x, v=y-ruler_content_origin_y",
        "crop_local_to_original": "original_x=crop_x0+u, original_y=crop_y0+v",
    }


def _overlay(rgb: np.ndarray, masks: dict[int, np.ndarray], scope: str):
    result, _ = color_native_masks(rgb, masks, shot_id=scope)
    return result


def _save_native_masks(case_dir: Path, masks: dict[int, np.ndarray]):
    folder = case_dir / "native_masks"
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for object_id, mask in sorted(masks.items()):
        path = folder / f"ID{object_id}.png"
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
        records[str(object_id)] = {
            "path": path.relative_to(case_dir).as_posix(),
            "sha256": file_sha256(path),
            "visible_pixels": int(np.count_nonzero(mask)),
        }
    return records


def _load_image_case(repo: Path, case_id: str, common_template: Path):
    automatic_root = repo / IMAGE_AUTOMATIC
    automatic_report_path = automatic_root / case_id / "report.json"
    automatic = json.loads(automatic_report_path.read_text())
    source_path = repo / "data/private/evaluation/test" / automatic["source_name"]
    package_root = repo / IMAGE_PACKAGE
    package_manifest_path = package_root / "manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text())
    package_record = next(
        row
        for row in package_manifest["records"]
        if row["file"] == automatic["source_name"]
    )
    selected_report_name = (
        f"{case_id}-assisted.json"
        if (package_root / "reports" / f"{case_id}-assisted.json").is_file()
        else f"{case_id}-automatic.json"
    )
    selected_report_path = package_root / "reports" / selected_report_name
    selected_report = json.loads(selected_report_path.read_text())
    latest_root = repo / "outputs/deterministic-vnext-final-20261010/images" / case_id
    old_replace = latest_root / "replace.png"
    old_blur = latest_root / "blur.png"
    latest_report_path = latest_root / "report.json"
    if not all(path.is_file() for path in (old_replace, old_blur, latest_report_path)):
        raise FileNotFoundError(
            f"Latest deterministic image result missing for {case_id}"
        )
    latest_report = json.loads(latest_report_path.read_text())

    rgb = _load_rgb(source_path)
    if [rgb.shape[1], rgb.shape[0]] != automatic["size"]:
        raise ValueError(f"Oriented source size differs from report for {case_id}")
    masks = {}
    source_masks = {}
    for surface in automatic["surfaces"]:
        object_id = int(surface["object_id"])
        path = automatic_root / case_id / surface["visible_mask"]
        mask = _load_rgb_mask(path)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Native mask size differs for {case_id} ID {object_id}")
        masks[object_id] = mask
        source_masks[object_id] = path

    target_id = package_record.get("target_id")
    if target_id is None:
        target_id = selected_report["results"]["replace"].get("target_id")
    target_mask = masks.get(int(target_id)) if target_id is not None else None
    crop_masks = [target_mask] if target_mask is not None else list(masks.values())
    crop = _crop_for_masks(
        crop_masks,
        rgb.shape[:2],
        padding_ratio=2.0 if case_id == "IMG_0205" else 1.25,
        minimum_padding=650 if case_id == "IMG_0205" else 256,
    )

    crop = _expand_crop_minimum(crop, rgb.shape[:2], (1400, 1100))
    tight_crop = _tight_crop_for_target(target_mask, rgb.shape[:2])
    case_dir = repo / DEFAULT_OUTPUT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    original_copy = case_dir / f"source-original{source_path.suffix.lower()}"
    shutil.copy2(source_path, original_copy)
    Image.fromarray(rgb).save(case_dir / "source-full-oriented.png", compress_level=2)
    _save_crop(rgb, crop, case_dir / "source-crop-native.png")
    ruler_meta = _save_ruler_crop(rgb, crop, case_dir / "source-crop-ruler.png")
    _save_crop(rgb, tight_crop, case_dir / "source-crop-tight-native.png")
    tight_ruler_meta = _save_ruler_crop(
        rgb, tight_crop, case_dir / "source-crop-tight-ruler.png"
    )
    mask_records = _save_native_masks(case_dir, masks)
    Image.fromarray(_overlay(rgb, masks, case_id)).save(
        case_dir / "raw-mask-overlay-full.jpg", quality=93
    )
    Image.fromarray(_crop_mask_overlay(rgb, masks, crop, case_id)).save(
        case_dir / "raw-mask-overlay-crop.jpg", quality=95
    )
    Image.fromarray(_crop_mask_overlay(rgb, masks, tight_crop, case_id)).save(
        case_dir / "raw-mask-overlay-tight.jpg", quality=95
    )

    replace_copy = case_dir / "latest-result-replace.png"
    blur_copy = case_dir / "latest-result-blur.png"
    shutil.copy2(old_replace, replace_copy)
    shutil.copy2(old_blur, blur_copy)
    for variant, result_path in (("replace", old_replace), ("blur", old_blur)):
        with Image.open(result_path) as image:
            oriented_result = ImageOps.exif_transpose(image)
            oriented_result.crop(crop).save(
                case_dir / f"latest-result-{variant}-crop.png", compress_level=2
            )
            oriented_result.crop(tight_crop).save(
                case_dir / f"latest-result-{variant}-tight-crop.png", compress_level=2
            )
    shutil.copy2(latest_report_path, case_dir / "latest-report.json")

    metadata = {
        "case_id": case_id,
        "domain": "image",
        "source_name": source_path.name,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_original_copy": original_copy.name,
        "oriented_size_px": [rgb.shape[1], rgb.shape[0]],
        "original_resolution": [rgb.shape[1], rgb.shape[0]],
        "crop_context": ruler_meta,
        "crop_tight": tight_ruler_meta,
        "target_id": target_id,
        "primary_ids": [int(target_id)] if target_id is not None else [],
        "exclude_ids": list(map(int, selected_report.get("exclude_ids", []))),
        "native_ids": mask_records,
        "automatic_detection_report": str(automatic_report_path),
        "automatic_detection_report_sha256": file_sha256(automatic_report_path),
        "previous_package_report": str(selected_report_path),
        "previous_package_report_sha256": file_sha256(selected_report_path),
        "latest_report": str(latest_report_path),
        "latest_report_sha256": file_sha256(latest_report_path),
        "review_status": latest_report["results"]["replace"]["status"],
        "review_flags": latest_report["results"]["replace"]["flags"],
        "provenance": {
            "latest_estimators": latest_report.get("estimators", {}),
            "previous_package_kind": package_record["kind"],
            "previous_package_correction": package_record.get("correction", {}),
            "previous_package_had_manual_corners": package_record.get(
                "correction", {}
            ).get("kind")
            == "explicit_external_override",
            "latest_geometry_uncertain": latest_report["results"]["replace"]["status"]
            == "needs_review"
            or bool(latest_report["results"]["replace"]["flags"]),
            "latest_report_reflects_frozen_automatic_run": True,
        },
        "latest_result": {
            "replace": str(old_replace),
            "replace_sha256": file_sha256(old_replace),
            "blur": str(old_blur),
            "blur_sha256": file_sha256(old_blur),
            "report": str(latest_report_path),
            "report_sha256": file_sha256(latest_report_path),
            "renderer_git_commit": latest_report.get("renderer_git_commit"),
            "source_changed_during_evaluation": latest_report.get(
                "source_changed_during_evaluation"
            ),
        },
        "approved_template": str(common_template),
        "approved_template_sha256": file_sha256(common_template),
        "coordinate_mapping": {
            "ruler_pixel_to_crop_local": ruler_meta["ruler_pixel_to_crop_local"],
            "crop_local_to_original": ruler_meta["crop_local_to_original"],
            "tight_crop_local_to_original": tight_ruler_meta["crop_local_to_original"],
        },
        "all_returned_masks_are_native_raw": True,
    }
    _write_json(case_dir / "manifest.json", metadata)
    return metadata


def _load_rgb_mask(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _load_video_frame(video_path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {frame_index} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _tracking_masks(cache_root: Path, shot_id: str, local_index: int, shape):
    report_path = cache_root / "tracking" / shot_id / "report.json"
    report = json.loads(report_path.read_text())
    row = next(
        (
            entry
            for entry in report["frames"]
            if int(entry["frame_index"]) == local_index
        ),
        None,
    )
    if row is None:
        raise ValueError(f"Tracking row absent for {shot_id} frame {local_index}")
    masks = {}
    source_paths = {}
    folder = cache_root / "tracking" / shot_id / f"frame_{local_index:06d}"
    for obj in row.get("objects", []):
        object_id = int(obj["object_id"])
        path = folder / obj["mask"]
        if not path.is_file():
            continue
        mask = _load_rgb_mask(path)
        if mask.shape != shape:
            raise ValueError(
                f"Mask dimensions mismatch for {shot_id} frame {local_index}"
            )
        masks[object_id] = mask
        source_paths[object_id] = path
    return masks, source_paths, report_path, report


def _anchor_for_shot(cache_root: Path, shot: dict, target_id: int):
    report_path = cache_root / "render" / f"{shot['id']}.jsonl"
    rows = [
        json.loads(line)
        for line in report_path.read_text().splitlines()
        if line.strip()
    ]
    tracking = json.loads(
        (cache_root / "tracking" / shot["id"] / "report.json").read_text()
    )
    by_frame = {int(row["frame_index"]): row for row in tracking["frames"]}
    candidates = []
    disqualifying = {
        "partially_out_of_frame",
        "surface_too_small",
        "occluded_or_non_planar_boundary",
        "possible_occlusion_or_extreme_foreshortening",
        "geometry_failed",
        "insufficient_geometry",
        "replacement_skipped_no_geometry",
        "target_not_visible_or_not_detected",
    }
    for row in rows:
        local = int(row["shot_frame_index"])
        tr = by_frame.get(local, {})
        objects = {int(obj["object_id"]): obj for obj in tr.get("objects", [])}
        obj = objects.get(target_id)
        if obj is None or int(obj.get("visible_pixels", 0)) <= 0:
            continue
        quality = row.get("quality", {})
        flags = set(row.get("flags", [])) | set(quality.get("flags", []))
        focus = row.get("focus", {})
        clean = not (flags & disqualifying) and row.get("observed_quad") is not None
        rank = (
            int(clean),
            int(bool(focus.get("reliable"))),
            -float(focus.get("sigma_px", 99.0)),
            int(obj.get("visible_pixels", 0)),
        )
        candidates.append((rank, local, row, int(obj["visible_pixels"]), sorted(flags)))
    if not candidates:
        raise ValueError(
            f"No non-empty target-ID frame available for anchor in {shot['id']}"
        )
    clean_candidates = [item for item in candidates if item[0][0] == 1]
    chosen = max(clean_candidates or candidates, key=lambda item: item[0])
    return {
        "frame_index": chosen[1],
        "global_frame_index": int(shot["start_frame"]) + chosen[1],
        "target_id": target_id,
        "visible_pixels": chosen[3],
        "selection_policy": (
            "prefer nonempty frames without clipping/small/occlusion geometry flags; then reliable focus, "
            "lower measured sigma, and larger visible mask; if none qualify, use best visible fallback"
        ),
        "used_clean_candidate": bool(clean_candidates),
        "render_report_status": chosen[2].get("status"),
        "render_report_flags": chosen[4],
        "quality": chosen[2].get("quality", {}),
        "focus": chosen[2].get("focus", {}),
        "render_jsonl": str(report_path),
    }


def _make_video_case(repo, global_index: int, common_template: Path):
    cache_root = repo / VIDEO_CACHE
    source_manifest = cache_root / "shot_manifest.json"
    source_info = json.loads(source_manifest.read_text())
    shot = next(
        item
        for item in source_info["shots"]
        if int(item["start_frame"]) <= global_index < int(item["end_frame"])
    )
    local_index = global_index - int(shot["start_frame"])
    primary_ids = list(map(int, shot.get("primary_ids") or []))
    if not primary_ids:
        raise ValueError(
            f"Video target frame {global_index} has no selected primary ID"
        )
    target_id = int(shot.get("target_id", primary_ids[0]))
    if target_id not in primary_ids:
        target_id = primary_ids[0]

    shot_video = cache_root / "shots" / f"{shot['id']}.mkv"
    source = _load_video_frame(shot_video, local_index)
    target_masks, _mask_paths, tracking_path, tracking_report = _tracking_masks(
        cache_root, shot["id"], local_index, source.shape[:2]
    )
    if target_id not in target_masks:
        raise ValueError(
            f"Selected raw mask ID {target_id} is absent at global frame {global_index}"
        )
    masks = target_masks
    local_reports = [
        json.loads(line)
        for line in (cache_root / "render" / f"{shot['id']}.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    frame_report = next(
        row for row in local_reports if int(row["shot_frame_index"]) == local_index
    )

    neighbor_ids = list(
        range(max(0, local_index - 2), min(int(shot["frame_count"]), local_index + 3))
    )
    neighbor_data = []
    neighbor_masks_for_crop = [mask for mask in masks.values()]
    for idx in neighbor_ids:
        rgb = source if idx == local_index else _load_video_frame(shot_video, idx)
        frame_masks, _, _, _ = _tracking_masks(
            cache_root, shot["id"], idx, rgb.shape[:2]
        )
        neighbor_masks_for_crop.extend(frame_masks.values())
        neighbor_data.append((idx, rgb, frame_masks))
    crop = _crop_for_masks(
        neighbor_masks_for_crop,
        source.shape[:2],
        padding_ratio=4.0,
        minimum_padding=128,
    )
    case_id = f"video-gf{global_index:05d}"
    case_dir = repo / DEFAULT_OUTPUT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source).save(case_dir / "source-full.png", compress_level=2)
    _save_crop(source, crop, case_dir / "source-crop-native.png")
    ruler_meta = _save_ruler_crop(source, crop, case_dir / "source-crop-ruler.png")
    tight_crop = _tight_crop_for_target(masks[target_id], source.shape[:2])
    _save_crop(source, tight_crop, case_dir / "source-crop-tight-native.png")
    tight_ruler_meta = _save_ruler_crop(
        source, tight_crop, case_dir / "source-crop-tight-ruler.png"
    )
    mask_records = _save_native_masks(case_dir, masks)
    Image.fromarray(_overlay(source, masks, shot["id"])).save(
        case_dir / "raw-mask-overlay-full.jpg", quality=95
    )
    crop_masks = {
        oid: mask[crop[1] : crop[3], crop[0] : crop[2]] for oid, mask in masks.items()
    }
    Image.fromarray(
        _overlay(
            source[crop[1] : crop[3], crop[0] : crop[2]],
            crop_masks,
            shot["id"],
        )
    ).save(case_dir / "raw-mask-overlay-crop.jpg", quality=95)
    Image.fromarray(_crop_mask_overlay(source, masks, tight_crop, shot["id"])).save(
        case_dir / "raw-mask-overlay-tight.jpg", quality=95
    )

    anchor = _anchor_for_shot(cache_root, shot, target_id)
    anchor_rgb = _load_video_frame(shot_video, anchor["frame_index"])
    anchor_masks, anchor_paths, anchor_tracking_path, _ = _tracking_masks(
        cache_root, shot["id"], anchor["frame_index"], anchor_rgb.shape[:2]
    )
    Image.fromarray(anchor_rgb).save(
        case_dir / "anchor-source-full.png", compress_level=2
    )
    Image.fromarray(_overlay(anchor_rgb, anchor_masks, shot["id"])).save(
        case_dir / "anchor-raw-mask-overlay.jpg", quality=95
    )
    anchor_obj = next(
        row
        for row in tracking_report["frames"]
        if int(row["frame_index"]) == anchor["frame_index"]
    )
    anchor_manifest = {
        **anchor,
        "native_mask_paths": {str(k): str(v) for k, v in anchor_paths.items()},
        "tracking_report": str(anchor_tracking_path),
        "native_returned_ids": [
            int(obj["object_id"]) for obj in anchor_obj.get("objects", [])
        ],
    }
    _write_json(case_dir / "anchor-selection.json", anchor_manifest)
    pad_crop = _crop_for_masks(
        [anchor_masks[target_id]],
        anchor_rgb.shape[:2],
        padding_ratio=4.0,
        minimum_padding=128,
    )
    Image.fromarray(
        anchor_rgb[pad_crop[1] : pad_crop[3], pad_crop[0] : pad_crop[2]]
    ).save(case_dir / "anchor-source-crop.png", compress_level=2)
    _save_ruler_crop(anchor_rgb, pad_crop, case_dir / "anchor-source-crop-ruler.png")

    strip_w = 360
    strip_h = 203
    labels_h = 26
    strip = Image.new(
        "RGB", (strip_w * len(neighbor_data), strip_h * 2 + labels_h), "#15191e"
    )
    draw = ImageDraw.Draw(strip)
    font = ImageFont.load_default(size=15)
    for col, (idx, rgb, frame_masks) in enumerate(neighbor_data):
        thumb = Image.fromarray(rgb).resize(
            (strip_w, strip_h), Image.Resampling.LANCZOS
        )
        overlay = Image.fromarray(_overlay(rgb, frame_masks, shot["id"])).resize(
            (strip_w, strip_h), Image.Resampling.LANCZOS
        )
        x = col * strip_w
        draw.text(
            (x + 8, 5),
            f"local {idx} · global {int(shot['start_frame']) + idx}",
            font=font,
            fill="white",
        )
        strip.paste(thumb, (x, labels_h))
        strip.paste(overlay, (x, labels_h + strip_h))
    strip.save(case_dir / "neighbor-frames-source-mask-strip.jpg", quality=93)

    old_video_path = cache_root / "render" / f"{shot['id']}.mp4"
    old_frame = _load_video_frame(old_video_path, local_index)
    Image.fromarray(old_frame).save(
        case_dir / "prior-shot-three-column.png", compress_level=2
    )
    old_panel_w = old_frame.shape[1] // 3
    Image.fromarray(old_frame[:, old_panel_w * 2 : old_panel_w * 3]).save(
        case_dir / "prior-shot-apex-panel.png", compress_level=2
    )

    latest_video_root = repo / "outputs/deterministic-vnext-final-20261010/video"
    latest_video_report = latest_video_root / "vnext-report.json"
    latest_report = json.loads(latest_video_report.read_text())
    latest_frames = {}
    for variant in ("replace", "blur"):
        latest_path = latest_video_root / f"ApexVeil-vnext-{variant}-diagnostics.mp4"
        latest_frame = _load_video_frame(latest_path, global_index)
        if latest_frame.shape[1] % 3:
            raise ValueError(
                f"Unexpected latest diagnostic frame width: {latest_frame.shape}"
            )
        panel_w = latest_frame.shape[1] // 3
        title_height = int(latest_report.get("title_height", 0))
        panel = latest_frame[:, panel_w * 2 : panel_w * 3]
        Image.fromarray(latest_frame).save(
            case_dir / f"latest-result-{variant}-three-column.png", compress_level=2
        )
        Image.fromarray(panel).save(
            case_dir / f"latest-result-{variant}-panel.png", compress_level=2
        )
        for crop_name, crop_bounds in (("crop", crop), ("tight-crop", tight_crop)):
            x0, y0, x1, y1 = crop_bounds
            Image.fromarray(panel[title_height + y0 : title_height + y1, x0:x1]).save(
                case_dir / f"latest-result-{variant}-{crop_name}.png", compress_level=2
            )
        latest_frames[variant] = {
            "path": str(latest_path),
            "sha256": file_sha256(latest_path),
            "global_frame_index": global_index,
            "three_column_frame": f"latest-result-{variant}-three-column.png",
            "right_panel": f"latest-result-{variant}-panel.png",
        }

    master = cache_root / source_info["master"]
    metadata = {
        "case_id": case_id,
        "domain": "video",
        "global_frame_index": global_index,
        "output_time_seconds": global_index / int(source_info["fps"]),
        "source_sample_time_seconds": float(source_info["source_interval_seconds"][0])
        + global_index / int(source_info["fps"]),
        "shot_id": shot["id"],
        "shot_kind": shot["kind"],
        "shot_interval_global": [int(shot["start_frame"]), int(shot["end_frame"])],
        "shot_local_frame_index": local_index,
        "source_video_path": str(shot_video),
        "source_video_sha256": file_sha256(shot_video),
        "master_rgb_path": str(master),
        "master_rgb_sha256": file_sha256(master),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "target_id": target_id,
        "primary_ids": primary_ids,
        "exclude_ids": list(map(int, shot.get("exclude_ids") or [])),
        "native_ids": mask_records,
        "target_frame_render_report": frame_report,
        "tracking_report": str(tracking_path),
        "tracking_report_sha256": file_sha256(tracking_path),
        "crop_context": ruler_meta,
        "crop_tight": tight_ruler_meta,
        "neighbor_frame_indices_local": neighbor_ids,
        "anchor": anchor_manifest,
        "latest_result": {
            "replace": latest_frames["replace"],
            "blur": latest_frames["blur"],
            "report": str(latest_video_report),
            "report_sha256": file_sha256(latest_video_report),
            "renderer_git_commit": latest_report.get("renderer_git_commit"),
            "source_changed_during_evaluation": latest_report.get(
                "source_changed_during_evaluation"
            ),
            "frame_count": latest_report.get("encoded_verification", {})
            .get("replace", {})
            .get("encoded_frame_count"),
            "duration_seconds": latest_report.get("duration_seconds"),
            "title_height_px": latest_report.get("title_height"),
        },
        "prior_shot_comparison": {
            "path": str(old_video_path),
            "sha256": file_sha256(old_video_path),
            "column_labels": ["RGB", "Independent", "ApexVeil"],
            "selected_column": "ApexVeil",
        },
        "approved_template": str(common_template),
        "approved_template_sha256": file_sha256(common_template),
        "coordinate_mapping": {
            "ruler_pixel_to_crop_local": ruler_meta["ruler_pixel_to_crop_local"],
            "crop_local_to_original": ruler_meta["crop_local_to_original"],
            "tight_crop_local_to_original": tight_ruler_meta["crop_local_to_original"],
        },
        "all_returned_masks_are_native_raw": True,
    }
    _write_json(case_dir / "manifest.json", metadata)
    return metadata


def _make_index(output_root: Path, case_manifests: list[dict]):
    cards = []
    for manifest in case_manifests:
        case_id = manifest["case_id"]
        folder = html.escape(case_id, quote=True)
        title = html.escape(case_id)
        raw_ids = ", ".join(str(i) for i in manifest["native_ids"])
        if manifest["domain"] == "image":
            old_links = (
                f'<a href="{folder}/latest-result-replace.png">Latest Replace</a> · '
                f'<a href="{folder}/latest-result-blur.png">Latest Blur</a> · '
                f'<a href="{folder}/latest-report.json">Latest report</a>'
            )
            crop_preview = (
                f'<img loading="lazy" src="{folder}/source-crop-ruler.png" alt="{title} context crop">'
                f'<img loading="lazy" src="{folder}/source-crop-tight-ruler.png" alt="{title} tight target crop">'
                f'<img loading="lazy" src="{folder}/latest-result-replace-tight-crop.png" alt="{title} latest replace crop">'
            )
            strip = ""
            image_flags = ", ".join(manifest["review_flags"]) or "no flags"
            summary = (
                f"{manifest['source_name']} · {manifest['original_resolution']} · "
                f"ID {manifest['target_id']} · {manifest['review_status']} · {image_flags}"
            )
        else:
            old_links = (
                f'<a href="{folder}/latest-result-replace-three-column.png">Latest Replace 3-column</a> · '
                f'<a href="{folder}/latest-result-blur-three-column.png">Latest Blur 3-column</a> · '
                f'<a href="{folder}/latest-result-replace-tight-crop.png">Latest Replace tight crop</a> · '
                f'<a href="{folder}/prior-shot-three-column.png">Prior shot render</a>'
            )
            crop_preview = (
                f'<img loading="lazy" src="{folder}/source-crop-ruler.png" alt="{title} context crop">'
                f'<img loading="lazy" src="{folder}/source-crop-tight-ruler.png" alt="{title} tight target crop">'
                f'<img loading="lazy" src="{folder}/latest-result-replace-tight-crop.png" alt="{title} latest replace crop">'
            )
            strip = f'<img loading="lazy" src="{folder}/neighbor-frames-source-mask-strip.jpg" alt="{title} neighbor strip">'
            video_flags = (
                ", ".join(manifest["target_frame_render_report"].get("flags", []))
                or "no flags"
            )
            summary = (
                f"global frame {manifest['global_frame_index']} · {manifest['shot_id']} "
                f"local {manifest['shot_local_frame_index']} · target ID {manifest['target_id']} · "
                f"{manifest['target_frame_render_report'].get('status', 'unknown')} · {video_flags} · anchor "
                f"global {manifest['anchor']['global_frame_index']} ({manifest['anchor']['selection_policy']})"
            )
        cards.append(
            f"<article><h2>{title}</h2><p>{html.escape(summary)}</p>"
            f"<p>Native SAM IDs: {html.escape(raw_ids)}</p><p>{old_links}</p>"
            f'<div class="previews">{crop_preview}{strip}</div>'
            f'<p><a href="{folder}/manifest.json">Case manifest</a> · '
            f'<a href="{folder}/raw-mask-overlay-full.jpg">Full raw mask overlay</a> · '
            f'<a href="{folder}/raw-mask-overlay-crop.jpg">Context overlay</a> · '
            f'<a href="{folder}/raw-mask-overlay-tight.jpg">Tight overlay</a> · '
            f'<a href="{folder}/native_masks/">Separate native masks</a></p>'
            "<details><summary>New method results (add columns/files here)</summary>"
            "<p>No new solver output is included in this evidence pack.</p></details></article>"
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>ApexVeil Scene Solver Cases</title>"
        "<style>body{margin:0;background:#111419;color:#eef1f4;font:15px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:30px}article{border-top:1px solid #37404a;padding:24px 0}img{max-width:100%;display:block;margin:10px 0}.previews{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}a{color:#8fcbff}small{color:#b7c0ca}</style>"
        "<main><h1>ApexVeil · Private Scene Solver Cases</h1>"
        "<p>Source-bound evidence packs for three image cases and three video frames. Each includes unchanged native masks, coordinate mapping, latest full-run outputs, reports, and a reproducibly selected same-shot anchor.</p>"
        + "".join(cards)
        + "</main></html>"
    )
    (output_root / "index.html").write_text(page, encoding="utf-8")


def build_case_packs(repo_root: Path, output_root: Path):
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    common = output_root / "common"
    common.mkdir(exist_ok=True)
    template_source = repo_root / TEMPLATE_RELATIVE
    common_template = common / "approved-template.png"
    if not common_template.exists():
        shutil.copy2(template_source, common_template)

    cases = []
    for case_id in IMAGE_CASES:
        cases.append(_load_image_case(repo_root, case_id, common_template))
    for global_index in VIDEO_GLOBAL_FRAMES:
        cases.append(_make_video_case(repo_root, global_index, common_template))
    for case in cases:
        case["case_dir"] = case["case_id"]
        _write_json(output_root / case["case_id"] / "manifest.json", case)
    _make_index(output_root, cases)
    root_manifest = {
        "case_count": len(cases),
        "image_cases": list(IMAGE_CASES),
        "video_global_frames": list(VIDEO_GLOBAL_FRAMES),
        "approved_template": str(common_template),
        "approved_template_sha256": file_sha256(common_template),
        "cases": [
            {
                "case_id": case["case_id"],
                "manifest": f"{case['case_id']}/manifest.json",
                "domain": case["domain"],
            }
            for case in cases
        ],
    }
    _write_json(output_root / "manifest.json", root_manifest)
    return root_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_case_packs(args.repo, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
