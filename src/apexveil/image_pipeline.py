"""Inspectable still-image jobs using the same surface contract as video."""

import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .imaging.compositor import (
    load_template,
    replace_surface,
    redact_surface,
    local_bounds,
)
from .imaging.geometry import fit_quad, edge_refine, assess_geometry, order_quad
from .imaging.focus import estimate_edge_softness

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def prepare_surfaces(rgb, detections, directory, quad_overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    surfaces = []
    for item in detections:
        mask = item.visible_mask
        mask_path = f"mask_{item.object_id:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / mask_path)
        surface = {
            "object_id": item.object_id,
            "score": item.score,
            "visible_mask": mask_path,
            "visible_pixels": int((mask > 0).sum()),
            "detection": item.diagnostics,
            "flags": [],
        }
        try:
            key = str(item.object_id)
            if quad_overrides and key in quad_overrides:
                quad = order_quad(quad_overrides[key])
                method, logs = "user_supplied_corners", []
            else:
                initial, method = fit_quad(mask)
                quad, logs = edge_refine(rgb, initial)
            quality = assess_geometry(mask, quad)
            surface.update(
                quad=quad.tolist(),
                geometry=method,
                refinement=logs,
                quality=quality,
                focus=estimate_edge_softness(rgb, quad),
                flags=quality["flags"],
            )
            pixels = rgb[mask > 0].astype(np.float32) / 255
            if np.percentile(pixels.mean(axis=1), 97) < 0.04:
                surface["flags"].append("very_dark_surface")
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            surface["flags"] = ["geometry_failed"]
            surface["geometry_error"] = str(error)
        surfaces.append(surface)
    return surfaces


def render(
    rgb,
    detections,
    surfaces,
    mode,
    template=None,
    target_id=None,
    exclude_ids=(),
    strength=0.12,
    blur_sigma=None,
    allow_uncertain=False,
):
    """Require selection for multi-target replacement; blur detected others."""
    output = rgb.copy()
    active = [d for d in detections if d.object_id not in exclude_ids]
    flags, edits = [], []
    if not active:
        return output, {
            "status": "needs_review",
            "flags": ["no_active_detection"],
            "edits": [],
        }
    if mode == "replace":
        if template is None:
            raise ValueError("Replacement requires a template")
        if target_id is None and len(active) == 1:
            target_id = active[0].object_id
        if target_id is None:
            return output, {
                "status": "needs_review",
                "flags": ["select_primary_target"],
                "edits": [],
            }
        if target_id not in {d.object_id for d in active}:
            return output, {
                "status": "needs_review",
                "flags": ["selected_target_not_found"],
                "edits": [],
            }
    by_id = {s["object_id"]: s for s in surfaces}
    for item in active:
        surface = by_id[item.object_id]
        action = mode if mode != "replace" or item.object_id == target_id else "blur"
        if action == "replace":
            if "quad" in surface:
                q = np.asarray(surface["quad"], np.float32)
                lengths = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
                aspect = (lengths[0] + lengths[2]) / max(1.0, lengths[1] + lengths[3])
                artwork_aspect = template.alpha.shape[1] / template.alpha.shape[0]
                # A tiny visible fragment must not become the entire wide design.
                # Strong true foreshortening can also trigger this conservative gate.
                if (
                    aspect / artwork_aspect < 0.55
                    and "possible_occlusion_or_extreme_foreshortening"
                    not in surface["flags"]
                ):
                    surface["flags"].append(
                        "possible_occlusion_or_extreme_foreshortening"
                    )
            uncertain = bool(surface["flags"])
            if "quad" not in surface or (
                uncertain
                and not allow_uncertain
                and surface.get("geometry") != "user_supplied_corners"
            ):
                flags.append(f"target_{item.object_id}_geometry_needs_review")
                edits.append(
                    {
                        "object_id": item.object_id,
                        "action": "skipped",
                        "reasons": surface["flags"],
                    }
                )
                continue
            softness = (
                blur_sigma
                if blur_sigma is not None
                else surface.get("focus", {}).get("sigma_px", 0.35)
            )
            output, detail = replace_surface(
                output,
                item.visible_mask,
                np.asarray(surface["quad"], np.float32),
                template,
                softness,
            )
            detail["softness_source"] = (
                "manual" if blur_sigma is not None else "edge_estimate"
            )
            if blur_sigma is None and softness > 4:
                flags.append(f"target_{item.object_id}_defocus_preview_needs_review")
            if uncertain:
                flags.append(f"target_{item.object_id}_uncertain_preview")
        else:
            output, detail = redact_surface(output, item.visible_mask, action, strength)
        edits.append({"object_id": item.object_id, "action": action, **detail})
    return output, {
        "status": "needs_review" if flags else "rendered_unreviewed",
        "flags": flags,
        "target_id": target_id,
        "edits": edits,
        "ai_review": "not_run",
        "privacy_certified": False,
    }


def previews(original, results, detections, surfaces, directory):
    # One overview and one local contact sheet per detected target.
    def tile(image, label, size=(540, 360)):
        im = Image.fromarray(image)
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size[0], size[1] + 28), "#14171d")
        canvas.paste(im, ((size[0] - im.width) // 2, 28 + (size[1] - im.height) // 2))
        ImageDraw.Draw(canvas).text((10, 8), label, fill="white")
        return canvas

    images = [("Original", original)] + list(results.items())
    board = Image.new("RGB", (540 * len(images), 388), "#14171d")
    for i, (name, im) in enumerate(images):
        board.paste(tile(im, name), (i * 540, 0))
    board.save(directory / "overview.jpg", quality=92)
    diagnostic = original.copy()
    for item in detections:
        mask = item.visible_mask > 0
        diagnostic[mask] = (
            diagnostic[mask] * 0.6 + np.array([255, 65, 65]) * 0.4
        ).astype(np.uint8)
    for surface in surfaces:
        if "quad" in surface:
            q = np.rint(surface["quad"]).astype(np.int32)
            cv2.polylines(diagnostic, [q], True, (0, 230, 200), 3)
            cv2.putText(
                diagnostic,
                str(surface["object_id"]),
                tuple(q[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 230, 0),
                3,
            )
    diag = Image.fromarray(diagnostic)
    diag.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
    diag.save(directory / "geometry.jpg", quality=93)
    for item in detections:
        x0, y0, x1, y1 = local_bounds(item.visible_mask)
        pad = round(max(x1 - x0, y1 - y0) * 0.25)
        x0, y0, x1, y1 = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(original.shape[1], x1 + pad),
            min(original.shape[0], y1 + pad),
        )
        board = Image.new("RGB", (540 * len(images), 388), "#14171d")
        for i, (name, im) in enumerate(images):
            board.paste(tile(im[y0:y1, x0:x1], name), (i * 540, 0))
        board.save(
            directory / f"target_{item.object_id:03d}_comparison.jpg", quality=94
        )


def process_image(
    path,
    output,
    segmenter,
    mode="replace",
    template=None,
    target_id=None,
    override=None,
    comparison=False,
    strength=0.12,
    blur_sigma=None,
    allow_uncertain=False,
    prompt="license plate",
    refine=True,
):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as source:
        icc_profile = source.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(source)
        rgb = np.array(oriented.convert("RGB"))
    decode_seconds = time.perf_counter() - started
    detections, segmentation = segmenter.detect(Image.fromarray(rgb), prompt, refine)
    override = override or {}
    surfaces = prepare_surfaces(rgb, detections, output, override.get("quads"))
    target_id = override.get("target_id", target_id)
    exclude_ids = override.get("exclude_ids", [])
    results, renders = {}, {}
    modes = [mode] + (
        [
            m
            for m in ("replace", "blur", "mosaic")
            if m != mode and (m != "replace" or template)
        ]
        if comparison
        else []
    )
    render_start = time.perf_counter()
    for effect in modes:
        rendered, detail = render(
            rgb,
            detections,
            surfaces,
            effect,
            template,
            target_id,
            exclude_ids,
            strength,
            blur_sigma,
            allow_uncertain,
        )
        detail["changed_pixels"] = int(np.any(rendered != rgb, axis=2).sum())
        union = np.zeros(rgb.shape[:2], bool)
        for detection in detections:
            if detection.object_id not in exclude_ids:
                union |= detection.visible_mask > 0
        detail["outside_visible_masks_changed_pixels"] = int(
            (np.any(rendered != rgb, axis=2) & ~union).sum()
        )
        results[effect] = rendered
        renders[effect] = detail
        Image.fromarray(rendered).save(
            output / f"{effect}.png", icc_profile=icc_profile
        )
    render_seconds = time.perf_counter() - render_start
    previews(rgb, results, detections, surfaces, output)
    report = {
        "schema_version": 1,
        "source_name": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": [rgb.shape[1], rgb.shape[0]],
        "icc_preserved": bool(icc_profile),
        "decode_seconds": decode_seconds,
        "segmentation": segmentation,
        "render_and_save_seconds": render_seconds,
        "total_seconds": time.perf_counter() - started,
        "surfaces": surfaces,
        "results": renders,
        "template": template.metadata if template else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_images(settings, source, output, **options):
    from .backends.image import ImageSegmenter

    source, output = source.resolve(), output.resolve()
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(
            p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
    if not paths:
        raise ValueError("No supported images found")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Image stems must be unique in a batch output")
    template_path = options.pop("template_path", None)
    template = load_template(template_path) if template_path else None
    if options.get("mode", "replace") == "replace" and template is None:
        raise ValueError("--template is required for replacement")
    overrides_path = options.pop("overrides_path", None)
    overrides = json.loads(overrides_path.read_text()) if overrides_path else {}
    segmenter = ImageSegmenter(settings)
    report = {
        "kind": "image_pipeline_evaluation",
        "backend": segmenter.metadata,
        "ai_review": "not_run",
        "records": [],
        "errors": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        try:
            item = process_image(
                path,
                output / path.stem,
                segmenter,
                template=template,
                override=overrides.get(path.name),
                **options,
            )
            report["records"].append(item)
        except Exception as error:
            report["errors"].append(
                {"file": path.name, "error": f"{type(error).__name__}: {error}"}
            )
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "processed": path.name,
                    "completed": len(report["records"]),
                    "errors": len(report["errors"]),
                }
            ),
            flush=True,
        )
    return report
