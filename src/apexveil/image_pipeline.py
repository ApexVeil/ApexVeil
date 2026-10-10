"""Inspectable still-image jobs using the same surface contract as video."""

import hashlib
import json
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .estimation import get_default_estimators
from .imaging.compositor import (
    load_template,
    local_bounds,
    redact_surface,
    replace_surface,
)
from .imaging.focus import estimate_edge_softness
from .imaging.geometry import (
    assess_geometry,
    order_quad,
    template_geometry_flags,
    visible_containment,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def prepare_surfaces(
    rgb,
    detections,
    directory,
    quad_overrides=None,
    *,
    template_aspect_ratio=3.0,
    estimators=None,
):
    """Preserve native visibility and estimate a separately documented plane."""
    estimators = estimators or get_default_estimators()
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
            "visible_mask_sha256": hashlib.sha256(
                (directory / mask_path).read_bytes()
            ).hexdigest(),
            "visible_pixels": int((mask > 0).sum()),
            "detection": item.diagnostics,
            "flags": [],
        }
        try:
            key = str(item.object_id)
            if quad_overrides and key in quad_overrides:
                quad = order_quad(quad_overrides[key])
                quality = assess_geometry(mask, quad)
                surface.update(
                    quad=quad.tolist(),
                    geometry="user_supplied_corners",
                    refinement=[],
                    quality=quality,
                    flags=list(quality["flags"]),
                    geometry_estimate={
                        "method": "user_supplied_corners",
                        "confidence": None,
                    },
                )
            else:
                estimate = estimators.geometry.estimate(
                    rgb, mask, template_aspect_ratio
                )
                quad = estimate.quad
                surface.update(
                    geometry=estimate.diagnostics.get("method", "automatic_estimation"),
                    quality=estimate.diagnostics,
                    geometry_estimate={
                        "confidence": estimate.confidence,
                        **estimate.diagnostics,
                    },
                    flags=list(estimate.flags),
                    refinement=estimate.diagnostics.get("rgb_edge_refinement", []),
                )
                if quad is not None:
                    surface["quad"] = quad.tolist()
            if quad is not None:
                surface["focus"] = estimate_edge_softness(rgb, quad)
            pixels = rgb[mask > 0].astype(np.float32) / 255
            if len(pixels) and np.percentile(pixels.mean(axis=1), 97) < 0.04:
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
    *,
    estimators=None,
    edit_support_out=None,
    noise_seed=0,
):
    """Require selection for multi-target replacement; blur detected others."""
    estimators = estimators or get_default_estimators()
    support_union = np.zeros(rgb.shape[:2], bool)
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
                for flag in template_geometry_flags(
                    q, template.alpha.shape[1] / template.alpha.shape[0]
                ):
                    if flag not in surface["flags"]:
                        surface["flags"].append(flag)
                if visible_containment(item.visible_mask, q) < 0.90:
                    surface["flags"].append("quad_does_not_explain_visible_surface")
            uncertain = bool(surface["flags"])
            blocking = bool(
                set(surface["flags"])
                & {
                    "partially_out_of_frame",
                    "geometry_failed",
                    "insufficient_boundary_support",
                    "quad_does_not_explain_visible_surface",
                    "degenerate_projected_surface",
                    "ill_conditioned_projected_surface",
                }
            )
            if "quad" not in surface or (
                blocking
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
            local_support = {}
            before = output
            output, detail = replace_surface(
                output,
                item.visible_mask,
                np.asarray(surface["quad"], np.float32),
                template,
                softness,
                estimators=estimators,
                edit_support_out=local_support,
                noise_seed=noise_seed + item.object_id,
            )
            x0, y0, x1, y1 = local_support["bbox_xyxy"]
            support_union[y0:y1, x0:x1] |= local_support["support"]
            protected = local_support["protected"]
            detail["protected_changed_pixels"] = int(
                np.any(output[y0:y1, x0:x1] != before[y0:y1, x0:x1], axis=2)[
                    protected
                ].sum()
            )
            if detail["protected_changed_pixels"]:
                raise RuntimeError("Renderer modified protected foreground")
            if detail.get("boundary", {}).get("needs_review"):
                flags.append(f"target_{item.object_id}_boundary_needs_review")
            detail["softness_source"] = (
                "manual" if blur_sigma is not None else "edge_estimate"
            )
            if blur_sigma is None and softness > 4:
                flags.append(f"target_{item.object_id}_defocus_preview_needs_review")
            if uncertain:
                flags.append(f"target_{item.object_id}_uncertain_preview")
        else:
            output, detail = redact_surface(output, item.visible_mask, action, strength)
            support_union |= item.visible_mask > 0
        edits.append({"object_id": item.object_id, "action": action, **detail})
    outside = int((np.any(output != rgb, axis=2) & ~support_union).sum())
    if outside:
        raise RuntimeError("Renderer modified pixels outside declared edit support")
    if edit_support_out is not None:
        edit_support_out["support"] = support_union
    return output, {
        "estimators": estimators.metadata(),
        "outside_declared_edit_support_changed_pixels": outside,
        "declared_edit_support_pixels": int(support_union.sum()),
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
    estimators=None,
):
    estimators = estimators or get_default_estimators()
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as source:
        icc_profile = source.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(source)
        rgb = np.array(oriented.convert("RGB"))
    decode_seconds = time.perf_counter() - started
    detections, segmentation = segmenter.detect(Image.fromarray(rgb), prompt, refine)
    override = override or {}
    surfaces = prepare_surfaces(
        rgb,
        detections,
        output,
        override.get("quads"),
        estimators=estimators,
        template_aspect_ratio=template.alpha.shape[1] / template.alpha.shape[0]
        if template
        else 3.0,
    )
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
        support_record = {}
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
            estimators=estimators,
            edit_support_out=support_record,
        )
        support = support_record.get("support", np.zeros(rgb.shape[:2], bool))
        support_name = f"edit_support_{effect}.png"
        Image.fromarray(support.astype(np.uint8) * 255).save(output / support_name)
        detail["edit_support_mask"] = support_name
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
    diagnostic = None
    if "replace" in results and "blur" in results:
        from .diagnostic_export import export_image_diagnostics, write_image_index

        diagnostic = export_image_diagnostics(
            rgb,
            results["replace"],
            results["blur"],
            {d.object_id: d.visible_mask for d in detections},
            output,
            image_id=path.stem,
            target_id=renders["replace"].get("target_id", target_id),
            exclude_ids=exclude_ids,
            provenance={
                "category": "manual_corners" if override.get("quads") else "automatic",
                "geometry_ground_truth": False,
                "status": renders["replace"]["status"],
                "flags": renders["replace"]["flags"],
                "estimators": estimators.metadata(),
            },
        )
        write_image_index(output, [diagnostic], title=f"ApexVeil · {path.stem}")
    report = {
        "schema_version": 2,
        "estimators": estimators.metadata(),
        "source_name": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": [rgb.shape[1], rgb.shape[0]],
        "icc_preserved": bool(icc_profile),
        "diagnostics": diagnostic,
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
        except Exception as error:  # noqa: BLE001 -- retain per-file batch failures
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
