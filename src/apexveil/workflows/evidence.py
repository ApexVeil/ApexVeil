"""Source-bound visual/numeric evidence shared by image and video guidance."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..diagnostic_export import color_native_masks, stable_color
from ..imaging.color import linear, luma
from ..imaging.compositor import load_template, local_bounds


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def save_preview(rgb, path, max_size=(1600, 1100)):
    image = Image.fromarray(np.asarray(rgb, np.uint8))
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    image.save(path)
    return str(Path(path).resolve())


def scene_crop(mask):
    x0, y0, x1, y1 = local_bounds(mask)
    pad = max(24, round(max(x1 - x0, y1 - y0) * 0.5))
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(mask.shape[1], x1 + pad),
        min(mask.shape[0], y1 + pad),
    ]


def _light_samples(rgb, mask, quad):
    if quad is None:
        return None
    quad = np.asarray(quad, np.float32)
    transform = cv2.getPerspectiveTransform(
        quad, np.array([[0, 0], [639, 0], [639, 191], [0, 191]], np.float32)
    )
    rect = cv2.warpPerspective(rgb, transform, (640, 192))
    valid = (
        cv2.warpPerspective(
            (mask > 0).astype(np.uint8), transform, (640, 192), flags=cv2.INTER_NEAREST
        )
        > 0
    )
    values = linear(rect)
    lum = luma(values)
    filled = cv2.morphologyEx(
        lum, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    )
    ink = (filled - lum) > np.maximum(0.00005, filled * 0.12)
    rows = []
    for u in [0.05, 0.35, 0.65, 0.95]:
        for v in [0.2, 0.8]:
            x0, x1 = max(0, int((u - 0.06) * 640)), min(640, int((u + 0.06) * 640))
            y0, y1 = max(0, int((v - 0.18) * 192)), min(192, int((v + 0.18) * 192))
            selected = valid[y0:y1, x0:x1] & ink[y0:y1, x0:x1]
            brightness = lum[y0:y1, x0:x1]
            if selected.sum() < 5:
                continue
            selected &= brightness <= np.percentile(brightness[selected], 35)
            rows.append(
                {
                    "uv": [u, v],
                    "dark_ink_linear_rgb": np.median(
                        values[y0:y1, x0:x1][selected], axis=0
                    ).tolist(),
                    "samples": int(selected.sum()),
                }
            )
    return {
        "coordinates": "complete-plane UV of current hypothesis, not ground truth",
        "interpretation": "Dark ink contains unknown reflectance plus additive veil; old green albedo is not illumination.",
        "rows": rows,
    }


def build_scene_evidence(
    rgb,
    mask,
    template_path,
    output_dir,
    *,
    baseline_rgb=None,
    scene_report=None,
    target_id=0,
    foreground_mask=None,
    extra_images=(),
    context=None,
):
    rgb = np.asarray(rgb)
    mask = np.asarray(mask)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or mask.shape != rgb.shape[:2]
    ):
        raise ValueError("Evidence requires native RGB and matching mask")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    crop = scene_crop(mask)
    x0, y0, x1, y1 = crop
    context = json_value(context or {})
    scope = context.get("palette_scope", "scene-evidence")
    overlay, _ = color_native_masks(
        rgb, {int(target_id): mask}, shot_id=scope, target_id=target_id
    )
    images = []
    descriptions = []

    def add(im, name, label, preview=False):
        path = out / name
        if preview:
            save_preview(im, path)
        else:
            Image.fromarray(np.asarray(im, np.uint8)).save(path)
        images.append(str(path))
        descriptions.append(label)

    add(rgb, "source-full.jpg", "full source context", True)
    add(
        rgb[y0:y1, x0:x1],
        "source-crop.png",
        "native source crop, geometry coordinate origin",
    )
    add(
        overlay[y0:y1, x0:x1],
        "mask-crop.jpg",
        "raw native target-mask overlay, model estimate not ground truth",
    )
    ruler = Image.new("RGB", (x1 - x0 + 112, y1 - y0 + 112), (28, 31, 36))
    ruler.paste(Image.fromarray(rgb[y0:y1, x0:x1]), (56, 56))
    draw = ImageDraw.Draw(ruler)
    for x in range(0, x1 - x0, 50):
        draw.text((x + 56, 10), str(x), fill="white")
    for y in range(0, y1 - y0, 50):
        draw.text((6, y + 56), str(y), fill="white")
    path = out / "source-ruler.png"
    ruler.save(path)
    images.append(str(path))
    descriptions.append(
        "same crop with56px ruler border; geometry xy excludes this border"
    )
    if baseline_rgb is not None:
        baseline_rgb = np.asarray(baseline_rgb)
        if baseline_rgb.shape != rgb.shape:
            raise ValueError("Baseline must match source RGB dimensions")
        add(
            baseline_rgb[y0:y1, x0:x1],
            "baseline-crop.png",
            "automatic baseline result crop, not an accepted final result",
        )
    if foreground_mask is not None:
        fg = np.asarray(foreground_mask)
        if fg.shape != mask.shape:
            raise ValueError("Foreground evidence must match source dimensions")
        fg_overlay, _ = color_native_masks(rgb, {0: fg}, shot_id="semantic-foreground")
        add(
            fg_overlay[y0:y1, x0:x1],
            "foreground-crop.jpg",
            "independent raw SAM person estimate, not ground truth",
        )
    for index, extra in enumerate(extra_images):
        if isinstance(extra, (str, Path)):
            source = Path(extra).resolve()
            path = out / f"extra-{index:02d}{source.suffix.lower()}"
            if source != path:
                shutil.copy2(source, path)
            images.append(str(path))
            descriptions.append(
                f"additional temporal/reference evidence{index}; not target coordinates"
            )
        else:
            im, label = extra
            add(im, f"extra-{index:02d}.png", str(label))
    template_path = Path(template_path).resolve()
    asset = out / ("approved-template" + template_path.suffix.lower())
    if template_path != asset:
        shutil.copy2(template_path, asset)
    images.append(str(asset))
    descriptions.append(
        "approved original asset with native alpha; fixed renderer preserves artwork"
    )
    template = load_template(asset)
    chosen = (scene_report or {}).get("selected_candidate") or {}
    numeric = {
        "source_size": [rgb.shape[1], rgb.shape[0]],
        "target_id": int(target_id),
        "target_color_rgb": list(stable_color(target_id, scope)),
        "crop_xyxy": crop,
        "ruler_border_px": 56,
        "geometry_coordinate_space": "advisor output quads: native crop pixels, source_xy=crop_xy+[x,y]",
        "current_scene_geometry_coordinate_space": "full source pixels; subtract crop_xyxy[:2] before proposing a crop-space quad",
        "appearance_coordinate_space": "complete plate UV0..1, linearRGB",
        "declared_template_aspect_prior": template.alpha.shape[1]
        / template.alpha.shape[0],
        "current_scene_is_not_ground_truth": json_value(scene_report or {}),
        "observed_source_light_samples": _light_samples(rgb, mask, chosen.get("quad")),
        "context": context,
        "image_order": descriptions,
    }
    (out / "numeric-context.json").write_text(
        json.dumps(numeric, ensure_ascii=False, indent=2)
    )
    manifest = {
        "kind": "scene_guidance_evidence",
        "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "native_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "mask_ground_truth": False,
        "crop_xyxy": crop,
        "source_size": numeric["source_size"],
        "images": [
            {
                "path": p,
                "label": label,
                "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            }
            for p, label in zip(images, descriptions)
        ],
        "numeric_context": numeric,
    }
    manifest_path = out / "case-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    evidence = {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "image_paths": images,
        "crop_xyxy": crop,
    }
    evidence["prompt"] = build_advisor_prompt(evidence)
    return evidence


def build_advisor_prompt(evidence):
    context = evidence["manifest"]["numeric_context"]
    return """You are the ApexVeil scene advisor. Inspect source, native-mask evidence, baseline and approved artwork at scene and detail scale. Return only the requested JSON; do not use tools, edit code or generate logo pixels.
The visible SAM plate mask is NOT the complete physical plane. Distinguish real plate edges, foreground occlusion and image clipping. Do not squeeze an entire logo into a visible fragment. Your proposed geometry candidate quads are TL,TR,BR,BL native CROP pixel coordinates; the current_scene_is_not_ground_truth report instead records full SOURCE pixel quads, so subtract the crop origin before reusing those coordinates; xy excludes the56px ruler border and may lie outside the crop. Propose at most2 plausible complete planes when needed. If current geometry is plausible, leave candidate_quads empty. Template physical aspect is an explicit prior, not measured from an occluded fragment.
Appearance coordinates are DIFFERENT: flare/shading xy are complete-plate UV0..1; values are LINEAR RGB, never0..255. Old green/blue albedo is not illumination. Keep spatial colored additive flare separate from multiplicative shading; warm glare should not become a hard black cut. Use at most8 lighting points and null for uncertain parameters. Native alpha/artwork are fixed. Independent person masks are estimates, not ground truth; preserve foreground and do not draw a new semantic mask.
This is one GUIDANCE call, not a final quality review. assessment.acceptable describes the SHOWN baseline, not your unrendered recipe. Explain concrete visible issues and provide bounded proposed parameters; never claim rerendered success in advance. Extra temporal images establish context only, not coordinates for the representative frame. No other provider/model fallback is permitted.
CONTEXT:\n""" + json.dumps(context, ensure_ascii=False)
