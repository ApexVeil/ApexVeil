"""Run the optional Codex advisor against a source-bound private case pack."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from apexveil.backends.codex_advisor import CodexSceneAdvisor
from apexveil.imaging.color import linear, luma
from apexveil.imaging.compositor import load_template


def source_light_samples(case_dir, metadata, quad):
    """Observed dark-ink samples, not a claim of recovered illumination."""
    if quad is None:
        return None
    rgb = np.asarray(Image.open(case_dir / "source-full-oriented.png").convert("RGB"))
    path = metadata["native_ids"][str(metadata["target_id"])]["path"]
    mask = np.asarray(Image.open(case_dir / path).convert("L")) > 0
    q = np.asarray(quad, np.float32)
    dst = np.array([[0, 0], [639, 0], [639, 191], [0, 191]], np.float32)
    h = cv2.getPerspectiveTransform(q, dst)
    rect = cv2.warpPerspective(rgb, h, (640, 192))
    valid = (
        cv2.warpPerspective(
            mask.astype(np.uint8), h, (640, 192), flags=cv2.INTER_NEAREST
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
            good = valid[y0:y1, x0:x1]
            dark = ink[y0:y1, x0:x1] & good
            pixels = values[y0:y1, x0:x1]
            brightness = lum[y0:y1, x0:x1]
            if dark.sum() >= 5:
                selected = dark & (brightness <= np.percentile(brightness[dark], 35))
                sample = np.median(pixels[selected], axis=0).tolist()
                rows.append(
                    {
                        "uv": [u, v],
                        "dark_ink_linear_rgb": sample,
                        "samples": int(selected.sum()),
                    }
                )
    return {
        "coordinates": "UV of current hypothesis, not necessarily correct complete plane",
        "interpretation": "Observed dark ink includes unknown ink reflectance plus additive veil; use as lighting evidence, not exact ground truth.",
        "rows": rows,
    }


def run(
    case_dir,
    output,
    profile="luna-medium",
    proxy_url=None,
    candidate=None,
    feedback=None,
    current_report=None,
):
    case_dir, output = Path(case_dir).resolve(), Path(output).resolve()
    metadata = json.loads((case_dir / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    source_name = (
        "source-full-oriented.png"
        if metadata["domain"] == "image"
        else "source-full.png"
    )
    full = Image.open(case_dir / source_name).convert("RGB")
    source_size = list(full.size)
    full.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
    full.save(output / "full-context.jpg", quality=95)
    crop = metadata["crop_tight"]
    report_path = case_dir / "latest-report.json"
    current = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        target = next(
            (
                s
                for s in report.get("surfaces", [])
                if s["object_id"] == metadata["target_id"]
            ),
            {},
        )
        current = {
            "quad_source_pixels": target.get("quad"),
            "geometry_flags": target.get("flags"),
        }
        if target.get("quad"):
            x0, y0, _, _ = crop["crop_xyxy_original"]
            current["quad_crop_pixels"] = [[x - x0, y - y0] for x, y in target["quad"]]
        edits = report.get("results", {}).get("replace", {}).get("edits", [])
        edit = next(
            (
                e
                for e in edits
                if e.get("object_id") == metadata["target_id"]
                and e.get("action") == "replace"
            ),
            {},
        )
        current["old_appearance"] = edit.get("appearance")
    if current_report is not None:
        current_report = json.loads(Path(current_report).read_text())
        chosen = current_report.get("selected_candidate") or {}
        current = {
            "quad_source_pixels": chosen.get("quad"),
            "geometry_flags": chosen.get("uncertainties"),
            "geometry_guidance": current_report.get("geometry_guidance"),
            "appearance_guidance": current_report.get("appearance_guidance"),
            "render_status": current_report.get("status"),
        }
        if chosen.get("quad"):
            x0, y0, _, _ = crop["crop_xyxy_original"]
            current["quad_crop_pixels"] = [[x - x0, y - y0] for x, y in chosen["quad"]]
    images = [
        output / "full-context.jpg",
        case_dir / "source-crop-tight-native.png",
        case_dir / "source-crop-tight-ruler.png",
        case_dir / "raw-mask-overlay-tight.jpg",
        Path(candidate).resolve()
        if candidate
        else case_dir / "latest-result-replace-tight-crop.png",
    ]
    light_samples = (
        source_light_samples(case_dir, metadata, current.get("quad_source_pixels"))
        if metadata["domain"] == "image"
        else None
    )
    template = load_template(Path(metadata["approved_template"]))
    context = {
        "declared_template_aspect_prior": template.alpha.shape[1]
        / template.alpha.shape[0],
        "case_id": metadata["case_id"],
        "source_size": source_size,
        "tight_crop": crop,
        "current_estimate_is_not_ground_truth": current,
        "observed_source_light_samples": light_samples,
        "image_order": [
            "full source context",
            "native source tight crop",
            "same source tight crop with coordinate ruler",
            "raw SAM mask overlay",
            "candidate replacement tight crop",
        ],
    }
    if metadata["domain"] == "video":
        for filename, description in [
            (
                "anchor-source-crop.png",
                "nearby clearer reference crop; different frame, not target coordinates",
            ),
            (
                "neighbor-frames-source-mask-strip.jpg",
                "temporal context strip; not target coordinates",
            ),
        ]:
            path = case_dir / filename
            if path.is_file():
                images.append(path)
                context["image_order"].append(description)
        context["shot"] = metadata.get("shot_id")
        context["global_frame"] = metadata.get("global_frame_index")
        context["temporal_guidance_scope"] = (
            "Judge and guide the target frame; reference images establish a plane hypothesis, never substitute source pixels."
        )
    images.append(Path(metadata["approved_template"]))
    context["image_order"].append(
        "approved original transparent P7+ artwork; white/silver logo fragments can be visible under clipping or occlusion"
    )
    prompt = """You are the scene advisor for ApexVeil. Inspect the actual attached pixels at both scene and detail scale. The goal is a physically plausible P7+ branded replacement, not merely a successful warp. Return only the requested JSON; do not use tools or edit code in this proposal stage.
The SAM mask describes visible old plate, NOT the complete plate plane. Distinguish real plate edges, foreground occlusion edges, and image-boundary clipping. A person/leg must not become a plate edge. A clipped plate can have corners outside the source image. Propose up to TWO plausible complete-plane quads when the current geometry is wrong; do not squeeze the whole logo into a visible fragment. If geometry is already plausible, candidate_quads may be empty. The approved black plate has a silver P7+ logo; it is rendered by a fixed compositor, never generated by you.
The images are listed in context.image_order. All geometry candidate xy values MUST be native tight-crop pixel coordinates with origin at the unbordered source tight crop; the ruler has the border described in metadata. They may be negative/outside crop. Quad order TL,TR,BR,BL. Edge labels refer to complete physical plate edges. Physical aspect is an explicit assumption, not the aspect of the visible fragment. A declared full-template aspect prior is provided: prefer it when hidden geometry is ambiguous, unless actual scene evidence contradicts it; do not estimate physical aspect from an occluded fragment.
Appearance is DIFFERENT coordinates: flare/shading xy are normalized COMPLETE PLATE UV [0,1], not crop pixels. Colors and exposure are LINEAR RGB, not 0..255/sRGB; values in the metadata are numeric reference only. Light should remain spatially coherent across replacement and surroundings; warm additive lens veil must not become a hard black cut. Old green/blue face albedo is not illumination. Preserve approved artwork. Use null for parameters you cannot responsibly infer; give no more than 8 lighting points. Avoid global over-brightening and copying green tint. The existing inward-shrink renderer can leave a colored rim: do not suggest shrinking the new plate to hide errors.
Return a complete proposed recipe. The current estimate includes prior guidance when supplied. Null appearance parameters revert to automatic estimation, so repeat any previous guided lighting field that should be retained.
assessment.acceptable evaluates the SHOWN candidate, not your unrendered proposed changes. State specific visible faults and a next action; do not claim success before rerender. If the candidate is already adequate, do not invent corrections. The best plan may keep geometry and change only a spatial additive field.
CONTEXT:\n""" + json.dumps(context, ensure_ascii=False)
    if feedback:
        prompt += (
            "\nRENDERED-CANDIDATE FEEDBACK (re-evaluate actual pixels, do not merely agree):\n"
            + str(feedback)
        )
    advice, record = CodexSceneAdvisor(
        profile=profile, proxy_url=proxy_url, timeout_seconds=180
    ).run(output, prompt, images)
    print(
        json.dumps(
            {
                "case": metadata["case_id"],
                "status": record["status"],
                "seconds": record["elapsed_seconds"],
                "model": record["requested_model"],
                "error": record.get("error"),
                "usage": record["usage"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return advice, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--profile", choices=["luna-medium", "sol-medium"], default="luna-medium"
    )
    p.add_argument("--proxy")
    p.add_argument("--candidate", type=Path)
    p.add_argument("--feedback-file", type=Path)
    p.add_argument("--current-report", type=Path)
    a = p.parse_args()
    _, r = run(
        a.case,
        a.output,
        a.profile,
        a.proxy,
        a.candidate,
        a.feedback_file.read_text() if a.feedback_file else None,
        a.current_report,
    )
    if r["status"] != "proposal_available":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
