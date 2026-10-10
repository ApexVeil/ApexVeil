"""Render one source-bound evidence case through the shared scene solver."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from apexveil.backends.codex_advisor import to_source_geometry, validate_guidance
from apexveil.diagnostic_export import export_image_diagnostics, write_image_index
from apexveil.imaging.compositor import load_template, redact_surface
from apexveil.imaging.scene_geometry import (
    propose_scene_geometry,
    select_scene_geometry,
)
from apexveil.scene_pipeline import solve_scene


def run(case_dir, output, *, advice_path=None, foreground_path=None):
    case_dir, output = Path(case_dir).resolve(), Path(output).resolve()
    m = json.loads((case_dir / "manifest.json").read_text())
    source = case_dir / (
        "source-full-oriented.png" if m["domain"] == "image" else "source-full.png"
    )
    rgb = np.asarray(Image.open(source).convert("RGB"))
    masks = {
        int(k): np.asarray(Image.open(case_dir / v["path"]).convert("L")) > 0
        for k, v in m["native_ids"].items()
    }
    target = m["target_id"]
    mask = masks[target]
    template = load_template(Path(m["approved_template"]))
    anchors = []
    if m["domain"] == "video" and m.get("anchor"):
        a = m["anchor"]
        am = (
            np.asarray(Image.open(a["native_mask_paths"][str(target)]).convert("L")) > 0
        )
        ar = np.asarray(Image.open(case_dir / "anchor-source-full.png").convert("RGB"))
        aq = select_scene_geometry(
            propose_scene_geometry(
                ar, am, aspect=template.alpha.shape[1] / template.alpha.shape[0]
            )
        )
        if aq is not None:
            anchors = [
                {
                    "rgb": ar,
                    "mask": am,
                    "quad": aq.quad,
                    "frame_index": a["frame_index"],
                }
            ]
    advice = None
    geometry = None
    appearance = None
    provenance = "automatic_scene_solver"
    if advice_path is not None:
        advice_path = Path(advice_path).resolve()
        advice = validate_guidance(json.loads(advice_path.read_text()))
        if not advice["plate_confirmed"]:
            raise ValueError("Advisor did not confirm target plate")
        geometry = to_source_geometry(
            advice,
            m["crop_tight"]["crop_xyxy_original"],
            source_size=[rgb.shape[1], rgb.shape[0]],
        )
        appearance = {**advice["appearance"], "provenance": str(advice_path.parent)}
        provenance = "agent_guided_scene_solver"
    foreground = (
        None
        if foreground_path is None
        else np.asarray(Image.open(foreground_path).convert("L")) > 0
    )
    rendered, report, candidates = solve_scene(
        rgb,
        mask,
        template,
        geometry_guidance=geometry,
        appearance_guidance=appearance,
        foreground_mask=foreground,
        anchors=anchors,
        noise_seed=int(m.get("global_frame_index", 0)) * 1000003 + (target + 1) * 9176,
    )
    blur = rgb.copy()
    additional_edits = []
    excluded = set(m.get("exclude_ids", []))
    for object_id, visible in masks.items():
        if object_id not in excluded and visible.any():
            safe_visible = visible if foreground is None else visible & ~foreground
            blur, _ = redact_surface(blur, safe_visible, "blur")
            if object_id != target:
                # An overlapping observation must not blur the selected artwork.
                safe_visible = safe_visible & ~mask
                previous = rendered.copy()
                rendered, redaction = redact_surface(rendered, safe_visible, "blur")
                additional_edits.append(
                    {
                        "object_id": object_id,
                        "action": "blur",
                        "changed_pixels": int(
                            np.any(rendered != previous, axis=2).sum()
                        ),
                        "details": redaction,
                    }
                )
    if foreground is not None:
        # Redaction feathering must obey the same foreground contract as replace.
        rendered[foreground] = rgb[foreground]
        blur[foreground] = rgb[foreground]
    report["invariant_scope"] = "target_scene_before_additional_id_redactions"
    report["additional_edits"] = additional_edits
    report["final_output"] = {
        "changed_pixels": int(np.any(rendered != rgb, axis=2).sum()),
        "foreground_changed_pixels": 0
        if foreground is None
        else int(np.any(rendered != rgb, axis=2)[foreground].sum()),
        "rgb_sha256": hashlib.sha256(rendered.tobytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered).save(output / "replace.png", compress_level=2)
    Image.fromarray(blur).save(output / "blur.png", compress_level=2)
    x0, y0, x1, y1 = m["crop_tight"]["crop_xyxy_original"]
    Image.fromarray(rendered[y0:y1, x0:x1]).save(output / "candidate-tight.png")
    full = Image.fromarray(rendered)
    full.thumbnail((1600, 1100))
    full.save(output / "full.jpg", quality=95)
    report.update(
        case_id=m["case_id"],
        method=provenance,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        advice_path=None if advice_path is None else str(advice_path),
        advice_sha256=None
        if advice_path is None
        else hashlib.sha256(advice_path.read_bytes()).hexdigest(),
        foreground_path=None if foreground_path is None else str(foreground_path),
        candidate_count=len(candidates),
        quality_accepted=False,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    d = export_image_diagnostics(
        rgb,
        rendered,
        blur,
        masks,
        output,
        image_id=m["case_id"],
        target_id=target,
        primary_ids=m.get("primary_ids", [target]),
        exclude_ids=excluded,
        provenance={
            "category": provenance,
            "status": report["status"],
            "flags": report.get("selected_candidate", {}).get("uncertainties", [])
            if report.get("selected_candidate")
            else [report.get("unresolved_reason")],
            "links": {
                "replace": "replace.png",
                "blur": "blur.png",
                "report": "report.json",
            },
        },
    )
    write_image_index(output, [d], title=f"ApexVeil · {m['case_id']} · {provenance}")
    print(
        json.dumps(
            {
                "case": m["case_id"],
                "method": provenance,
                "status": report["status"],
                "candidate_count": len(candidates),
                "rendered": report["rendered"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--advice", type=Path)
    p.add_argument("--foreground", type=Path)
    a = p.parse_args()
    cv2.setNumThreads(2)
    report = run(a.case, a.output, advice_path=a.advice, foreground_path=a.foreground)
    if not report["rendered"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
