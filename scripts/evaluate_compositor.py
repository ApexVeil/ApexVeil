"""Re-render saved detections without loading a model, for controlled comparisons."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from apexveil.imaging.compositor import load_template, replace_surface, local_bounds
from apexveil.imaging.geometry import order_quad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overrides", type=Path)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--full-resolution", action="store_true")
    args = p.parse_args()
    overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    template = load_template(args.template)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(args.reports.glob("*/report.json")):
        if args.cases and path.parent.name not in args.cases:
            continue
        report = json.loads(path.read_text())
        name = report["source_name"]
        with Image.open(args.images / name) as im:
            rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        override = overrides.get(name, {})
        target = override.get(
            "target_id", report["results"]["replace"].get("target_id", 0)
        )
        if target is None:
            continue
        surface = next(s for s in report["surfaces"] if s["object_id"] == target)
        manual = override.get("quads", {}).get(str(target))
        q = order_quad(manual if manual is not None else surface["quad"])
        mask = (np.array(Image.open(path.parent / surface["visible_mask"])) > 0).astype(
            "uint8"
        )
        start = time.perf_counter()
        rendered, detail = replace_surface(
            rgb, mask, q, template, surface.get("focus", {}).get("sigma_px", 0.35)
        )
        record = {
            "case": name,
            "geometry": "manual_estimate_not_ground_truth" if manual else "automatic",
            "quad": q.tolist(),
            "render_seconds": time.perf_counter() - start,
            "source_flags": surface["flags"],
            "status": "experimental_preview_needs_review",
            "outside_visible_mask_changed": int(
                np.any(rendered != rgb, axis=2)[mask == 0].sum()
            ),
            "detail": detail,
        }
        x0, y0, x1, y1 = local_bounds(mask)
        pad = round(max(x1 - x0, y1 - y0) * 0.35)
        box = [
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(rgb.shape[1], x1 + pad),
            min(rgb.shape[0], y1 + pad),
        ]
        x0, y0, x1, y1 = box
        images = [
            Image.fromarray(rgb[y0:y1, x0:x1]),
            Image.open(path.parent / "replace.png").crop(box),
            Image.fromarray(rendered[y0:y1, x0:x1]),
        ]
        for im in images:
            im.thumbnail((680, 460))
        w, h = max(i.width for i in images), max(i.height for i in images)
        board = Image.new("RGB", (w * 3, h + 38), (22, 23, 27))
        draw = ImageDraw.Draw(board)
        for i, (im, label) in enumerate(
            zip(
                images,
                [
                    "Original",
                    "Previous / may be skipped",
                    "New / " + record["geometry"],
                ],
            )
        ):
            board.paste(im, (i * w, 38))
            draw.text((i * w + 8, 12), label, fill="white")
        stem = path.parent.name
        board.save(args.output / f"{stem}.jpg", quality=96)
        if args.full_resolution:
            Image.fromarray(rendered).save(args.output / f"{stem}.png")
        (args.output / f"{stem}.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        print(
            json.dumps(
                {
                    "case": name,
                    "seconds": record["render_seconds"],
                    "outside_changed": record["outside_visible_mask_changed"],
                }
            ),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
