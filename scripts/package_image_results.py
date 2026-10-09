"""Package complete private image results with explicit correction provenance."""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "automatic", "assisted", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for folder in [
        "originals",
        "replacement",
        "blur",
        "mosaic",
        "automatic-before-correction",
        "previews",
        "reports",
    ]:
        (a.output / folder).mkdir(exist_ok=True)
    labels = ["RGB", "P7+", "Blur", "Mosaic"]
    cards = []
    overview = []
    records = []
    font = ImageFont.load_default(size=19)
    for report_path in sorted(a.automatic.glob("*/report.json")):
        auto = json.loads(report_path.read_text())
        stem = report_path.parent.name
        name = auto["source_name"]
        source = a.source / name
        if sha(source) != auto["source_sha256"]:
            raise ValueError("Original source differs from detection report")
        assisted_path = a.assisted / stem / "report.json"
        assisted = (
            json.loads(assisted_path.read_text()) if assisted_path.exists() else None
        )
        selected = assisted or auto
        replacement = (
            a.assisted / stem if assisted else report_path.parent
        ) / "replace.png"
        paths = {
            "original": ("originals/" + name, source),
            "replacement": ("replacement/" + stem + ".png", replacement),
            "blur": ("blur/" + stem + ".png", report_path.parent / "blur.png"),
            "mosaic": ("mosaic/" + stem + ".png", report_path.parent / "mosaic.png"),
        }
        outputs = {}
        for kind, (relative, src) in paths.items():
            dst = a.output / relative
            shutil.copy2(src, dst)
            if kind != "original":
                with Image.open(dst) as image:
                    if list(image.size) != auto["size"]:
                        raise ValueError("Output dimensions changed")
            outputs[kind] = {
                "path": relative,
                "sha256": sha(dst),
                "bytes": dst.stat().st_size,
            }
        if assisted:
            shutil.copy2(
                report_path.parent / "replace.png",
                a.output / "automatic-before-correction" / f"{stem}.png",
            )
        (a.output / "reports" / f"{stem}-automatic.json").write_text(
            json.dumps(auto, indent=2)
        )
        if assisted:
            (a.output / "reports" / f"{stem}-assisted.json").write_text(
                json.dumps(assisted, indent=2)
            )
        target = selected["results"]["replace"].get("target_id")
        if target is None:
            target = 0
        surface = next(s for s in auto["surfaces"] if s["object_id"] == target)
        mask = np.array(Image.open(report_path.parent / surface["visible_mask"])) > 0
        ys, xs = np.where(mask)
        pad = round(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.4)
        box = (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(auto["size"][0], int(xs.max()) + pad + 1),
            min(auto["size"][1], int(ys.max()) + pad + 1),
        )
        board = Image.new("RGB", (2160, 360), "#17191d")
        draw = ImageDraw.Draw(board)
        for j, (key, label) in enumerate(
            zip(["original", "replacement", "blur", "mosaic"], labels)
        ):
            with Image.open(a.output / outputs[key]["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                tile = ImageOps.contain(
                    im.crop(box), (530, 310), Image.Resampling.LANCZOS
                )
                board.paste(
                    tile,
                    (j * 540 + (540 - tile.width) // 2, 45 + (310 - tile.height) // 2),
                )
                if key == "replacement":
                    thumb = ImageOps.contain(im, (560, 375), Image.Resampling.LANCZOS)
                    ov = Image.new("RGB", (600, 425), "#17191d")
                    ov.paste(thumb, ((600 - thumb.width) // 2, 40))
                    ImageDraw.Draw(ov).text(
                        (15, 10),
                        stem
                        + (
                            " / assisted preview"
                            if assisted
                            else " / automatic preview"
                        ),
                        font=font,
                        fill="white",
                    )
                    overview.append(ov)
            draw.text((j * 540 + 15, 12), label, font=font, fill="white")
        board.save(a.output / "previews" / f"{stem}.jpg", quality=94)
        provenance = selected.get("correction_provenance", {})
        kind = (
            "manual_corners_preview"
            if provenance and not provenance.get("automatic_geometry", True)
            else ("very_dark_override_preview" if assisted else "automatic_preview")
        )
        status = selected["results"]["replace"]["status"]
        record = {
            "file": name,
            "size": auto["size"],
            "target_id": target,
            "detected_candidates": len(auto["surfaces"]),
            "kind": kind,
            "status": status,
            "flags": selected["results"]["replace"]["flags"],
            "correction": provenance,
            "outputs": outputs,
            "outside_visible_masks_changed_pixels": selected["results"]["replace"][
                "outside_visible_masks_changed_pixels"
            ],
        }
        records.append(record)
        note = {
            "manual_corners_preview": "手动估计完整四角；遮挡 mask 保持原样，仍需复核",
            "very_dark_override_preview": "极暗场景：强制生成的待复核预览",
            "automatic_preview": "自动几何预览；尚未通过 AI 或人工质量验收",
        }[kind]
        links = " · ".join(
            f'<a href="{html.escape(outputs[k]["path"], quote=True)}">{label} 全图</a>'
            for k, label in zip(
                ["original", "replacement", "blur", "mosaic"],
                ["原图", "P7+", "模糊", "马赛克"],
            )
        )
        if assisted:
            links += (
                f' · <a href="automatic-before-correction/{stem}.png">修正前自动版</a>'
            )
        cards.append(
            f'<article><h2>{html.escape(stem)}</h2><p>{note}</p><img loading="lazy" src="previews/{stem}.jpg" alt="{html.escape(stem)}：原图、P7+、模糊、马赛克局部对比"><p>{links}</p><small>尺寸 {auto["size"][0]} × {auto["size"][1]} · 候选 {len(auto["surfaces"])} · {html.escape(status)}</small></article>'
        )
    summary = Image.new("RGB", (1800, 425 * ((len(overview) + 2) // 3)), "#111316")
    for i, im in enumerate(overview):
        summary.paste(im, ((i % 3) * 600, (i // 3) * 425))
    summary.save(a.output / "previews" / "overview.jpg", quality=94)
    manifest = {
        "core_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "count": len(records),
        "ai_review": "not_run",
        "privacy_certified": False,
        "records": records,
    }
    (a.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ApexVeil · Test 图片结果</title><style>body{margin:0;background:#0d0f12;color:#e8e9ec;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1400px;margin:auto;padding:40px 24px}h1{font-size:32px}h2{font-size:20px}p,small{color:#aeb4bd}a{color:#ffab7e}article{border-top:1px solid #30343a;padding:22px 0 30px}img{display:block;width:100%;border-radius:8px}small{font-size:13px}</style><main><h1>ApexVeil · Test 图片结果</h1><p>__COUNT__ 张原图，保留原始分辨率。每行依次为 RGB / P7+ / 模糊 / 马赛克，点击链接查看或下载全图。</p><p>默认主车牌替换，其他检出的车牌模糊。手动四角和极暗强制预览逐项注明，修正前自动结果同时保留。强失焦、局部阴影和遮挡边缘仍需复核；本包不是已认证的隐私遮挡成品。</p>"""
        + "".join(cards)
        + "</main></html>"
    )
    (a.output / "index.html").write_text(page.replace("__COUNT__", str(len(records))))
    (a.output / "README.txt").write_text(
        "ApexVeil test image results\nOpen index.html for comparisons and full-resolution links.\nreplacement/: selected previews, including explicitly labeled assisted cases.\nautomatic-before-correction/: unchanged automatic outcomes for those 4 cases.\nblur/ and mosaic/: all detected non-excluded surfaces.\noriginals/: exact supplied source files.\nreports/ and manifest.json: quality flags, source hashes and correction provenance.\nNo AI acceptance or OCR privacy certification has been performed.\n"
    )
    archive = a.output.parent / "ApexVeil-test-images.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as z:
        for f in sorted(a.output.rglob("*")):
            if f.is_file():
                z.write(f, Path("images") / f.relative_to(a.output))
    print(
        json.dumps(
            {
                "images": len(records),
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "kinds": {
                    k: sum(r["kind"] == k for r in records)
                    for k in set(r["kind"] for r in records)
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
