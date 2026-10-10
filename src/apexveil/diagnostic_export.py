"""Private diagnostic exports from saved native SAM masks; no segmentation is run."""

import colorsys
import hashlib
import html
import json
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HEADER = 54


def manifest_frame_map(manifest):
    """Map global frame indexes to reviewed shot IDs and zero-based local indexes."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    result = {}
    for shot in manifest.get("shots", []):
        start = int(shot["start_frame"])
        count = int(shot["frame_count"])
        end = int(shot.get("end_frame", start + count))
        if end - start != count:
            raise ValueError(f"{shot.get('id')} interval and frame_count disagree")
        for local in range(count):
            global_index = start + local
            if global_index in result:
                raise ValueError(f"overlapping shot intervals at frame {global_index}")
            result[global_index] = {
                "shot_id": shot["id"],
                "local_frame": local,
                "target_id": shot.get("target_id"),
                "primary_ids": shot.get("primary_ids", []),
                "exclude_ids": shot.get("exclude_ids", []),
            }
    if result and sorted(result) != list(range(max(result) + 1)):
        raise ValueError("shot intervals must cover a contiguous timeline from frame 0")
    expected_count = manifest.get("frame_count")
    if expected_count is not None and len(result) != int(expected_count):
        raise ValueError(
            f"manifest frame_count disagrees with shot intervals: {expected_count} != {len(result)}"
        )
    return result


def stable_color(object_id, scope=""):
    b = hashlib.sha256(f"{scope}:{int(object_id)}".encode()).digest()
    return tuple(
        round(x * 255)
        for x in colorsys.hsv_to_rgb(int.from_bytes(b[:2], "big") / 65535, 0.8, 0.94)
    )


def rgb(a):
    a = np.asarray(a)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 RGB")
    return a.copy()


def color_native_masks(
    frame, masks, *, shot_id, target_id=None, primary_ids=(), exclude_ids=()
):
    out = rgb(frame)
    h, w = out.shape[:2]
    union = np.zeros((h, w), bool)
    edges = []
    for oid, raw in sorted(masks.items()):
        m = np.asarray(raw)
        m = m[..., 0] if m.ndim == 3 else m
        if m.shape != (h, w):
            raise ValueError(f"mask {oid} dimensions do not match RGB frame")
        m = m > 0
        union |= m
        c = np.asarray(stable_color(oid, shot_id), np.uint8)
        if m.any():
            out[m] = np.rint(
                out[m].astype(float) * 0.52 + c.astype(float) * 0.48
            ).astype(np.uint8)
            e = np.zeros_like(m)
            e[:, 1:] |= m[:, 1:] != m[:, :-1]
            e[1:, :] |= m[1:, :] != m[:-1, :]
            edges.append((e, c))
    for e, c in edges:
        out[e] = c
    return out, union


def _frames(source):
    if isinstance(source, (str, Path)):
        path = str(source)
        pinfo = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        s = json.loads(pinfo.stdout)["streams"][0]
        w, h = int(s["width"]), int(s["height"])
        size = w * h * 3
        p = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            while True:
                b = p.stdout.read(size)
                if not b:
                    break
                if len(b) != size:
                    raise ValueError(f"truncated frame in {path}")
                yield np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
            err = p.stderr.read().decode("utf-8", "replace")
            if p.wait():
                raise RuntimeError(err[-2000:])
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            p.stdout.close()
            p.stderr.close()
    else:
        for x in source:
            yield rgb(x)


def _load_masks(root, record, shape):
    folder = Path(root) / f"frame_{int(record['frame_index']):06d}"
    result = {}
    for obj in record.get("objects", []):
        oid = int(obj["object_id"])
        path = folder / obj["mask"]
        if not path.is_file():
            raise FileNotFoundError(f"missing native mask ID {oid}: {path}")
        with Image.open(path) as im:
            mask = np.asarray(im.convert("L")) > 0
        if mask.shape != shape:
            raise ValueError(f"native mask dimensions differ: {path}")
        result[oid] = mask
    return result


def _encoder(path, w, h, fps):
    return subprocess.Popen(
        [
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
            f"{w}x{h}",
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
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _compose(
    left,
    mid,
    right,
    shot,
    target,
    primary,
    excluded,
    index,
    status="",
    native_ids=(),
    frame_ids=(),
    empty_ids=(),
    right_label="ApexVeil",
):
    h, w = left.shape[:2]
    if mid.shape != left.shape or right.shape != left.shape:
        raise ValueError("three panel frames must have matching dimensions")
    out = np.empty((h + HEADER, w * 3, 3), np.uint8)
    out[HEADER:, :w] = left
    out[HEADER:, w : 2 * w] = mid
    out[HEADER:, 2 * w :] = right
    head = Image.new("RGB", (w * 3, HEADER), "#15191e")
    d = ImageDraw.Draw(head)
    f = ImageFont.load_default(size=16)
    d.text((12, 7), "RGB", font=f, fill="white")
    d.text((w + 12, 7), "SAM 3.1 native masks · all returned IDs", font=f, fill="white")
    x = w + 12
    ids = sorted(set(native_ids) | set(primary) | set(excluded))
    if not ids:
        d.text(
            (w + 12, 29),
            "No reviewed IDs; consult native mask report",
            font=f,
            fill="#ffcf8b",
        )
    for oid in ids:
        text = (
            f"ID {oid}"
            + (
                " target"
                if oid == target
                else " excluded (shown)"
                if oid in excluded
                else " primary"
                if oid in primary
                else " native"
            )
            + (
                " empty"
                if oid in empty_ids
                else " absent"
                if oid not in frame_ids
                else ""
            )
        )
        d.rectangle((x, 31, x + 12, 43), fill=stable_color(oid, shot))
        d.text((x + 17, 28), text, font=f, fill="#d7dbe0")
        x += 115 + len(text) * 5
    d.text(
        (2 * w + 12, 7),
        right_label,
        font=f,
        fill="white",
    )
    d.text(
        (2 * w + 12, 29),
        f"{shot} · frame {index}" + (f" · {status}" if status else ""),
        font=f,
        fill="white",
    )
    out[:HEADER] = np.asarray(head)
    return out


def export_video_diagnostics(
    rgb_source,
    tracking_report,
    tracking_root,
    replacement_source,
    blur_source,
    output_dir,
    *,
    shot_id,
    fps,
    target_id=None,
    primary_ids=(),
    exclude_ids=(),
    frame_status=None,
    audio_source=None,
    audio_muxer=None,
):
    """Export RGB | native colored SAM masks | Replace/Blur; sources are paths or RGB iterables."""
    report = (
        json.loads(Path(tracking_report).read_text())
        if isinstance(tracking_report, (str, Path))
        else tracking_report
    )
    records = report.get("frames", [])
    n = int(report.get("frame_count", len(records)))
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(records) != n or [int(r["frame_index"]) for r in records] != list(range(n)):
        raise ValueError("mask report indexes must be contiguous local frame indexes")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sources = map(_frames, (rgb_source, replacement_source, blur_source))
    its = list(sources)
    paths = {
        "replace": outdir / f"{shot_id}-replace-diagnostics.mp4",
        "blur": outdir / f"{shot_id}-blur-diagnostics.mp4",
    }
    enc = None
    primary = set(map(int, primary_ids))
    excluded = set(map(int, exclude_ids))
    allids = sorted(
        {int(o["object_id"]) for r in records for o in r.get("objects", [])}
    )
    empty_frames = {oid: [] for oid in allids}
    try:
        for i, record in enumerate(records):
            try:
                left, rep, blu = (next(it) for it in its)
            except StopIteration as e:
                raise ValueError("input and mask frame counts differ") from e
            h, w = left.shape[:2]
            if enc is None:
                enc = {k: _encoder(p, 3 * w, h + HEADER, fps) for k, p in paths.items()}
            masks = _load_masks(tracking_root, record, (h, w))
            mid, _ = color_native_masks(
                left,
                masks,
                shot_id=shot_id,
                target_id=target_id,
                primary_ids=primary,
                exclude_ids=excluded,
            )
            empty_ids = {
                oid for oid, mask in masks.items() if not np.asarray(mask).any()
            }
            for object_id in empty_ids:
                empty_frames.setdefault(object_id, []).append(i)
            for key, right in (("replace", rep), ("blur", blu)):
                enc[key].stdin.write(
                    _compose(
                        left,
                        mid,
                        right,
                        shot_id,
                        target_id,
                        primary,
                        excluded,
                        i,
                        (frame_status or {}).get(i, ""),
                        allids,
                        set(masks),
                        empty_ids,
                        f"ApexVeil · {key.title()}",
                    ).tobytes()
                )
        for it in its:
            try:
                next(it)
            except StopIteration:
                continue
            raise ValueError("input has extra frames")
    except Exception:
        if enc:
            for p in enc.values():
                if p.poll() is None:
                    p.kill()
                p.wait()
                if p.stdin and not p.stdin.closed:
                    p.stdin.close()
                if p.stderr:
                    p.stderr.close()
        raise
    finally:
        for it in its:
            if hasattr(it, "close"):
                it.close()
    encoder_errors = []
    for key, p in (enc or {}).items():
        try:
            p.stdin.close()
            err = p.stderr.read().decode("utf-8", "replace")
            return_code = p.wait()
        except Exception as error:  # noqa: BLE001 - wait for every encoder on failures
            if p.poll() is None:
                p.kill()
            p.wait()
            encoder_errors.append(f"{key}: {type(error).__name__}: {error}")
            continue
        if return_code:
            encoder_errors.append(f"{key}: ffmpeg exited {return_code}: {err[-2000:]}")
    if encoder_errors:
        raise RuntimeError("Diagnostic encoder failure: " + " | ".join(encoder_errors))
    if audio_source is not None:
        if audio_muxer is None:
            from apexveil.video_pipeline import mux_audio as audio_muxer
        for path in paths.values():
            silent = path.with_name(path.stem + "-silent.mp4")
            path.replace(silent)
            try:
                audio_muxer(silent, audio_source, path)
            finally:
                silent.unlink(missing_ok=True)
    allids = sorted(
        {int(o["object_id"]) for r in records for o in r.get("objects", [])}
    )
    meta = {
        "shot_id": shot_id,
        "frame_count": n,
        "fps": fps,
        "layout": ["RGB", "SAM 3.1 colored raw mask", "ApexVeil Replace or Blur"],
        "native_ids_visible": allids,
        "native_id_legend": [
            {
                "object_id": oid,
                "color_rgb": list(stable_color(oid, shot_id)),
                "role": "target"
                if oid == target_id
                else "excluded (still displayed)"
                if oid in excluded
                else "primary"
                if oid in primary
                else "native",
            }
            for oid in allids
        ],
        "empty_native_id_frames": {
            str(oid): idx for oid, idx in empty_frames.items() if idx
        },
        "excluded_ids_still_visible": sorted(set(allids) & excluded),
        "outputs": {k: str(v) for k, v in paths.items()},
        "audio_source": str(audio_source) if audio_source else None,
    }
    (outdir / f"{shot_id}-diagnostics.json").write_text(json.dumps(meta, indent=2))
    return meta


def _image(x):
    if isinstance(x, (str, Path)):
        with Image.open(x) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return rgb(x)


def _image_board(rgb_im, mask_im, right, labels, crop=None, title=""):
    if crop:
        rgb_im, mask_im, right = (im.crop(crop) for im in (rgb_im, mask_im, right))
    tw, th = 720, 480
    board = Image.new("RGB", (tw * 3, th + 44), "#171a1f")
    d = ImageDraw.Draw(board)
    f = ImageFont.load_default(size=18)
    d.text((12, 4), title, font=f, fill="#bbc1c9")
    for i, (im, label) in enumerate(zip((rgb_im, mask_im, right), labels)):
        d.text((i * tw + 12, 24), label, font=f, fill="white")
        thumb = ImageOps.contain(im, (tw - 18, th - 8), Image.Resampling.LANCZOS)
        board.paste(
            thumb, (i * tw + (tw - thumb.width) // 2, 44 + (th - thumb.height) // 2)
        )
    return board


def export_image_diagnostics(
    source_rgb,
    replacement_rgb,
    blur_rgb,
    raw_masks,
    output_dir,
    *,
    image_id,
    target_id=None,
    primary_ids=(),
    exclude_ids=(),
    provenance=None,
):
    """Write full/detail JPEG panels. raw_masks maps every native ID to mask path or array."""
    a, b, c = map(_image, (source_rgb, replacement_rgb, blur_rgb))
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("RGB, Replace, and Blur dimensions must match")
    masks = {}
    for oid, value in raw_masks.items():
        if isinstance(value, (str, Path)):
            with Image.open(value) as im:
                masks[int(oid)] = np.asarray(im.convert("L")) > 0
        else:
            masks[int(oid)] = np.asarray(value) > 0
    color, _ = color_native_masks(
        a,
        masks,
        shot_id=image_id,
        target_id=target_id,
        primary_ids=primary_ids,
        exclude_ids=exclude_ids,
    )
    ims = list(map(Image.fromarray, (a, color, b, c)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_id).stem
    selected = masks.get(int(target_id)) if target_id is not None else None
    if selected is None or not selected.any():
        selected = np.logical_or.reduce(list(masks.values())) if masks else None
    if selected is None or not selected.any():
        box = (0, 0, a.shape[1], a.shape[0])
    else:
        ys, xs = np.where(selected)
        pad = round(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * 0.45)
        box = (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(a.shape[1], int(xs.max()) + pad + 1),
            min(a.shape[0], int(ys.max()) + pad + 1),
        )
    names = {}
    for mode, right in (("replace", ims[2]), ("blur", ims[3])):
        for view, crop in (("full", None), ("detail", box)):
            filename = f"{stem}-{view}-{mode}.jpg"
            _image_board(
                ims[0],
                ims[1],
                right,
                ("RGB", "SAM 3.1 colored raw mask", f"ApexVeil {mode.title()}"),
                crop,
                f"{stem} · {view}",
            ).save(out / filename, "JPEG", quality=91, optimize=True)
            names[f"{view}_{mode}"] = filename
    report = {
        "image_id": image_id,
        "dimensions": [a.shape[1], a.shape[0]],
        "target_id": target_id,
        "primary_ids": list(map(int, primary_ids)),
        "exclude_ids": list(map(int, exclude_ids)),
        "native_ids_in_masks": sorted(masks),
        "native_id_legend": [
            {
                "object_id": i,
                "color_rgb": list(stable_color(i, image_id)),
                "role": "target"
                if i == target_id
                else "excluded (still displayed)"
                if i in set(map(int, exclude_ids))
                else "primary"
                if i in set(map(int, primary_ids))
                else "native",
                "empty": not bool(m.any()),
            }
            for i, m in sorted(masks.items())
        ],
        "empty_native_ids": sorted(i for i, m in masks.items() if not m.any()),
        "excluded_ids_still_visible": sorted(set(map(int, exclude_ids)) & set(masks)),
        "crop_xyxy": list(box),
        "provenance": provenance or {},
        "diagnostics": names,
        "claim": "SAM 3.1 masks are raw model outputs for review, not ground truth.",
    }
    (out / f"{stem}-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report


def _safe_local_href(value):
    path = str(value).replace("\\\\", "/")
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ":" in path:
        return None
    if any(part in {"", ".", ".."} for part in parsed.parts):
        return None
    return html.escape(path, quote=True)


def write_image_index(output_dir, reports, title="ApexVeil diagnostics"):
    """Create a local index showing review state, output downloads, and panels."""
    root = Path(output_dir)
    cards = []
    for report in reports:
        stem = Path(report["image_id"]).stem
        names = report["diagnostics"]
        provenance = report.get("provenance", {})
        category = html.escape(str(provenance.get("category", "unspecified")))
        status = html.escape(str(provenance.get("status", "not provided")))
        correction = provenance.get("correction", {})
        note = html.escape(str(correction.get("note", "")))
        flags = provenance.get("flags") or []
        if isinstance(flags, str):
            flags = [flags]
        if flags:
            flag_html = (
                "<ul>"
                + "".join(f"<li>{html.escape(str(flag))}</li>" for flag in flags)
                + "</ul>"
            )
        else:
            flag_html = "<p>No review flags recorded.</p>"
        links = provenance.get("links") or {}
        if not links:
            # Standalone image-pipeline exports are stored beside the index.
            links = {
                key: relative
                for key, relative in {
                    "replace": "replace.png",
                    "blur": "blur.png",
                    "report": "report.json",
                }.items()
                if (root / relative).is_file()
            }
            if "report" not in links:
                diagnostic_report = f"{stem}-report.json"
                if (root / diagnostic_report).is_file():
                    links["report"] = diagnostic_report
        link_labels = {
            "replace": "Download full-resolution Replace PNG",
            "blur": "Download full-resolution Blur PNG",
            "report": "Open report JSON",
        }
        link_html = []
        for key, label in link_labels.items():
            href = _safe_local_href(links.get(key, ""))
            if href:
                download = " download" if key in {"replace", "blur"} else ""
                link_html.append(f'<a href="{href}"{download}>{html.escape(label)}</a>')
        links_markup = (
            '<p class="downloads">' + " · ".join(link_html) + "</p>"
            if link_html
            else ""
        )
        full_replace = html.escape(names["full_replace"], quote=True)
        full_blur = html.escape(names["full_blur"], quote=True)
        detail_replace = html.escape(names["detail_replace"], quote=True)
        detail_blur = html.escape(names["detail_blur"], quote=True)
        card = (
            f"<article><h2>{html.escape(stem)}</h2>"
            f"<p>Provenance: {category}. Review status: <strong>{status}</strong>. {note}</p>"
            f"{flag_html}{links_markup}"
            '<label>Right output <select><option value="replace">Replace</option><option value="blur">Blur</option></select></label>'
            f'<img loading="lazy" src="{full_replace}" data-replace="{full_replace}" data-blur="{full_blur}" alt="{html.escape(stem)} full view">'
            f'<img loading="lazy" src="{detail_replace}" data-replace="{detail_replace}" data-blur="{detail_blur}" alt="{html.escape(stem)} target detail">'
            "<p>RGB | SAM 3.1 colored raw mask | ApexVeil. Every returned mask ID remains visible, including excluded IDs; raw masks are model outputs, not ground truth. Product quality acceptance has not been run.</p></article>"
        )
        cards.append(card)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{margin:0;background:#101317;color:#eef0f2;font:15px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}article{border-top:1px solid #343a42;padding:20px 0}p,label{color:#bac1c9}img{display:block;width:100%;margin:14px 0}select{margin:8px;padding:6px;background:#252b33;color:white}.downloads a{color:#ffb486;margin-right:8px}li{color:#e3c6a8}</style>"
        f"<main><h1>{html.escape(title)}</h1><p>Choose Replace or Blur for the right panel. Each image includes full-view and target-detail panels. Review status and flags are shown for every image.</p>{''.join(cards)}</main>"
        '<script>for(const a of document.querySelectorAll("article")){const s=a.querySelector("select");s.onchange=()=>a.querySelectorAll("img").forEach(i=>i.src=i.dataset[s.value])}</script></html>'
    )
    (root / "index.html").write_text(page, encoding="utf-8")
    return root / "index.html"
