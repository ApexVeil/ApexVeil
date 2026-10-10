"""Bounded single-shot video workflow for the local Studio.

The caller serializes GPU work and may reuse ImageSegmenter.predictor. Preparing
uses native SAM3.1 text initialization/tracking; rendering uses scene candidates,
one fixed trajectory and the shared scene compositor. No cross-ID association or
cross-cut propagation is attempted.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..backends.codex_advisor import (
    PROFILES,
    CodexSceneAdvisor,
    to_source_geometry,
    validate_guidance,
)
from ..backends.sam31 import build_predictor
from ..diagnostic_export import color_native_masks, stable_color
from ..imaging.compositor import load_template, redact_surface
from ..imaging.photometry import TemporalVideoAppearance
from ..imaging.scene_geometry import (
    propose_scene_geometry,
    select_scene_geometry,
    select_scene_trajectory,
)
from ..imaging.scene_temporal import stabilize_scene_trajectory
from ..imaging.temporal import flow_prediction
from ..scene_pipeline import solve_scene
from ..video_pipeline import command, file_sha256, probe, verify_encoded_output

MAX_DURATION = 8.0
MAX_FPS = 30
MAX_SIZE = (1920, 1080)


def _progress(callback, message, fraction=None):
    if callback is not None:
        callback(message, fraction)


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


class _VideoFrames:
    """Reuse decoded RGB only when the complete clip fits a bounded cache.

    Longer clips retain the one-frame policy to avoid geometry/render LRU
    thrashing. Caching changes neither source resolution nor decode pixels.
    """

    def __init__(self, path, count, *, cache_bytes=192 * 1024 * 1024):
        self.cache_bytes = max(0, int(cache_bytes))
        self.decoded = {}
        self.cache_complete_clip = None
        self.path, self.count = Path(path), int(count)
        self.capture = None
        self.next_index = 0
        self.cached_index = None
        self.cached = None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index in self.decoded:
            return self.decoded[index]
        if index == self.cached_index and self.cached is not None:
            return self.cached
        if self.capture is None:
            self.capture = cv2.VideoCapture(
                str(self.path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            if not self.capture.isOpened():
                raise RuntimeError("Cannot decode prepared video")
            self.next_index = 0
        # Decode gaps sequentially only when the complete clip fits the budget.
        sequential = self.cache_complete_clip is not False and self.next_index <= index
        if self.next_index != index and not sequential:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.next_index = index
        while self.next_index <= index:
            current = self.next_index
            ok, bgr = self.capture.read()
            if not ok:
                raise RuntimeError(f"Prepared video ended before frame {current}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.cache_complete_clip is None:
                self.cache_complete_clip = rgb.nbytes * self.count <= self.cache_bytes
            if self.cache_complete_clip:
                self.decoded[current] = rgb
            self.next_index = current + 1
            self.cached_index, self.cached = current, rgb
            if not self.cache_complete_clip and current < index:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                self.next_index = index
        return self.cached

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.cached_index = self.cached = None
        self.decoded.clear()
        self.cache_complete_clip = None


def _rate(stream):
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            value = Fraction(stream.get(key, "0"))
            if 1 <= value <= 240:
                return value
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError("Cannot determine a supported video frame rate")


def _processing_size(stream):
    width, height = int(stream["width"]), int(stream["height"])
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for data in stream.get("side_data_list", []):
        if "rotation" in data:
            rotation = float(data["rotation"])
    if abs(round(rotation)) % 180 == 90:
        width, height = height, width
    if min(width, height) < 2:
        raise ValueError("Video dimensions are too small")
    factor = min(1.0, MAX_SIZE[0] / width, MAX_SIZE[1] / height)
    size = (max(2, int(width * factor) // 2 * 2), max(2, int(height * factor) // 2 * 2))
    return (width, height), size


def _check_continuous(frames):
    """Conservative cut/flash check; it is not a scene-boundary guarantee."""
    previous = histogram = None
    events = []
    for index in range(len(frames)):
        small = cv2.resize(frames[index], (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        current_hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(current_hist, current_hist, alpha=1, norm_type=cv2.NORM_L1)
        if previous is not None:
            difference = float(np.abs(small.astype(float) - previous).mean() / 255)
            distance = float(
                cv2.compareHist(histogram, current_hist, cv2.HISTCMP_BHATTACHARYYA)
            )
            if difference > 0.28 and distance > 0.45:
                events.append(
                    {
                        "frame_index": index,
                        "rgb_difference": difference,
                        "histogram_distance": distance,
                    }
                )
        previous, histogram = small.astype(float), current_hist
    return events


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _native_record(response, directory, shape):
    index = int(response["frame_index"])
    outputs = response["outputs"]
    ids = _array(outputs.get("out_obj_ids", [])).reshape(-1)
    masks = _array(outputs.get("out_binary_masks", []))
    scores = _array(outputs.get("out_probs", [])).reshape(-1)
    directory.mkdir(parents=True, exist_ok=True)
    objects = []
    for slot, identity in enumerate(ids):
        mask = (masks[slot].squeeze() > 0).astype(np.uint8)
        if mask.shape != shape:
            raise ValueError("Native mask dimensions differ from the decoded source")
        identity = int(identity)
        path = directory / f"object_{identity:04d}.png"
        Image.fromarray(mask * 255).save(path)
        x, y, width, height = cv2.boundingRect(mask)
        score = (
            float(scores[slot])
            if slot < len(scores) and np.isfinite(scores[slot])
            else None
        )
        objects.append(
            {
                "object_id": identity,
                "mask": path.name,
                "score": score,
                "visible_pixels": int(mask.sum()),
                "bbox": [x, y, x + width, y + height],
            }
        )
    return {"frame_index": index, "objects": objects}


def _read_masks(prepared, index):
    shape = (prepared["public"]["height"], prepared["public"]["width"])
    result = {}
    for identity, path in prepared["_mask_paths"][index].items():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            raise ValueError(f"Cached mask is missing or has wrong dimensions: {path}")
        result[int(identity)] = (mask > 0).astype(np.uint8)
    return result


def _clarity(rgb, mask):
    x, y, w, h = cv2.boundingRect(mask)
    if not w or not h:
        return 0.0
    gray = cv2.cvtColor(rgb[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    crop_penalty = (
        0.35
        if x <= 1 or y <= 1 or x + w >= rgb.shape[1] - 1 or y + h >= rgb.shape[0] - 1
        else 1.0
    )
    return (
        math.sqrt(float(mask.sum()))
        * (0.25 + 0.75 * sharpness / (sharpness + 80))
        * crop_penalty
    )


def _previews(prepared, output):
    frames, records = prepared["_frames"], prepared["_records"]
    best, counts = {}, Counter()
    for i, record in enumerate(records):
        masks = _read_masks(prepared, i)
        if not masks:
            continue
        rgb = frames[i]
        for obj in record["objects"]:
            identity = obj["object_id"]
            if not obj["visible_pixels"]:
                continue
            counts[identity] += 1
            clarity = _clarity(rgb, masks[identity])
            if identity not in best or clarity > best[identity][0]:
                best[identity] = (clarity, i, obj)
    previews = {}
    for index in sorted({entry[1] for entry in best.values()} or {0}):
        rgb = frames[index]
        masks = _read_masks(prepared, index)
        overlay, _ = color_native_masks(rgb, masks, shot_id=prepared["_palette_scope"])
        source_path, mask_path = (
            output / f"preview-{index:06d}-rgb.png",
            output / f"preview-{index:06d}-sam.png",
        )
        Image.fromarray(rgb).save(source_path)
        Image.fromarray(overlay).save(mask_path)
        previews[index] = (str(source_path), str(mask_path))
    targets = []
    for identity, (clarity, index, obj) in sorted(best.items()):
        targets.append(
            {
                "id": identity,
                "object_id": identity,
                "area": obj["visible_pixels"],
                "bbox": obj["bbox"],
                "score": obj["score"],
                "color_rgb": list(stable_color(identity, prepared["_palette_scope"])),
                "representative_frame_index": index,
                "visible_frames": counts[identity],
                "clarity_score": clarity,
                "source_preview_path": previews[index][0],
                "mask_preview_path": previews[index][1],
            }
        )
    default = (
        max(targets, key=lambda t: (t["visible_frames"], t["clarity_score"]))
        if targets
        else None
    )
    index = default["representative_frame_index"] if default else 0
    return targets, index, previews[index], None if default is None else default["id"]


def prepare_video(
    source_path,
    output_dir,
    settings,
    *,
    predictor=None,
    start=0,
    duration=2,
    progress=None,
):
    """Prepare one SDR continuous shot, <=8 seconds, <=1080p and <=30fps.

    Oversized/high-fps input is explicitly resized/resampled. Suspected cuts or
    flashes are rejected before SAM. The Root scheduler owns single-GPU access.
    """
    begun = time.perf_counter()
    if (
        not np.isfinite([start, duration]).all()
        or start < 0
        or not 0 < duration <= MAX_DURATION
    ):
        raise ValueError(
            "Choose one continuous shot with 0 < duration <= 8 seconds and start >= 0"
        )
    source, output = Path(source_path).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    clip = output / "prepared-video.mkv"
    if source == clip:
        raise ValueError("Input cannot be the workflow's prepared-video output")
    cv2.setNumThreads(1)
    _progress(progress, "Checking the video and selected interval", 0.02)
    info = probe(source)
    stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("Input has no video stream")
    if stream.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise ValueError(
            "This workflow currently accepts SDR video; convert HDR explicitly first"
        )
    source_rate = _rate(stream)
    rate = min(source_rate, Fraction(MAX_FPS))
    source_size, size = _processing_size(stream)
    available = float(info.get("format", {}).get("duration", stream.get("duration", 0)))
    if available <= start:
        raise ValueError("Start is outside the source duration")
    requested = float(duration)
    duration = min(requested, available - start)
    warnings = []
    if size != source_size:
        warnings.append(
            f"Processing resized from {source_size[0]}x{source_size[1]} to {size[0]}x{size[1]}"
        )
    if rate != source_rate:
        warnings.append(
            f"Processing resampled from {float(source_rate):.3f} to {float(rate):.3f} fps"
        )
    if duration < requested:
        warnings.append("Selected interval shortened at source end")
    timings = {"inspection_seconds": time.perf_counter() - begun}
    phase = time.perf_counter()
    expected = max(1, math.ceil(duration * float(rate) - 1e-6))
    _progress(progress, "Preparing the video segment", 0.08)
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            source,
            "-t",
            str(duration),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={rate}:start_time=0,scale={size[0]}:{size[1]}:flags=lanczos,setsar=1",
            "-frames:v",
            str(expected),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            "-threads",
            "1",
            clip,
        ]
    )
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not 1 <= count <= math.ceil(MAX_DURATION * MAX_FPS):
        raise ValueError("Prepared clip has an invalid or unbounded frame count")
    frames = _VideoFrames(clip, count)
    try:
        cuts = _check_continuous(frames)
        timings["preprocessing_and_cut_check_seconds"] = time.perf_counter() - phase
        if cuts:
            frames.close()
            _save(
                output / "rejected-shot.json",
                {"reason": "possible_cut_or_flash", "events": cuts},
            )
            raise ValueError(
                f"Possible cut/flash at frame {cuts[0]['frame_index']}; choose a shorter continuous-shot interval"
            )
        reused = predictor is not None
        phase = time.perf_counter()
        if predictor is None:
            _progress(progress, "Starting video analysis", 0.18)
            predictor = build_predictor(settings)
        timings["sam_cold_load_seconds"] = (
            0.0 if reused else time.perf_counter() - phase
        )
        phase = time.perf_counter()
        cache = output / "tracking"
        cache.mkdir(exist_ok=True)
        records, session_id, native_error = {}, None, None
        try:
            _progress(progress, "Finding plate candidates", 0.25)
            session = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(clip),
                    "offload_video_to_cpu": True,
                }
            )
            session_id = session["session_id"]
            response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": "license plate",
                    "output_prob_thresh": settings.confidence_threshold,
                }
            )
            records[0] = _native_record(
                response, cache / "frame_000000", (size[1], size[0])
            )
            if count > 1:
                request = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": count,
                    "output_prob_thresh": settings.confidence_threshold,
                }
                for response in predictor.handle_stream_request(request):
                    index = int(response["frame_index"])
                    if not 0 <= index < count:
                        raise ValueError(
                            "Native tracking response escaped the selected clip"
                        )
                    records[index] = _native_record(
                        response, cache / f"frame_{index:06d}", (size[1], size[0])
                    )
                    _progress(
                        progress,
                        f"Tracking plates {index + 1}/{count}",
                        0.25 + 0.6 * (index + 1) / count,
                    )
        except Exception as error:  # noqa: BLE001 -- retain partial native evidence explicitly
            native_error = f"{type(error).__name__}: {error}"
            warnings.append(f"Native tracking incomplete: {native_error}")
        finally:
            if session_id is not None:
                predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
        timings["segmentation_seconds"] = time.perf_counter() - phase
        rows = [
            records.get(
                i, {"frame_index": i, "objects": [], "native_output_missing": True}
            )
            for i in range(count)
        ]
        missing = sum(bool(r.get("native_output_missing")) for r in rows)
        if missing:
            warnings.append(
                f"Native output missing for {missing}/{count} frames; no masks were invented"
            )
        report = {
            "kind": "sam31_text_native_single_shot",
            "resource_sha256": file_sha256(clip),
            "frame_count": count,
            "frames": rows,
            "native_error": native_error,
            "point_prompts": False,
            "predictor_reused": reused,
            "timings": timings,
            "calls": {"text_initialization": 1, "native_propagation": int(count > 1)},
            "continuous_shot_check": {
                "suspected_boundaries": [],
                "method": "RGB change + HSV histogram; heuristic not certification",
            },
        }
        _save(cache / "report.json", report)
        prepared = {
            "public": {"width": size[0], "height": size[1]},
            "_frames": frames,
            "_records": rows,
            "_mask_paths": [
                {
                    o["object_id"]: cache / f"frame_{i:06d}" / o["mask"]
                    for o in r["objects"]
                }
                for i, r in enumerate(rows)
            ],
            "_source_path": source,
            "_clip_path": clip,
            "_predictor": predictor,
            "_palette_scope": "video-" + report["resource_sha256"][:12],
            "_source_info": info,
            "_source_start": float(start),
            "_fps": float(rate),
            "_source_sha256": file_sha256(source),
            "_clip_sha256": report["resource_sha256"],
            "_cache_report": cache / "report.json",
            "warnings": warnings,
            "timings": timings,
        }
        phase = time.perf_counter()
        targets, representative, previews, default = _previews(prepared, output)
        if not targets:
            warnings.append("No visible native plate candidates were returned")
        timings["preview_seconds"] = time.perf_counter() - phase
        timings["prepare_seconds"] = time.perf_counter() - begun
        prepared["public"].update(
            source_preview_path=previews[0],
            mask_preview_path=previews[1],
            targets=targets,
            default_target_id=default,
            representative_frame_index=representative,
            fps=float(rate),
            duration=count / float(rate),
            frame_count=count,
            media_type="video",
            source_width=source_size[0],
            source_height=source_size[1],
            source_fps=float(source_rate),
            source_interval=[float(start), float(start) + count / float(rate)],
            warnings=list(warnings),
            timings=dict(timings),
            single_continuous_shot=True,
        )
        frames.close()
        _save(
            output / "prepare-report.json",
            {
                "public": prepared["public"],
                "source_sha256": prepared["_source_sha256"],
                "clip_sha256": prepared["_clip_sha256"],
                "tracking_report": str(prepared["_cache_report"]),
                "source_selection": {
                    "start": float(start),
                    "requested_duration": requested,
                    "effective_duration": duration,
                },
            },
        )
        _progress(progress, "Preview ready: choose a target", 1.0)
        return prepared
    except BaseException:
        frames.close()
        raise


def restore_video_prepared(prepare_dir, *, source_path, start, duration):
    """Reopen durable Stage 1 observations, without creating a SAM session.

    The source and lossless processing clip are content-addressed. Interval
    selection is checked against the persisted request and effective source
    interval; legacy caches use CFR frame quantization. Missing native responses are
    retained as missing, rather than synthesized during cache loading.
    """
    output, source = Path(prepare_dir).resolve(), Path(source_path).resolve()
    report = json.loads((output / "prepare-report.json").read_text())
    public = report["public"]
    if public.get("media_type") != "video":
        raise ValueError("Stage 1 cache is not a video")
    if file_sha256(source) != report["source_sha256"]:
        raise ValueError("Source changed; run Stage 1 again")
    clip = output / "prepared-video.mkv"
    if file_sha256(clip) != report["clip_sha256"]:
        raise ValueError("Stage 1 prepared clip changed; run Stage 1 again")
    if (
        not np.isfinite([start, duration]).all()
        or start < 0
        or not 0 < duration <= MAX_DURATION
    ):
        raise ValueError("Invalid source interval")
    info = probe(source)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    available = float(info.get("format", {}).get("duration", stream.get("duration", 0)))
    count, fps = int(public["frame_count"]), float(public["fps"])
    effective = min(float(duration), available - float(start))
    selection = report.get("source_selection")
    if selection is None:
        # Older caches retained only the CFR interval. FFmpeg rounds output
        # duration to the nearest frame (half up), rather than always up.
        interval_matches = max(1, math.floor(effective * fps + 0.5 + 1e-7)) == count
    else:
        interval_matches = (
            abs(float(start) - float(selection["start"])) <= 1e-7
            and abs(float(duration) - float(selection["requested_duration"])) <= 1e-7
            and abs(effective - float(selection["effective_duration"])) <= 1e-7
        )
    if (
        abs(float(start) - public["source_interval"][0]) > 1e-7
        or effective <= 0
        or not interval_matches
    ):
        raise ValueError("Source interval changed; run Stage 1 again")
    cache = output / "tracking"
    tracking = json.loads((cache / "report.json").read_text())
    if (
        tracking["resource_sha256"] != report["clip_sha256"]
        or tracking["frame_count"] != count
    ):
        raise ValueError("Stage 1 tracking does not match the prepared clip")
    rows = tracking["frames"]
    if len(rows) != count or any(r["frame_index"] != i for i, r in enumerate(rows)):
        raise ValueError("Stage 1 frame sequence is incomplete")
    paths = []
    for i, row in enumerate(rows):
        objects = {}
        for obj in row["objects"]:
            name = obj["mask"]
            if Path(name).name != name:
                raise ValueError("Invalid cached mask filename")
            path = cache / f"frame_{i:06d}" / name
            if not path.is_file():
                raise ValueError("Stage 1 mask is missing; run Stage 1 again")
            objects[int(obj["object_id"])] = path
        paths.append(objects)
    prepared = {
        "public": public,
        "_frames": _VideoFrames(clip, count),
        "_records": rows,
        "_mask_paths": paths,
        "_source_path": source,
        "_clip_path": clip,
        "_predictor": None,
        "_palette_scope": "video-" + report["clip_sha256"][:12],
        "_source_info": info,
        "_source_start": float(start),
        "_fps": fps,
        "_source_sha256": report["source_sha256"],
        "_clip_sha256": report["clip_sha256"],
        "_cache_report": cache / "report.json",
        "warnings": list(public.get("warnings", [])),
        "timings": dict(public["timings"]),
        "_stage1_restored": True,
    }
    # Decode dimensions/count are independently checked before reuse.
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    actual = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    if actual != (count, int(public["width"]), int(public["height"])):
        raise ValueError("Stage 1 clip dimensions or frame count changed")
    for i in range(count):
        _read_masks(prepared, i)
    return prepared


def _make_evidence(*args, **kwargs):
    from .evidence import build_scene_evidence

    return build_scene_evidence(*args, **kwargs)


def _candidate_data(candidate):
    if candidate is None:
        return None
    return {
        "quad": candidate.quad.tolist(),
        "confidence": float(candidate.confidence),
        "evidence": candidate.evidence,
        "uncertainties": candidate.uncertainties,
        "diagnostics": candidate.diagnostics,
    }


def _transport_advised_plane(frames, masks, seed, quad):
    """Motion evidence only; never propagate visibility or reuse an absolute quad."""
    quads = {seed: quad}
    diagnostics = []
    for direction in (-1, 1):
        previous_index, previous_quad = seed, quad
        for index in range(
            seed + direction, len(frames) if direction > 0 else -1, direction
        ):
            if not masks[index].any() or not masks[previous_index].any():
                break
            a = cv2.cvtColor(frames[previous_index], cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
            predicted, detail = flow_prediction(
                a, b, masks[previous_index], previous_quad, masks[index]
            )
            if predicted is None or detail.get("confidence", 0) < 0.3:
                break
            returned, reverse = flow_prediction(
                b, a, masks[index], predicted, masks[previous_index]
            )
            if returned is None or reverse.get("confidence", 0) < 0.3:
                break
            cycle = float(np.max(np.linalg.norm(returned - previous_quad, axis=1)))
            short = min(
                np.linalg.norm(previous_quad[(j + 1) % 4] - previous_quad[j])
                for j in range(4)
            )
            if cycle > max(2, short * 0.1):
                break
            quads[index] = predicted
            diagnostics.append(
                {
                    "frame_index": index,
                    "previous_index": previous_index,
                    "cycle_error_px": cycle,
                    "method": "native_visible_LK_forward_backward",
                }
            )
            previous_index, previous_quad = index, predicted
    return quads, diagnostics


def _writer(path, width, height, fps):
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
            "-s",
            f"{width}x{height}",
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
            "-threads",
            "1",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _finish(writer):
    writer.stdin.close()
    error = writer.stderr.read().decode(errors="replace")
    if writer.wait():
        raise RuntimeError(f"Video encoding failed: {error[-1500:]}")


def render_video(
    prepared,
    template_path,
    output_dir,
    *,
    target_id,
    mode="replace",
    advisor_profile=None,
    proxy_url=None,
    progress=None,
):
    """Render the selected native ID; other IDs blur. One advisor call maximum."""
    begun = time.perf_counter()
    if mode not in {"replace", "blur", "mosaic"}:
        raise ValueError("Mode must be replace, blur or mosaic")
    if advisor_profile is not None and advisor_profile not in PROFILES:
        raise ValueError("Choose a supported advisor profile")
    if file_sha256(prepared["_source_path"]) != prepared["_source_sha256"]:
        raise ValueError("Source video changed; prepare it again")
    target_id = int(target_id)
    target = next(
        (t for t in prepared["public"]["targets"] if int(t["id"]) == target_id), None
    )
    if target is None:
        raise ValueError("Choose a visible native target ID from preparation")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        output / name in {Path(prepared["_source_path"]), Path(prepared["_clip_path"])}
        for name in ("result.mp4", "diagnostic.mp4")
    ):
        raise ValueError("Rendering output must not overwrite source media")
    cv2.setNumThreads(1)
    if file_sha256(prepared["_clip_path"]) != prepared["_clip_sha256"]:
        raise ValueError("Prepared clip changed; prepare the video again")
    frames = prepared["_frames"]
    try:
        count, fps = len(frames), prepared["_fps"]
        shape = (prepared["public"]["height"], prepared["public"]["width"])
        warnings = list(prepared["warnings"])
        timings = {
            "sam_cold_load_seconds": prepared["timings"]["sam_cold_load_seconds"],
            "segmentation_seconds": prepared["timings"]["segmentation_seconds"],
            "prepare_seconds": prepared["timings"]["prepare_seconds"],
            "geometry_seconds": 0.0,
            "trajectory_seconds": 0.0,
            "ai_seconds": 0.0,
            "evidence_seconds": 0.0,
            "compositing_seconds": 0.0,
            "encoding_seconds": 0.0,
        }
        calls = {
            "advisor_attempted": 0,
            "advisor_succeeded": 0,
            "advisor_budget": 1,
            "automatic_second_review": False,
        }
        ai_record = {"status": "not_requested", "quality_accepted": False}
        all_masks = [_read_masks(prepared, i) for i in range(count)]
        target_masks = [m.get(target_id, np.zeros(shape, np.uint8)) for m in all_masks]
        template = load_template(Path(template_path)) if mode == "replace" else None
        proposals, selected = [[] for _ in range(count)], [None] * count
        appearance_ratios = {"exposure": 1.0, "tint": [1.0, 1.0, 1.0]}
        if mode == "replace":
            phase = time.perf_counter()
            aspect = template.alpha.shape[1] / template.alpha.shape[0]
            representative = int(target["representative_frame_index"])
            indices = sorted(
                set(
                    [representative]
                    + np.linspace(0, count - 1, min(5, count))
                    .round()
                    .astype(int)
                    .tolist()
                )
            )
            anchors = []
            for index in indices:
                if not target_masks[index].any():
                    continue
                candidate = select_scene_geometry(
                    propose_scene_geometry(
                        frames[index], target_masks[index], aspect=aspect
                    )
                )
                if candidate is not None and not candidate.diagnostics.get(
                    "corners_outside_viewport"
                ):
                    anchors.append(
                        {
                            "rgb": frames[index].copy(),
                            "mask": target_masks[index],
                            "quad": candidate.quad,
                            "frame_index": index,
                            "source": "automatic_same_native_identity",
                        }
                    )
            _progress(progress, "Estimating plate geometry", 0.08)
            for i in range(count):
                cv2.setRNGSeed(i)
                proposals[i] = propose_scene_geometry(
                    frames[i], target_masks[i], aspect=aspect, anchors=anchors
                )
            timings["geometry_seconds"] += time.perf_counter() - phase
            phase = time.perf_counter()
            selected = select_scene_trajectory(proposals)
            timings["trajectory_seconds"] += time.perf_counter() - phase
            if advisor_profile:
                automatic_proposals = [list(items) for items in proposals]
                phase = time.perf_counter()
                baseline, baseline_report, _ = solve_scene(
                    frames[representative],
                    target_masks[representative],
                    template,
                    selected_candidate=selected[representative],
                )
                extras = []
                for index in sorted(
                    {max(0, representative - 3), min(count - 1, representative + 3)}
                    - {representative}
                ):
                    path = output / f"advisor-neighbor-{index:06d}.png"
                    Image.fromarray(frames[index]).save(path)
                    extras.append(path)
                try:
                    evidence = _make_evidence(
                        frames[representative],
                        target_masks[representative],
                        template_path,
                        output / "evidence",
                        baseline_rgb=baseline,
                        scene_report=baseline_report,
                        target_id=target_id,
                        extra_images=extras,
                        context={
                            "media_type": "video",
                            "palette_scope": prepared["_palette_scope"],
                            "single_continuous_shot": True,
                            "representative_frame_index": representative,
                            "neighbor_frame_indices": [p.stem for p in extras],
                            "frame_count": count,
                            "fps": fps,
                            "target_native_id": target_id,
                            "instruction": "One shot-level call only. Geometry coordinates belong to the representative frame. Do not assume native IDs or masks are complete.",
                        },
                    )
                    timings["evidence_seconds"] = time.perf_counter() - phase
                    _progress(progress, "AI scene analysis", 0.3)
                    phase = time.perf_counter()
                    calls["advisor_attempted"] = 1
                    advice, ai_record = CodexSceneAdvisor(
                        profile=advisor_profile, proxy_url=proxy_url
                    ).run(
                        output / ("advisor-" + uuid.uuid4().hex[:12]),
                        evidence["prompt"],
                        evidence["image_paths"],
                    )
                    timings["ai_seconds"] = time.perf_counter() - phase
                    if advice is None:
                        warnings.append(
                            "AI proposal failed; automatic geometry retained. No retry or AI quality acceptance."
                        )
                    else:
                        validate_guidance(advice)
                        if not advice["plate_confirmed"]:
                            warnings.append(
                                "AI did not confirm this plate; automatic candidates retained for explicit review."
                            )
                        else:
                            calls["advisor_succeeded"] = 1
                            geometry = to_source_geometry(
                                advice,
                                evidence["crop_xyxy"],
                                source_size=[shape[1], shape[0]],
                            )
                            phase = time.perf_counter()
                            guided = propose_scene_geometry(
                                frames[representative],
                                target_masks[representative],
                                aspect=aspect,
                                guidance=geometry,
                                anchors=anchors,
                            )
                            guide_best = select_scene_geometry(guided)
                            if guide_best is not None:
                                transported, motion = _transport_advised_plane(
                                    frames,
                                    target_masks,
                                    representative,
                                    guide_best.quad,
                                )
                                ai_record["geometry_transport"] = motion
                                guided_anchors = [
                                    {
                                        "rgb": frames[representative].copy(),
                                        "mask": target_masks[representative],
                                        "quad": guide_best.quad,
                                        "frame_index": representative,
                                        "source": "one_call_advisor_representative",
                                    }
                                ] + anchors[:7]
                                for i in range(count):
                                    frame_guidance = (
                                        geometry
                                        if i == representative
                                        else {
                                            "source": "one_call_shot_prior_and_verified_motion",
                                            "physical_aspect_guess": geometry.get(
                                                "physical_aspect_guess"
                                            ),
                                            "assumptions": geometry.get(
                                                "assumptions", []
                                            ),
                                            "candidate_quads": [transported[i].tolist()]
                                            if i in transported
                                            else [],
                                        }
                                    )
                                    proposals[i] += propose_scene_geometry(
                                        frames[i],
                                        target_masks[i],
                                        aspect=aspect,
                                        guidance=frame_guidance,
                                        anchors=guided_anchors,
                                    )
                            else:
                                warnings.append(
                                    "AI geometry produced no eligible representative candidate; automatic geometry retained."
                                )
                            timings["geometry_seconds"] += time.perf_counter() - phase
                            automatic = baseline_report.get("compositor", {}).get(
                                "appearance", {}
                            )
                            raw_gain = automatic.get("exposure_gain_linear")
                            requested_gain = advice["appearance"].get(
                                "exposure_gain_linear"
                            )
                            if (
                                requested_gain is not None
                                and raw_gain is not None
                                and raw_gain > 1e-6
                            ):
                                ratio = requested_gain / raw_gain
                                appearance_ratios["exposure"] = float(
                                    np.clip(ratio, 0.25, 4)
                                )
                                if not 0.25 <= ratio <= 4:
                                    warnings.append(
                                        "AI relative exposure correction was bounded to 0.25..4x"
                                    )
                            requested_tint = advice["appearance"].get(
                                "diffuse_tint_rgb"
                            )
                            raw_tint = automatic.get("tint_rgb")
                            if requested_tint is not None and raw_tint is not None:
                                appearance_ratios["tint"] = np.clip(
                                    np.asarray(requested_tint)
                                    / np.maximum(raw_tint, 1e-4),
                                    0.5,
                                    2,
                                ).tolist()
                            ignored = [
                                key
                                for key in (
                                    "additive_rgb_linear",
                                    "flare_control_points",
                                    "shading_control_points",
                                )
                                if advice["appearance"].get(key) is not None
                            ]
                            ai_record["ignored_video_appearance_fields"] = ignored
                            ai_record["appearance_policy"] = (
                                "relative exposure/tint correction only; current frame spatial light remains automatic"
                            )
                            if ignored:
                                warnings.append(
                                    "Video AI spatial/additive fields were not reused across frames: "
                                    + ", ".join(ignored)
                                )
                except Exception as error:  # noqa: BLE001 -- advisory failure never becomes an AI success
                    if calls["advisor_attempted"] and not timings["ai_seconds"]:
                        timings["ai_seconds"] = time.perf_counter() - phase
                    proposals = automatic_proposals
                    appearance_ratios = {"exposure": 1.0, "tint": [1.0, 1.0, 1.0]}
                    calls["advisor_succeeded"] = 0
                    ai_record = {
                        **ai_record,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "quality_accepted": False,
                    }
                    warnings.append(
                        f"AI advisory failed; automatic candidates retained: {type(error).__name__}: {error}"
                    )
                phase = time.perf_counter()
                selected = select_scene_trajectory(proposals)
                timings["trajectory_seconds"] += time.perf_counter() - phase
        elif advisor_profile:
            warnings.append(
                "AI was not invoked for blur/mosaic; native masks are used directly"
            )
        temporal_geometry = {"method": "not_applicable"}
        if mode == "replace":
            phase = time.perf_counter()
            _progress(progress, "Stabilizing geometry with bidirectional motion", 0.46)
            selected, temporal_geometry = stabilize_scene_trajectory(
                frames, target_masks, selected, fps=fps
            )
            timings["trajectory_seconds"] += time.perf_counter() - phase
        visible_count = sum(bool(m.any()) for m in target_masks)
        if visible_count < count:
            warnings.append(
                f"Selected native ID visible in {visible_count}/{count} frames; absent masks remain absent, IDs are not merged"
            )
        state = TemporalVideoAppearance(fps)

        def filter_appearance(estimate, levels, **context):
            corrected = replace(
                estimate,
                gain=estimate.gain * appearance_ratios["exposure"],
                tint=np.asarray(estimate.tint) * appearance_ratios["tint"],
            )
            return state.filter_appearance(corrected, levels, **context)

        result_path, diagnostic_path = output / "result.mp4", output / "diagnostic.mp4"
        phase = time.perf_counter()
        writers = []
        rows, rendered_targets = [], 0
        try:
            result_writer = _writer(result_path, shape[1], shape[0], fps)
            writers.append(result_writer)
            diagnostic_writer = _writer(diagnostic_path, shape[1] * 3, shape[0], fps)
            writers.append(diagnostic_writer)
            timings["encoding_seconds"] += time.perf_counter() - phase
            for i in range(count):
                rgb, masks, mask = frames[i], all_masks[i], target_masks[i]
                phase = time.perf_counter()
                if mode == "replace":
                    if selected[i] is None or not mask.any():
                        state.reset()
                    rendered, detail, _ = solve_scene(
                        rgb,
                        mask,
                        template,
                        selected_candidate=selected[i],
                        appearance_estimate_filter=filter_appearance,
                        degradation_estimate_filter=state.filter_degradation,
                        noise_seed=i * 1000003 + (target_id + 1) * 9176,
                    )
                    applied = bool(detail["rendered"])
                else:
                    applied = bool(mask.any())
                    rendered = rgb.copy()
                    detail = {
                        "status": "rendered_unreviewed" if applied else "unresolved",
                        "rendered": applied,
                    }
                    if applied:
                        rendered, redaction = redact_surface(rendered, mask, mode)
                        detail["redaction"] = redaction
                rendered_targets += applied
                others = []
                for identity, other_mask in masks.items():
                    if identity == target_id:
                        continue
                    safe = ((other_mask > 0) & ~(mask > 0)).astype(np.uint8)
                    if safe.any():
                        rendered, _ = redact_surface(rendered, safe, "blur")
                        others.append(identity)
                middle, _ = color_native_masks(
                    rgb, masks, shot_id=prepared["_palette_scope"]
                )
                panel = np.concatenate([rgb, middle, rendered], axis=1)
                timings["compositing_seconds"] += time.perf_counter() - phase
                phase = time.perf_counter()
                result_writer.stdin.write(rendered.tobytes())
                diagnostic_writer.stdin.write(panel.tobytes())
                timings["encoding_seconds"] += time.perf_counter() - phase
                rows.append(
                    {
                        "frame_index": i,
                        "source_time_seconds": prepared["_source_start"] + i / fps,
                        "native_ids": sorted(masks),
                        "target_visible_pixels": int(mask.sum()),
                        "other_blurred_ids": others,
                        "scene": detail,
                        "candidate_count": len(proposals[i]),
                        "selected_geometry": _candidate_data(selected[i]),
                    }
                )
                _progress(
                    progress,
                    f"Processing video {i + 1}/{count}",
                    0.5 + 0.43 * (i + 1) / count,
                )
            phase = time.perf_counter()
            _finish(result_writer)
            _finish(diagnostic_writer)
            timings["encoding_seconds"] += time.perf_counter() - phase
        finally:
            for writer in writers:
                if writer.poll() is None:
                    writer.kill()
                    writer.wait()
                for pipe in (writer.stdin, writer.stderr):
                    if pipe is not None:
                        pipe.close()
        phase = time.perf_counter()
        verification = {
            "output": verify_encoded_output(result_path, count),
            "diagnostic": verify_encoded_output(diagnostic_path, count),
        }
        # The selected video interval is preserved; first-source audio is reencoded
        # explicitly to AAC for browser compatibility, never claimed packet-identical.
        audio = any(
            s.get("codec_type") == "audio" for s in prepared["_source_info"]["streams"]
        )
        if audio:
            for path in (result_path, diagnostic_path):
                temp = path.with_name(path.stem + "-audio.mp4")
                command(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        path,
                        "-ss",
                        str(prepared["_source_start"]),
                        "-t",
                        str(count / fps),
                        "-i",
                        prepared["_source_path"],
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        temp,
                    ]
                )
                verify_encoded_output(temp, count)
                temp.replace(path)
            warnings.append(
                "First source audio stream reencoded to AAC for browser playback; not packet-identical"
            )
        timings["encoding_seconds"] += time.perf_counter() - phase
        if mode == "replace" and rendered_targets < visible_count:
            warnings.append(
                f"Scene geometry/compositing unresolved on {visible_count - rendered_targets} visible target frames"
            )
        timings["render_seconds"] = time.perf_counter() - begun
        timings["pipeline_seconds_excluding_user_wait"] = (
            timings["prepare_seconds"] + timings["render_seconds"]
        )
        report = {
            "media_type": "video",
            "mode": mode,
            "target_id": target_id,
            "single_continuous_shot": True,
            "frame_count": count,
            "fps": fps,
            "duration": count / fps,
            "processing_size": list(shape[::-1]),
            "source_sha256": prepared["_source_sha256"],
            "prepared_clip_sha256": prepared["_clip_sha256"],
            "tracking_report": str(prepared["_cache_report"]),
            "source_interval": prepared["public"]["source_interval"],
            "timings": timings,
            "timing_policy": "Wall-clock stages; encoding includes writer I/O/wait and verification, not exclusive encoder CPU time",
            "calls": calls,
            "advisor": ai_record,
            "appearance_correction_ratios": appearance_ratios,
            "temporal_geometry": temporal_geometry,
            "automatic_quality_review_run": False,
            "quality_accepted": False,
            "rendered_target_frames": rendered_targets,
            "native_visible_target_frames": visible_count,
            "warnings": list(dict.fromkeys(warnings)),
            "encoded_verification": verification,
            "frames": rows,
        }
        report["status"] = (
            "needs_review"
            if report["warnings"]
            or any(r["scene"]["status"] != "rendered_unreviewed" for r in rows)
            else "rendered_unreviewed"
        )
        report_path = output / "report.json"
        _save(report_path, report)
        if mode == "replace" and rendered_targets == 0:
            raise RuntimeError(
                f"Replacement unresolved on every target frame; no successful original-frame result. Details: {report_path}"
            )
        _progress(progress, "Video and diagnostic previews are ready", 1.0)
        return {
            "output_path": str(result_path),
            "diagnostic_path": str(diagnostic_path),
            "report_path": str(report_path),
            "timings": timings,
            "warnings": report["warnings"],
            "media_type": "video",
            "status": report["status"],
            "calls": calls,
        }
    finally:
        frames.close()
