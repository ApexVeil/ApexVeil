"""Single-worker, loopback-only studio. SSH forwarding is outside this module."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import statistics
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlsplit

from apexveil.backends.codex_advisor import (
    EFFORTS,
    MODEL_OPTIONS,
    PROFILES,
    advisor_profile,
    profile_selection,
)
from apexveil.config import load_settings

ID = re.compile(r"^[0-9a-f]{32}$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def contained_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError("File not found")
    return path


@dataclass
class StudioConfig:
    settings_path: Path
    output_root: Path
    default_template: Path | None = None
    advisor_profile: str = "sol-medium"
    proxy_url: str | None = None
    max_video_seconds: float = 8.0
    max_upload_mb: int = 512


@dataclass(frozen=True)
class RenderRequest:
    revision: int
    render_id: str
    target_id: int
    mode: str
    ai_assist: bool
    ai_model: str
    ai_effort: str
    protect_foreground: bool
    template_id: str | None
    template_name: str | None
    template_path: Path | None
    proxy_url: str | None

    def public(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if k not in {"template_path", "proxy_url"}
        }


def source_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class CachedForeground:
    """Replay Stage 1 person detections; Stage 2 must never invoke SAM."""

    def __init__(self, folder, metadata=None):
        self.folder = Path(folder)
        self.metadata = metadata or {}

    def detect(self, image, *, prompt, refine=True):
        import numpy as np
        from PIL import Image

        cache = self.folder / "foreground-cache.json"
        if prompt != "person" or not cache.is_file():
            raise ValueError("人物遮挡缓存缺失，请重新执行阶段 1。")
        data = json.loads(cache.read_text())
        items = []
        for record in data["instances"]:
            with Image.open(self.folder / record["mask"]) as mask_image:
                mask = (np.asarray(mask_image.convert("L")) > 0).astype(np.uint8)
            if mask.shape != (image.height, image.width):
                raise ValueError("人物遮挡缓存尺寸不一致，请重新执行阶段 1。")
            items.append(
                SimpleNamespace(
                    **{k: v for k, v in record.items() if k != "mask"},
                    visible_mask=mask,
                )
            )
        return items, data.get("detection", {})


def cache_foreground(prepared, folder, segmenter, progress):
    import numpy as np
    from PIL import Image

    from apexveil.workflows.evidence import json_value

    progress("阶段 1 · 识别并缓存人物遮挡", 0.92)
    begun = time.perf_counter()
    detections, metadata = segmenter.detect(
        Image.fromarray(prepared["private"]["rgb"]), prompt="person", refine=True
    )
    directory = folder / "native-person-masks"
    directory.mkdir(exist_ok=True)
    records = []
    for item in detections:
        mask = np.asarray(item.visible_mask)
        if mask.shape != prepared["private"]["rgb"].shape[:2]:
            raise ValueError("人物遮罩尺寸与源素材不一致。")
        path = directory / f"person-{int(item.object_id)}.png"
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
        records.append(
            {
                "object_id": int(item.object_id),
                "score": float(item.score),
                "mask": path.relative_to(folder).as_posix(),
                "diagnostics": json_value(getattr(item, "diagnostics", {})),
            }
        )
    (folder / "foreground-cache.json").write_text(
        json.dumps(
            {"instances": records, "detection": json_value(metadata)},
            default=json_default,
        )
    )
    prepared["timings"]["foreground_detection_seconds"] = time.perf_counter() - begun
    prepared["private"]["segmenter"] = CachedForeground(
        folder, getattr(segmenter, "metadata", {})
    )


class Studio:
    def __init__(self, config, *, segmenter_factory=None):
        self.config = config
        self.settings = load_settings(config.settings_path)
        self.root = config.output_root.expanduser().resolve()
        (self.root / "assets").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.assets = {}
        self.jobs = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="apexveil-gpu"
        )
        self.segmenter = None
        self.segmenter_factory = segmenter_factory
        # Restore assets and finished outputs. Running GPU state cannot survive a restart.
        for file in (self.root / "assets").iterdir():
            if (
                ID.fullmatch(file.stem)
                and file.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
            ):
                kind = "image" if file.suffix.lower() in IMAGE_EXTENSIONS else "video"
                self.assets[file.stem] = {
                    "asset_id": file.stem,
                    "name": file.name,
                    "kind": kind,
                    "_path": file,
                }
        for metadata in (self.root / "assets").glob("*.json"):
            try:
                row = json.loads(metadata.read_text())
                if row.get("asset_id") in self.assets:
                    self.assets[row["asset_id"]]["name"] = str(
                        row.get("name", metadata.stem)
                    )
            except (ValueError, OSError):
                continue
        for saved in (self.root / "jobs").glob("*/job.json"):
            try:
                row = json.loads(saved.read_text())
                if not ID.fullmatch(str(row.get("id", ""))):
                    continue
                row["_directory"] = saved.parent
                report = saved.parent / "prepare/prepare-report.json"
                ready = (
                    row.get("media_id") in self.assets
                    and report.is_file()
                    and bool((row.get("public") or {}).get("targets"))
                )
                row.update(
                    _media=self.assets.get(row.get("media_id")),
                    _template=self.assets.get(row.get("template_id"), {}).get(
                        "_path", config.default_template
                    ),
                    _model_load_seconds=0.0,
                    _prepare_wall=float(row.get("stage1", {}).get("wall_seconds", 0)),
                    _queue_seconds=0.0,
                )
                if ready:
                    row["_resume_selection"] = True
                    row.setdefault("stage1", {"state": "ready", "restored": True})
                    if row["state"] not in {
                        "completed",
                        "failed",
                        "cancelled",
                        "awaiting_selection",
                    }:
                        row.update(
                            state="awaiting_selection",
                            phase="prepare",
                            message="阶段 1 缓存已恢复，可重新生成阶段 2",
                            error=None,
                        )
                        for revision in row.get("renders", []):
                            if revision.get("state") in {"queued", "rendering"}:
                                revision.update(
                                    state="interrupted",
                                    error="服务重启，中断的合成未自动重试。",
                                )
                elif row["state"] not in {"completed", "failed", "cancelled"}:
                    row.update(
                        state="failed",
                        error="阶段 1 缓存不完整，请重新识别素材。",
                        message="需要重新执行阶段 1",
                    )
                if not ready:
                    row["stage1"] = {"state": "missing"}
                self.jobs[row["id"]] = row
            except (ValueError, KeyError, OSError):
                continue

    def file_url(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Output is outside this studio's data root")
        return "/files/" + quote(path.relative_to(self.root).as_posix(), safe="/")

    def public_paths(self, value):
        if isinstance(value, list):
            return [self.public_paths(v) for v in value]
        if not isinstance(value, dict):
            return value
        return {
            key[:-5] + "_url" if key.endswith("_path") else key: self.file_url(item)
            if key.endswith("_path") and item is not None
            else self.public_paths(item)
            for key, item in value.items()
        }

    def update(self, job, **fields):
        with self.lock:
            if fields.get("state") in {
                "awaiting_selection",
                "completed",
                "failed",
                "cancelled",
            } and job.get("_phase_started"):
                fields["phase_elapsed_seconds"] = (
                    time.perf_counter() - job["_phase_started"]
                )
            job.update(fields)
            public = {k: v for k, v in job.items() if not k.startswith("_")}
            temporary = job["_directory"] / "job.json.tmp"
            temporary.write_text(
                json.dumps(public, ensure_ascii=False, indent=2, default=json_default)
            )
            temporary.replace(job["_directory"] / "job.json")

    def selected_advisor(self, data, previous=None):
        previous = previous or {}
        fallback = (
            self.config.advisor_profile
            if self.config.advisor_profile != "off"
            else "sol-medium"
        )
        default_model, default_effort = profile_selection(fallback)
        model = data.get("ai_model", previous.get("ai_model", default_model))
        effort = data.get("ai_effort", previous.get("ai_effort", default_effort))
        return model, effort, advisor_profile(model, effort)

    def expected_seconds(self, job, phase):
        """Presentation estimate, never a renderer-completion signal."""
        samples = []
        with self.lock:
            history = list(self.jobs.values())
        for old in history:
            if old.get("state") != "completed" or old.get("kind") != job["kind"]:
                continue
            if old.get("mode") != job["mode"]:
                continue
            if phase == "render" and bool(old.get("ai_assist")) != bool(
                job.get("ai_assist")
            ):
                continue
            if (
                phase == "render"
                and job.get("ai_assist")
                and (
                    old.get("ai_model", "gpt-6.1-sol") != job.get("ai_model")
                    or old.get("ai_effort", "medium") != job.get("ai_effort")
                )
            ):
                continue
            value = (
                old.get("result", {}).get("timings", {}).get(f"{phase}_wall_seconds")
            )
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                if job["kind"] == "video":
                    value *= job.get("duration", 2) / max(0.1, old.get("duration", 2))
                samples.append(value)
        if samples:
            return round(
                max(2, statistics.median(samples[-16:]))
                + (18 if phase == "prepare" and self.segmenter is None else 0),
                2,
            )
        if phase == "prepare":
            return (
                20
                if self.segmenter is None
                else (2 if job["kind"] == "image" else 6 * job.get("duration", 2))
            )
        base = 8 if job["kind"] == "image" else 17 * job.get("duration", 2)
        if job.get("ai_assist"):
            base += {"gpt-6-luna": 15, "gpt-6.1-sol": 45, "gpt-6-astra": 65}.get(
                job.get("ai_model"), 45
            ) * {"low": 0.7, "medium": 1, "high": 1.6, "xhigh": 2.3, "max": 3.2}[
                job.get("ai_effort", "medium")
            ]
        return base

    def snapshot(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("Job not found; restart creates a new session")
            job = self.jobs[job_id]
            public = {k: v for k, v in job.items() if not k.startswith("_")}
            public["phase_elapsed_seconds"] = (
                max(0, time.perf_counter() - job["_phase_started"])
                if job.get("_phase_started")
                and job.get("state") in {"preparing", "rendering"}
                else job.get("phase_elapsed_seconds", 0)
            )
            return copy.deepcopy(public)

    def progress(self, job):
        def callback(message, fraction=None):
            self.update(job, message=str(message), progress=fraction)

        return callback

    def register_asset(self, path, name, kind, asset_id):
        record = {"asset_id": asset_id, "name": name, "kind": kind, "_path": path}
        (self.root / "assets" / (asset_id + ".json")).write_text(
            json.dumps(
                {"asset_id": asset_id, "name": name, "kind": kind}, ensure_ascii=False
            )
        )
        with self.lock:
            self.assets[asset_id] = record
        return {k: v for k, v in record.items() if not k.startswith("_")}

    def create_job(self, data):
        with self.lock:
            old = self.jobs.get(data.get("supersedes_id"))
            old = (
                old
                if old is not None and old["state"] == "awaiting_selection"
                else None
            )

            if (
                sum(
                    j["state"] not in {"completed", "failed", "cancelled"}
                    and j is not old
                    for j in self.jobs.values()
                )
                >= 4
            ):
                raise ValueError("最多保留 4 个未完成任务，请先完成已有任务。")
            media = self.assets.get(data.get("media_id"))
            if media is None:
                raise ValueError("请先上传图片或视频。")
            mode = data.get("mode", "replace")
            if mode not in {"replace", "blur", "mosaic"}:
                raise ValueError("Unsupported treatment")
            template = self.config.default_template
            if data.get("template_id"):
                asset = self.assets.get(data["template_id"])
                if asset is None or asset["kind"] != "image":
                    raise ValueError("模板必须是已上传的图片。")
                template = asset["_path"]
            ai = data.get("ai_assist", False)
            protection = data.get("protect_foreground", True)
            if not isinstance(ai, bool) or not isinstance(protection, bool):
                raise TypeError("Invalid toggle value")
            ai_model, ai_effort, _ = self.selected_advisor(data)
            start, duration = (
                float(data.get("start", 0)),
                float(data.get("duration", 2)),
            )
            if not math.isfinite(start) or start < 0:
                raise ValueError("视频起点需为非负秒数。")
            if (
                not math.isfinite(duration)
                or not 0 < duration <= self.config.max_video_seconds
            ):
                raise ValueError(
                    f"试用片段需在 0–{self.config.max_video_seconds:g} 秒之间。"
                )
            if old is not None:
                self.release_prepared(old)
                self.update(
                    old, state="cancelled", message="已被新设置替代", progress=None
                )
            key = uuid.uuid4().hex
            directory = self.root / "jobs" / key
            directory.mkdir()
            job = {
                "id": key,
                "state": "queued",
                "progress": None,
                "message": "等待处理",
                "kind": media["kind"],
                "media_id": media["asset_id"],
                "media_name": media["name"],
                "template_id": data.get("template_id"),
                "template_name": self.assets.get(data.get("template_id"), {}).get(
                    "name"
                ),
                "ai_assist": ai,
                "ai_model": ai_model,
                "ai_effort": ai_effort,
                "phase": "prepare",
                "stage1": {"state": "pending"},
                "renders": [],
                "render_revision": 0,
                "protect_foreground": protection,
                "start": start,
                "duration": duration,
                "mode": mode,
                "public": None,
                "result": None,
                "error": None,
                "_directory": directory,
                "_media": media,
                "_template": template,
                "_ai": ai and mode == "replace",
                "_protection": protection,
                "_start": start,
                "_duration": duration,
                "_submitted": time.perf_counter(),
            }
            self.jobs[key] = job
            self.update(job)
            self.executor.submit(self.prepare, job)
            return {"id": key}

    def fail(self, job, error):
        (job["_directory"] / "error.log").write_text(traceback.format_exc())
        self.update(
            job, state="failed", message="处理未完成", error=str(error), progress=None
        )
        self.release_prepared(job)

    @staticmethod
    def release_prepared(job):
        prepared = job.pop("_prepared", None)
        if isinstance(prepared, dict):
            frames = prepared.get("_frames")
            if hasattr(frames, "close"):
                frames.close()

    def prepare(self, job):
        started = time.perf_counter()
        job["_queue_seconds"] = started - job["_submitted"]
        try:
            callback = self.progress(job)
            self.update(
                job,
                state="preparing",
                message="阶段 1 · 载入模型并识别车牌",
                progress=None,
            )
            job["_phase_started"] = time.perf_counter()
            self.update(
                job,
                phase="prepare",
                estimated_phase_seconds=self.expected_seconds(job, "prepare"),
            )
            load_started = time.perf_counter()
            if self.segmenter is None:
                if self.segmenter_factory is None:
                    from apexveil.backends.image import ImageSegmenter

                    self.segmenter = ImageSegmenter(self.settings)
                else:
                    self.segmenter = self.segmenter_factory(self.settings)
                job["_model_load_seconds"] = time.perf_counter() - load_started
            else:
                job["_model_load_seconds"] = 0.0
            self.update(job, stage1={"state": "running"})
            output = job["_directory"] / "prepare"
            if job["kind"] == "image":
                from apexveil.workflows.image import prepare_image

                prepared = prepare_image(
                    job["_media"]["_path"], output, self.segmenter, progress=callback
                )
            else:
                from apexveil.workflows.video import prepare_video

                prepared = prepare_video(
                    job["_media"]["_path"],
                    output,
                    self.settings,
                    predictor=self.segmenter.predictor,
                    start=job["_start"],
                    duration=job["_duration"],
                    progress=callback,
                )
            if job["kind"] == "image":
                cache_foreground(prepared, output, self.segmenter, callback)
            if not prepared["public"]["targets"]:
                raise ValueError("未识别到可选择的车牌，请换一张图片或调整视频片段。")
            job["_prepared"] = prepared
            job["_prepare_wall"] = time.perf_counter() - started
            self.update(
                job,
                state="awaiting_selection",
                progress=1,
                message="阶段 1 已缓存，选择目标并生成阶段 2",
                stage1={
                    "state": "ready",
                    "cache_version": 1,
                    "media_id": job["media_id"],
                    "start": job["start"],
                    "duration": job["duration"],
                    "source_sha256": source_sha256(job["_media"]["_path"]),
                    "wall_seconds": job["_prepare_wall"],
                },
                public=self.public_paths(prepared["public"]),
            )
        except Exception as error:  # noqa: BLE001 -- isolate worker failures and preserve diagnostics
            self.fail(job, error)

    def verify_stage1(self, job, data):
        """A render may alter treatment settings, never the Stage 1 source interval."""
        for key in ("media_id", "start", "duration"):
            if key in data and data[key] != job.get(key):
                raise ValueError("素材或片段已变化，请重新执行阶段 1。")
        if job.get("stage1", {}).get("state") in {"invalid", "missing"}:
            raise ValueError("阶段 1 缓存不可用，请重新识别。")
        expected = job.get("stage1", {}).get("source_sha256")
        if expected and source_sha256(job["_media"]["_path"]) != expected:
            self.update(job, stage1={**job["stage1"], "state": "invalid"})
            raise ValueError("源素材已变化，请重新执行阶段 1。")

    def request_render(self, job_id, data):
        with self.lock:
            job = self.jobs[job_id]
            if job["state"] not in {"awaiting_selection", "completed", "failed"}:
                raise ValueError("当前阶段仍在运行，请等待完成。")
            if not (job.get("public") or {}).get("targets"):
                raise ValueError("请先完成阶段 1 的识别。")
            self.verify_stage1(job, data)
            target = data.get("target_id")
            if (
                not isinstance(target, int)
                or isinstance(target, bool)
                or target not in [t["id"] for t in job["public"]["targets"]]
            ):
                raise ValueError("请选择一个实际识别到的目标。")
            mode = data.get("mode", job["mode"])
            if mode not in {"replace", "blur", "mosaic"}:
                raise ValueError("Unsupported treatment")
            template_id = data.get("template_id", job.get("template_id"))
            template = self.config.default_template
            template_name = None
            if template_id:
                asset = self.assets.get(template_id)
                if asset is None or asset["kind"] != "image":
                    raise ValueError("模板必须是已上传的图片。")
                template, template_name = asset["_path"], asset["name"]
            if mode == "replace" and (template is None or not Path(template).is_file()):
                raise ValueError("请上传透明车牌模板。")
            ai = data.get("ai_assist", job.get("ai_assist", False))
            protection = data.get(
                "protect_foreground", job.get("protect_foreground", True)
            )
            if not isinstance(ai, bool) or not isinstance(protection, bool):
                raise TypeError("Invalid toggle value")
            if ai and mode == "replace" and self.config.advisor_profile == "off":
                raise ValueError("当前服务器未启用 AI 辅助。")
            model, effort, _ = self.selected_advisor(data, job)
            frozen = RenderRequest(
                revision=int(job.get("render_revision", 0)) + 1,
                render_id=uuid.uuid4().hex,
                target_id=target,
                mode=mode,
                ai_assist=ai and mode == "replace",
                ai_model=model,
                ai_effort=effort,
                protect_foreground=protection,
                template_id=template_id,
                template_name=template_name,
                template_path=Path(template) if template else None,
                proxy_url=self.config.proxy_url,
            )
            previous_mode = job["mode"]
            job.update(
                **{
                    k: v
                    for k, v in frozen.public().items()
                    if k not in {"revision", "render_id", "target_id"}
                }
            )
            job["_render_config"] = frozen
            job["_render_requested"] = time.perf_counter()
            job.pop("_phase_started", None)
            if job.get("result") and not job.get("renders"):
                job["renders"] = [
                    {
                        "revision": 0,
                        "state": "completed",
                        "mode": previous_mode,
                        "result": job["result"],
                    }
                ]
            job.setdefault("renders", []).append(
                {**frozen.public(), "state": "queued", "result": None}
            )
            self.update(
                job,
                state="queued",
                phase="render",
                render_revision=frozen.revision,
                phase_elapsed_seconds=0,
                selected_id=target,
                error=None,
                message="阶段 2 · 等待合成（复用阶段 1 缓存）",
                progress=None,
            )
            self.executor.submit(self.render, job, target, frozen)
            return {"id": job_id, "render_revision": frozen.revision}

    def restore_image_selection(self, job):
        """Restore verified source + native image masks without repeating detection."""
        import numpy as np
        from PIL import Image, ImageOps

        folder = job["_directory"] / "prepare"
        metadata = json.loads((folder / "prepare-report.json").read_text())
        source = job["_media"]["_path"]
        digest = source_sha256(source)
        if digest != metadata["source_sha256"]:
            raise ValueError("素材已变化，请重新识别。")
        with Image.open(source) as image:
            icc = image.info.get("icc_profile")
            rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        masks = {}
        for target in job["public"]["targets"]:
            oid = int(target["id"])
            mask = (
                np.asarray(
                    Image.open(folder / "native-masks" / f"plate-{oid}.png").convert(
                        "L"
                    )
                )
                > 0
            )
            if mask.shape != rgb.shape[:2]:
                raise ValueError("缓存尺寸不一致，请重新识别。")
            masks[oid] = mask.astype(np.uint8)
        job["_prepared"] = {
            "public": job["public"],
            "private": {
                "rgb": rgb,
                "masks": masks,
                "segmenter": CachedForeground(
                    folder, metadata.get("model_metadata", {})
                ),
                "detection_metadata": metadata.get("detection", {}),
                "model_metadata": metadata.get("model_metadata", {}),
                "source_name": source.name,
                "source_sha256": digest,
                "icc_profile": icc,
            },
            "timings": {"detect_seconds": 0.0, "segmentation_cache_reused": True},
            "warnings": [],
        }
        job.pop("_resume_selection", None)

    def restore_prepared(self, job):
        if job["kind"] == "image":
            self.restore_image_selection(job)
        else:
            from apexveil.workflows.video import restore_video_prepared

            job["_prepared"] = restore_video_prepared(
                job["_directory"] / "prepare",
                source_path=job["_media"]["_path"],
                start=job["start"],
                duration=job["duration"],
            )
            job.pop("_resume_selection", None)

    def render(self, job, target, frozen=None):
        revision = None
        started = time.perf_counter()
        try:
            frozen = frozen or job["_render_config"]
            with self.lock:
                revision = next(
                    r for r in job["renders"] if r.get("render_id") == frozen.render_id
                )
                revision["state"] = "rendering"
                job["_phase_started"] = started
            self.update(
                job,
                state="rendering",
                phase="render",
                estimated_phase_seconds=self.expected_seconds(job, "render"),
                message="阶段 2 · 复用识别缓存进行场景匹配与合成",
                progress=None,
            )
            self.verify_stage1(job, {})
            if "_prepared" not in job:
                self.restore_prepared(job)
            # Pre-stage caches did not persist person masks. Upgrade them once,
            # keeping the original plate detections, before entering Stage 2.
            if (
                job["kind"] == "image"
                and frozen.mode == "replace"
                and frozen.protect_foreground
                and job.get("stage1", {}).get("cache_version", 0) < 1
                and (job["_directory"] / "prepare/prepare-report.json").is_file()
                and not (job["_directory"] / "prepare/foreground-cache.json").is_file()
            ):
                self.update(
                    job,
                    state="preparing",
                    phase="prepare",
                    message="阶段 1 · 补全旧缓存的人物遮挡，保留原有车牌识别",
                )
                if self.segmenter is None:
                    if self.segmenter_factory is None:
                        from apexveil.backends.image import ImageSegmenter

                        self.segmenter = ImageSegmenter(self.settings)
                    else:
                        self.segmenter = self.segmenter_factory(self.settings)
                cache_foreground(
                    job["_prepared"],
                    job["_directory"] / "prepare",
                    self.segmenter,
                    self.progress(job),
                )
                self.update(
                    job,
                    stage1={
                        **job.get("stage1", {}),
                        "state": "ready",
                        "cache_version": 1,
                        "foreground_upgraded": True,
                    },
                )
                self.update(
                    job,
                    state="rendering",
                    phase="render",
                    message="阶段 2 · 复用识别缓存进行场景匹配与合成",
                    progress=None,
                )
            common = {
                "target_id": frozen.target_id,
                "mode": frozen.mode,
                "advisor_profile": advisor_profile(frozen.ai_model, frozen.ai_effort)
                if frozen.ai_assist
                else None,
                "proxy_url": frozen.proxy_url,
                "progress": self.progress(job),
            }
            output = (
                job["_directory"]
                / "renders"
                / f"{frozen.revision:04d}-{frozen.render_id[:8]}"
            )
            output.parent.mkdir(exist_ok=True)
            if job["kind"] == "image":
                from apexveil.workflows.image import render_image

                result = render_image(
                    job["_prepared"],
                    frozen.template_path,
                    output,
                    protect_foreground=frozen.protect_foreground,
                    **common,
                )
            else:
                from apexveil.workflows.video import render_video

                result = render_video(
                    job["_prepared"], frozen.template_path, output, **common
                )
            elapsed = time.perf_counter() - started
            timings = {
                **result.get("timings", {}),
                "model_load_seconds_this_job": job["_model_load_seconds"],
                "prepare_wall_seconds": job["_prepare_wall"],
                "render_wall_seconds": elapsed,
                "processing_wall_seconds": (
                    job["_prepare_wall"] if frozen.revision == 1 else 0.0
                )
                + elapsed,
                "stage1_cached_wall_seconds": job["_prepare_wall"],
                "stage2_wall_seconds": elapsed,
                "queue_seconds": job["_queue_seconds"]
                + started
                - job["_render_requested"],
            }
            result["timings"] = timings
            result["ai_selection"] = {
                "enabled": frozen.ai_assist,
                "model": frozen.ai_model,
                "effort": frozen.ai_effort,
            }
            result["render_config"] = frozen.public()
            result["stage1_reused"] = True
            report_path = result.get("report_path")
            if report_path and Path(report_path).is_file():
                report = json.loads(Path(report_path).read_text())
                report["gui_timings"] = timings
                report["ai_selection"] = result["ai_selection"]
                report["render_config"] = frozen.public()
                report["stage1_reused"] = True
                Path(report_path).write_text(
                    json.dumps(
                        report, ensure_ascii=False, indent=2, default=json_default
                    )
                )
            with self.lock:
                revision.update(state="completed", result=self.public_paths(result))
            self.update(
                job,
                state="completed",
                message="处理完成，请检查效果后下载",
                progress=1,
                result=self.public_paths(result),
            )
        except Exception as error:  # noqa: BLE001 -- isolate worker failures and preserve diagnostics
            if revision is not None:
                with self.lock:
                    revision.update(state="failed", error=str(error))
            self.fail(job, error)
        finally:
            self.release_prepared(job)
            job["_model_load_seconds"] = 0.0

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for job in self.jobs.values():
            self.release_prepared(job)


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, studio):
        self.studio = studio
        super().__init__(address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "ApexVeilStudio/0.1"

    def valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def json_response(self, value, code=200):
        body = json.dumps(value, ensure_ascii=False, default=json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 16384:
            raise ValueError("Invalid request length")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise TypeError("Request must be an object")
        return value

    def do_GET(self):
        if not self.valid_host():
            self.json_response({"error": "Local studio host required"}, 403)
            return
        path = urlsplit(self.path).path
        studio = self.server.studio
        try:
            if path == "/api/config":
                self.json_response(
                    {
                        "default_template": bool(
                            studio.config.default_template
                            and Path(studio.config.default_template).is_file()
                        ),
                        "max_video_seconds": studio.config.max_video_seconds,
                        "max_upload_mb": studio.config.max_upload_mb,
                        "ai_models": [
                            {
                                "id": m["id"],
                                "label": m["label"],
                                "efforts": list(EFFORTS),
                            }
                            for m in MODEL_OPTIONS
                        ],
                        "ai_defaults": dict(
                            zip(
                                ("model", "effort"),
                                profile_selection(
                                    studio.config.advisor_profile
                                    if studio.config.advisor_profile != "off"
                                    else "sol-medium"
                                ),
                            )
                        ),
                        "ai_available": studio.config.advisor_profile != "off"
                        and shutil.which("codex") is not None,
                    }
                )
            elif path in {"/progress.js", "/interactions.js"}:
                self.serve_file(Path(__file__).with_name(path[1:]))
            elif path == "/api/template":
                source = studio.config.default_template
                if source is None or not Path(source).is_file():
                    raise FileNotFoundError("No default template")
                from PIL import Image

                preview = studio.root / "default-template-preview.png"
                with Image.open(source) as original:
                    rgba = original.convert("RGBA")
                    bounds = rgba.getchannel("A").getbbox()
                    if bounds:
                        rgba = rgba.crop(bounds)
                    rgba.thumbnail((600, 180), Image.Resampling.LANCZOS)
                    rgba.save(preview)
                self.serve_file(preview)
            elif path.startswith("/api/jobs/") and ID.fullmatch(path[10:]):
                self.json_response(studio.snapshot(path[10:]))
            elif path in {"/", "/index.html"}:
                self.serve_file(Path(__file__).with_name("index.html"))
            elif path.startswith("/files/"):
                self.serve_file(contained_file(studio.root, unquote(path[7:])))
            elif path.startswith("/api/assets/") and ID.fullmatch(path[12:]):
                self.serve_file(studio.assets[path[12:]]["_path"])
            else:
                self.json_response({"error": "Not found"}, 404)
        except (FileNotFoundError, KeyError):
            self.json_response({"error": "Not found"}, 404)
        except (ValueError, TypeError) as error:
            self.json_response({"error": str(error)}, 400)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.valid_host() or (
            origin and origin != "http://" + self.headers.get("Host", "")
        ):
            self.json_response(
                {"error": "Same-origin local studio request required"}, 403
            )
            return
        parsed = urlsplit(self.path)
        studio = self.server.studio
        try:
            if parsed.path == "/api/assets":
                name = parse_qs(parsed.query).get("name", [""])[0]
                suffix = Path(name).suffix.lower()
                if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                    raise ValueError("请上传 JPG、PNG 等图片或 MP4、MOV 等视频。")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= studio.config.max_upload_mb * 1024 * 1024:
                    raise ValueError(f"文件需小于 {studio.config.max_upload_mb} MB。")
                asset_id = uuid.uuid4().hex
                file = studio.root / "assets" / (asset_id + suffix)
                with file.open("wb") as output:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1048576, remaining))
                        if not chunk:
                            raise ValueError("文件上传中断。")
                        output.write(chunk)
                        remaining -= len(chunk)
                kind = "image" if suffix in IMAGE_EXTENSIONS else "video"
                self.json_response(
                    studio.register_asset(file, Path(name).name[:200], kind, asset_id),
                    201,
                )
            elif parsed.path == "/api/jobs":
                self.json_response(studio.create_job(self.read_json()), 202)
            elif parsed.path.startswith("/api/jobs/") and parsed.path.endswith(
                "/render"
            ):
                job_id = parsed.path[10:-7]
                if not ID.fullmatch(job_id):
                    raise KeyError("Job not found")
                self.json_response(studio.request_render(job_id, self.read_json()), 202)
            else:
                self.json_response({"error": "Not found"}, 404)
        except KeyError:
            self.json_response({"error": "Not found"}, 404)
        except (ValueError, TypeError, OverflowError) as error:
            self.json_response({"error": str(error)}, 400)

    def serve_file(self, path):
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        start, end = 0, size - 1
        ranged = self.headers.get("Range")
        if ranged:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", ranged.strip())
            if not match or not any(match.groups()):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            a, b = match.groups()
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            if start >= size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
        self.send_response(206 if ranged else 200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(str(path))[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = source.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--port", type=int, default=18768)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/studio"))
    parser.add_argument("--template", type=Path)
    parser.add_argument(
        "--advisor-profile",
        choices=["off", *PROFILES],
        default="sol-medium",
    )
    parser.add_argument("--proxy")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.gpu < 0:
        parser.error("Use a user port and a nonnegative GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import cv2

    cv2.setNumThreads(2)
    studio = Studio(
        StudioConfig(
            args.config, args.output, args.template, args.advisor_profile, args.proxy
        )
    )
    server = StudioHTTPServer(("127.0.0.1", args.port), studio)
    print(f"ApexVeil Studio: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        studio.close()


if __name__ == "__main__":
    main()
