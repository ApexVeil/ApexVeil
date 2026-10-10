"""Optional, bounded scene advice using the operator's existing Codex login.

The advisor produces data for the fixed renderer. It never changes renderer
source, switches model/provider on failure, or silently supplies an API key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODEL_OPTIONS = (
    {"id": "gpt-6-luna", "label": "GPT-6 Luna", "prefix": "luna"},
    {"id": "gpt-6.1-sol", "label": "GPT-6.1 Sol", "prefix": "sol"},
    {"id": "gpt-6-astra", "label": "GPT-6 Astra", "prefix": "astra"},
)
PROFILES = {
    f"{m['prefix']}-{effort}": m["id"] for m in MODEL_OPTIONS for effort in EFFORTS
}
EFFORT_TIMEOUTS = {"low": 180, "medium": 180, "high": 300, "xhigh": 480, "max": 600}


def advisor_profile(model, effort):
    if not isinstance(model, str) or not isinstance(effort, str):
        raise TypeError("模型与思考等级需为字符串。")
    option = next((m for m in MODEL_OPTIONS if m["id"] == model), None)
    if option is None or effort not in EFFORTS:
        raise ValueError("不支持的模型或思考等级。")
    return f"{option['prefix']}-{effort}"


def profile_selection(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown Codex profile: {profile}")
    return PROFILES[profile], profile.rsplit("-", 1)[1]


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def guidance_schema():
    number = {"type": "number"}
    xy = {
        "type": "array",
        "items": number,
        "minItems": 2,
        "maxItems": 2,
        "description": "Geometry only: native tight-crop pixel coordinates, excluding ruler border; may be outside the crop.",
    }
    uv = {
        "type": "array",
        "items": {"type": "number", "minimum": 0, "maximum": 1},
        "minItems": 2,
        "maxItems": 2,
        "description": "Appearance only: normalized UV on the COMPLETE plate, from [0,0] top-left to [1,1] bottom-right.",
    }
    rgb = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    nullable = lambda schema: {"anyOf": [schema, {"type": "null"}]}
    return _object(
        {
            "plate_confirmed": {"type": "boolean"},
            "coordinate_space": {"type": "string", "enum": ["crop_pixels"]},
            "geometry": _object(
                {
                    "edge_labels": _object(
                        {
                            side: {
                                "type": "string",
                                "enum": [
                                    "real_plate",
                                    "image_crop",
                                    "occlusion",
                                    "unknown",
                                ],
                            }
                            for side in ["top", "right", "bottom", "left"]
                        }
                    ),
                    "physical_aspect_guess": nullable(number),
                    "candidate_quads": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "array",
                            "items": xy,
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                }
            ),
            "appearance": _object(
                {
                    "exposure_gain_linear": nullable(number),
                    "additive_rgb_linear": nullable(rgb),
                    "flare_control_points": nullable(
                        {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": _object({"xy": uv, "rgb_linear": rgb}),
                        }
                    ),
                    "shading_control_points": nullable(
                        {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": _object({"xy": uv, "value": number}),
                        }
                    ),
                    "diffuse_tint_rgb": nullable(rgb),
                }
            ),
            "assessment": _object(
                {
                    "acceptable": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "next_action": {"type": "string"},
                }
            ),
        }
    )


def _validate_node(value, schema, path="advice"):
    """Validate our small closed JSON contract without a runtime dependency."""
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            try:
                _validate_node(value, variant, path)
                return
            except (ValueError, TypeError):
                pass
        raise ValueError(f"{path}: no allowed type matches")
    kind = schema["type"]
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }[kind]
    if not valid:
        raise TypeError(f"{path}: expected {kind}")
    if kind == "object":
        properties = schema["properties"]
        if set(value) != set(properties):
            raise ValueError(f"{path}: fields must match the closed advice contract")
        for key, item in value.items():
            _validate_node(item, properties[key], f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000):
            raise ValueError(f"{path}: invalid array length")
        for index, item in enumerate(value):
            _validate_node(item, schema["items"], f"{path}[{index}]")
    elif kind == "number":
        if not math.isfinite(value):
            raise ValueError(f"{path}: number must be finite")
        if (
            not schema.get("minimum", -math.inf)
            <= value
            <= schema.get("maximum", math.inf)
        ):
            raise ValueError(f"{path}: number outside supported bounds")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unsupported value")


def validate_guidance(value):
    _validate_node(value, guidance_schema())
    geometry = value["geometry"]
    aspect = geometry["physical_aspect_guess"]
    if aspect is not None and not 0.25 <= aspect <= 10:
        raise ValueError("Physical aspect assumption must be within 0.25..10")
    # Validate the same appearance bounds before launching a render. No image-sized
    # arrays are needed for contract validation.
    import numpy as np

    from apexveil.imaging.appearance import AppearanceEstimate
    from apexveil.imaging.scene_appearance import apply_scene_guidance

    estimate = AppearanceEstimate(
        1.0,
        np.zeros((2, 2, 3), np.float32),
        np.ones(3, np.float32),
        np.ones((2, 2), np.float32),
        {},
    )
    apply_scene_guidance(estimate, value["appearance"])
    return value


def to_source_geometry(advice, crop_xyxy, *, source_size):
    """Map numerical proposals, never a drawn/rasterized model mask, to source."""
    validate_guidance(advice)
    x0, y0, _, _ = crop_xyxy
    width, height = source_size
    candidates = []
    for quad in advice["geometry"]["candidate_quads"]:
        mapped = [[float(x + x0), float(y + y0)] for x, y in quad]
        if any(
            not -4 * width <= x <= 5 * width or not -4 * height <= y <= 5 * height
            for x, y in mapped
        ):
            raise ValueError(
                "Proposed corners exceed the bounded source hypothesis space"
            )
        candidates.append(mapped)
    return {
        "source": "codex_scene_advisor",
        "coordinate_space": "source",
        "edge_labels": advice["geometry"]["edge_labels"],
        "physical_aspect_guess": advice["geometry"]["physical_aspect_guess"],
        "candidate_quads": candidates,
        "assumptions": advice["geometry"]["assumptions"],
    }


@dataclass
class CodexSceneAdvisor:
    profile: str = "luna-medium"
    timeout_seconds: float | None = None
    proxy_url: str | None = None
    executable: str = "codex"

    def run(self, job_dir, prompt, images):
        model, effort = profile_selection(self.profile)
        timeout = (
            self.timeout_seconds
            if self.timeout_seconds is not None
            else EFFORT_TIMEOUTS[effort]
        )
        if not 1 <= timeout <= 600:
            raise ValueError("Advisor timeout must be within 1..600 seconds")
        job = Path(job_dir).resolve()
        job.mkdir(parents=True, exist_ok=True)
        if (job / "advisor-report.json").exists():
            raise ValueError("Use a new directory for each advisor iteration")
        image_paths = [Path(p).resolve() for p in images]
        if not image_paths or any(not p.is_file() for p in image_paths):
            raise ValueError("Advisor needs existing input images")
        schema_path = job / "response.schema.json"
        schema_path.write_text(json.dumps(guidance_schema(), indent=2))
        (job / "prompt.txt").write_text(prompt)
        reply = job / "reply.json"
        args = [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--disable",
            "apps",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{effort}"',
            "--json",
            "--output-schema",
            str(schema_path),
            "-o",
            str(reply),
            "-C",
            str(job),
        ]
        for path in image_paths:
            args += ["--image", str(path)]
        args += ["--", "-"]
        env = os.environ.copy()
        for key in ["OPENAI_API_KEY", "CODEX_API_KEY"]:
            env.pop(key, None)
        if self.proxy_url:
            for key in ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"]:
                env[key] = self.proxy_url
        started = time.monotonic()
        timed_out = False
        launch_error = None
        with (
            (job / "events.jsonl").open("w") as out,
            (job / "stderr.log").open("w") as err,
        ):
            try:
                process = subprocess.Popen(
                    args,
                    env=env,
                    stdin=subprocess.PIPE,
                    text=True,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
            except OSError as error:
                code = None
                launch_error = f"{type(error).__name__}: {error}"
            else:
                try:
                    process.communicate(input=prompt, timeout=timeout)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = process.wait()
        record = {
            "requested_model": model,
            "reasoning_effort": effort,
            "timeout_seconds": timeout,
            "profile": self.profile,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": code,
            "timed_out": timed_out,
            "auth_policy": "existing_codex_login_no_api_key_injection",
            "images": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in image_paths
            ],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "status": "failed",
            "quality_accepted": False,
            "usage": None,
        }
        if launch_error:
            record["error"] = launch_error
        for line in (job / "events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "turn.completed":
                record["usage"] = event.get("usage")
            if event.get("type") in ["error", "turn.failed"]:
                record["error"] = str(event.get("message", event.get("error", "")))[
                    :1000
                ]
        advice = None
        if code == 0 and not timed_out and reply.is_file():
            try:
                advice = validate_guidance(json.loads(reply.read_text()))
                record["status"] = "proposal_available"
                # One model saying acceptable is advisory, not independent acceptance.
                record["model_assessment"] = advice["assessment"]
            except (ValueError, KeyError, TypeError) as error:
                record["error"] = str(error)
        if not reply.exists() and "error" not in record:
            record["error"] = "No structured advisor response"
        (job / "advisor-report.json").write_text(json.dumps(record, indent=2))
        return advice, record
