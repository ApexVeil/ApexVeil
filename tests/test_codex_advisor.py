import json
import subprocess

import pytest

from apexveil.backends.codex_advisor import (
    CodexSceneAdvisor,
    to_source_geometry,
    validate_guidance,
)


def proposal():
    return {
        "plate_confirmed": True,
        "coordinate_space": "crop_pixels",
        "geometry": {
            "edge_labels": {s: "unknown" for s in ["top", "right", "bottom", "left"]},
            "physical_aspect_guess": None,
            "candidate_quads": [],
            "assumptions": [],
        },
        "appearance": {
            s: None
            for s in [
                "exposure_gain_linear",
                "additive_rgb_linear",
                "flare_control_points",
                "shading_control_points",
                "diffuse_tint_rgb",
            ]
        },
        "assessment": {"acceptable": True, "issues": [], "next_action": "review"},
    }


def test_coordinate_contract_preserves_offscreen_hypotheses_and_rejects_bad_values():
    p = proposal()
    p["geometry"]["candidate_quads"] = [[[-20, 0], [80, 0], [80, 30], [-20, 30]]]
    mapped = to_source_geometry(p, [0, 100, 100, 150], source_size=[200, 200])
    assert mapped["candidate_quads"][0][0] == [-20, 100]
    for value in [float("nan"), True, "20"]:
        p["geometry"]["candidate_quads"][0][0][0] = value
        with pytest.raises((ValueError, TypeError)):
            validate_guidance(p)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["geometry"]["edge_labels"].update(left="leg"),
        lambda p: p["appearance"].update(
            flare_control_points=[{"xy": [5, 0], "rgb_linear": [0.1, 0.1, 0.1]}]
        ),
        lambda p: p["appearance"].update(additive_rgb_linear=[-1, 0, 0]),
        lambda p: p.update(extra="not allowed"),
    ],
)
def test_advice_contract_rejects_unsupported_fields_and_lighting(mutation):
    p = proposal()
    mutation(p)
    with pytest.raises((ValueError, TypeError)):
        validate_guidance(p)


def test_cli_prompt_is_stdin_after_image_separator_and_never_injects_api_key(
    tmp_path, monkeypatch
):
    image = tmp_path / "source.png"
    image.write_bytes(b"test")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-reach-child")

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            assert args[-2:] == ["--", "-"]
            assert args[args.index("--model") + 1] == "gpt-6-luna"
            assert 'model_reasoning_effort="medium"' in args
            assert "OPENAI_API_KEY" not in kwargs["env"]
            assert "CODEX_API_KEY" not in kwargs["env"]
            assert kwargs["start_new_session"]
            self.reply = args[args.index("-o") + 1]
            self.out = kwargs["stdout"]

        def communicate(self, *, input, timeout):
            assert input == "Inspect the actual crop."
            from pathlib import Path

            Path(self.reply).write_text(json.dumps(proposal()))
            self.out.write(
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10}})
                + "\n"
            )
            self.out.flush()

    monkeypatch.setattr(subprocess, "Popen", Process)
    advice, report = CodexSceneAdvisor().run(
        tmp_path / "job", "Inspect the actual crop.", [image]
    )
    assert advice["assessment"]["acceptable"]
    assert report["status"] == "proposal_available"
    assert report["quality_accepted"] is False
    assert report["usage"]["input_tokens"] == 10


def test_missing_executable_is_reported_without_fallback(tmp_path):
    image = tmp_path / "source.png"
    image.write_bytes(b"test")
    advice, report = CodexSceneAdvisor(executable=str(tmp_path / "missing-codex")).run(
        tmp_path / "job", "inspect", [image]
    )
    assert advice is None
    assert report["status"] == "failed"
    assert "FileNotFoundError" in report["error"]
    assert report["quality_accepted"] is False


def test_timeout_kills_process_group_and_cannot_accept_reply(tmp_path, monkeypatch):
    image = tmp_path / "source.png"
    image.write_bytes(b"test")
    killed = []

    class Process:
        pid = 412

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, **kwargs):
            raise subprocess.TimeoutExpired("codex", 1)

        def wait(self):
            return -9

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(
        "apexveil.backends.codex_advisor.os.killpg", lambda pid, sig: killed.append(pid)
    )
    advice, report = CodexSceneAdvisor(timeout_seconds=1).run(
        tmp_path / "job", "inspect", [image]
    )
    assert killed == [412]
    assert advice is None
    assert report["timed_out"]


@pytest.mark.parametrize(
    "prefix,model",
    [("luna", "gpt-6-luna"), ("sol", "gpt-6.1-sol"), ("astra", "gpt-6-astra")],
)
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_every_selection_reaches_exact_cli_model_and_effort(
    tmp_path, monkeypatch, prefix, model, effort
):
    image = tmp_path / "source.png"
    image.write_bytes(b"input")

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            assert args[args.index("--model") + 1] == model
            assert args[args.index("-c") + 1] == f'model_reasoning_effort="{effort}"'
            self.reply = args[args.index("-o") + 1]

        def communicate(self, *, input, timeout):
            assert 180 <= timeout <= 600
            from pathlib import Path

            Path(self.reply).write_text(json.dumps(proposal()))

    monkeypatch.setattr(subprocess, "Popen", Process)
    _, record = CodexSceneAdvisor(profile=f"{prefix}-{effort}").run(
        tmp_path / "job", "inspect", [image]
    )
    assert record["requested_model"] == model
    assert record["reasoning_effort"] == effort
    assert record["status"] == "proposal_available"


def test_arbitrary_model_or_effort_cannot_be_injected():
    from apexveil.backends.codex_advisor import advisor_profile

    for model, effort in [
        ("gpt-unknown", "medium"),
        ("gpt-6-astra", "arbitrary"),
        ("gpt-6-luna", None),
    ]:
        with pytest.raises((ValueError, TypeError)):
            advisor_profile(model, effort)
