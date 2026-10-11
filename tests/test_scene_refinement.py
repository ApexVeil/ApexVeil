import importlib.util
import json
import sys
from pathlib import Path


def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "scene_refinement_test", scripts / "refine_scene_case.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_retry_budget_reviews_each_actual_candidate_and_stops(tmp_path, monkeypatch):
    mod = module(monkeypatch)
    rendered, reviewed = [], []

    def render(case, output, **kwargs):
        output.mkdir()
        (output / "report.json").write_text("{}")
        (output / "candidate-tight.png").write_bytes(b"candidate")
        rendered.append(output.name)
        return {"rendered": True}

    def advise(case, output, profile, proxy, *, candidate, current_report):
        assert candidate.is_file() and current_report.is_file()
        reviewed.append(candidate.parent.name)
        output.mkdir()
        (output / "reply.json").write_text("{}")
        return {"plate_confirmed": True, "assessment": {"acceptable": False}}, {
            "status": "proposal_available"
        }

    monkeypatch.setattr(mod, "render", render)
    monkeypatch.setattr(mod, "advise", advise)
    result = mod.refine(tmp_path / "case", tmp_path / "out", max_repairs=2)
    assert rendered == reviewed == ["render-00", "render-01", "render-02"]
    assert result["status"] == "repair_budget_exhausted"
    assert result["quality_accepted"] is False
    json.dumps(result)


def test_unsupported_model_never_calls_another_profile(tmp_path, monkeypatch):
    mod = module(monkeypatch)
    monkeypatch.setattr(mod, "render", lambda *a, **k: {"rendered": True})
    profiles = []

    def unavailable(case, output, profile, proxy, **kwargs):
        profiles.append(profile)
        return None, {"status": "failed"}

    monkeypatch.setattr(mod, "advise", unavailable)
    result = mod.refine(tmp_path / "case", tmp_path / "out", profile="sol-medium")
    assert profiles == ["sol-medium"]
    assert result["status"] == "advisor_unavailable"
