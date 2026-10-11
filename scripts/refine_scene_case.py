"""Bounded Codex proposal/render/review loop over a source-bound evidence case.

The operator supplies a case pack (including any foreground evidence). A model
assessment never becomes an independent product-quality certificate. Every
attempt and actual rendered candidate is retained, including unresolved ones.
"""

import argparse
import json
from pathlib import Path

import cv2
from run_scene_advisor import run as advise
from solve_scene_case import run as render


def refine(
    case,
    output,
    *,
    profile="luna-medium",
    proxy=None,
    max_repairs=2,
    foreground=None,
    initial_advice=None,
):
    if not 0 <= max_repairs <= 3:
        raise ValueError("max_repairs must be within 0..3")
    case, output = Path(case).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory for each refinement job")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "profile": profile,
        "max_repairs": max_repairs,
        "quality_accepted": False,
        "status": "needs_review",
        "rounds": [],
        "foreground_path": None if foreground is None else str(foreground),
        "case": str(case),
    }
    current = output / "render-00"
    report = render(
        case, current, advice_path=initial_advice, foreground_path=foreground
    )
    for index in range(max_repairs + 1):
        advisor_dir = output / f"advisor-{index:02d}"
        advice, call = advise(
            case,
            advisor_dir,
            profile,
            proxy,
            candidate=current / "candidate-tight.png",
            current_report=current / "report.json",
        )
        result["rounds"].append(
            {
                "reviewed_render": current.name,
                "advisor": advisor_dir.name,
                "call_status": call["status"],
                "rendered": report["rendered"],
                "model_assessment": None if advice is None else advice["assessment"],
            }
        )
        if advice is None:
            result["status"] = "advisor_unavailable"
            break
        if not advice["plate_confirmed"]:
            result["status"] = "target_not_confirmed"
            break
        if advice["assessment"]["acceptable"] and report["rendered"]:
            result["status"] = "model_reviewed_candidate"
            break
        if index == max_repairs:
            result["status"] = "repair_budget_exhausted"
            break
        current = output / f"render-{index + 1:02d}"
        report = render(
            case,
            current,
            advice_path=advisor_dir / "reply.json",
            foreground_path=foreground,
        )
    result["final_candidate"] = str(current)
    result["rendered"] = report["rendered"]
    (output / "refinement-report.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {"status": result["status"], "candidate": str(current)}, ensure_ascii=False
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["luna-medium", "sol-medium"], default="luna-medium"
    )
    parser.add_argument("--proxy")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--foreground", type=Path)
    parser.add_argument("--initial-advice", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    result = refine(
        args.case,
        args.output,
        profile=args.profile,
        proxy=args.proxy,
        max_repairs=args.max_repairs,
        foreground=args.foreground,
        initial_advice=args.initial_advice,
    )
    if result["status"] != "model_reviewed_candidate":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
