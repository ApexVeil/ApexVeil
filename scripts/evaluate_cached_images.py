"""Re-evaluate automatic compositing against source-bound SAM image caches.

Selection files identify targets/exclusions only. Geometry corrections and per-image
appearance parameters are deliberately not accepted by this automatic evaluation.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from apexveil.backends.image import Detection
from apexveil.image_pipeline import process_image
from apexveil.imaging.compositor import load_template


class SavedDetections:
    def __init__(self, size, detections, source_hash, report_path):
        self.size = size
        self.detections = detections
        self.provenance = {
            "cache_reused": True,
            "inference_ran": False,
            "source_sha256": source_hash,
            "cache_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "native_mask_sha256": {
                str(d.object_id): hashlib.sha256(
                    np.asarray(d.visible_mask, np.uint8).tobytes()
                ).hexdigest()
                for d in detections
            },
        }

    def detect(self, image, prompt, refine):
        if list(image.size) != self.size:
            raise ValueError("Cache dimensions differ")
        return self.detections, dict(self.provenance)


def evaluate(source, reports, template_path, output, selections=None):
    source, reports, output = map(Path, (source, reports, output))
    output.mkdir(parents=True, exist_ok=True)
    template = load_template(Path(template_path))
    selections = selections or {}
    records, errors = [], []
    paths = sorted(reports.glob("*/report.json"))
    if not paths:
        raise ValueError("No image reports found")
    snapshot = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("src/apexveil").rglob("*.py"))
    }
    (output / "source-snapshot.json").write_text(json.dumps(snapshot, indent=2))
    for report_path in paths:
        old = json.loads(report_path.read_text())
        path = source / old["source_name"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != old["source_sha256"]:
            raise ValueError(f"Cache/source mismatch: {path.name}")
        selection = selections.get(path.name, {})
        if set(selection) - {"target_id", "exclude_ids"}:
            raise ValueError("Automatic evaluation accepts target selection only")
        detections = [
            Detection(
                int(s["object_id"]),
                float(s["score"]),
                np.asarray(
                    Image.open(report_path.parent / s["visible_mask"]).convert("L")
                )
                > 0,
                s.get("detection", {}),
            )
            for s in old["surfaces"]
        ]

        saved = SavedDetections(old["size"], detections, actual, report_path)

        try:
            result = process_image(
                path,
                output / path.stem,
                saved,
                template=template,
                override=selection,
                comparison=True,
            )
            records.append(
                {
                    "source": path.name,
                    "report": f"{path.stem}/report.json",
                    "selection": selection,
                    "automatic_geometry": True,
                    "status": result["results"]["replace"]["status"],
                    "flags": result["results"]["replace"]["flags"],
                }
            )
            print(json.dumps(records[-1]), flush=True)
        except Exception as error:  # noqa: BLE001 -- record batch failure, never mark success
            errors.append(
                {"source": path.name, "error": f"{type(error).__name__}: {error}"}
            )
            print(json.dumps(errors[-1]), flush=True)
    from apexveil.diagnostic_export import write_image_index

    diagnostic_reports = []
    for record in records:
        stem = Path(record["source"]).stem
        report = json.loads((output / record["report"]).read_text())
        diagnostic = report.get("diagnostics")
        if diagnostic:
            for key, value in diagnostic["diagnostics"].items():
                diagnostic["diagnostics"][key] = f"{stem}/{value}"
            for key, value in diagnostic.get("provenance", {}).get("links", {}).items():
                diagnostic["provenance"]["links"][key] = f"{stem}/{value}"
            diagnostic_reports.append(diagnostic)
    if diagnostic_reports:
        write_image_index(
            output, diagnostic_reports, title="ApexVeil · Automatic image evaluation"
        )
    current = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("src/apexveil").rglob("*.py"))
    }
    manifest = {
        "kind": "automatic_cached_image_evaluation",
        "records": records,
        "errors": errors,
        "source_snapshot": snapshot,
        "source_changed_during_evaluation": snapshot != current,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "template": template.metadata,
        "quality_certified": False,
        "note": "Native masks are reused model outputs, not ground truth. Status is a review workflow, not a quality score.",
    }
    (output / "evaluation.json").write_text(json.dumps(manifest, indent=2))
    if errors or snapshot != current:
        raise RuntimeError("Evaluation incomplete or code changed during evaluation")
    return manifest


def main():
    import cv2

    cv2.setNumThreads(4)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reports", "template", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--selections", type=Path)
    a = p.parse_args()
    evaluate(
        a.source,
        a.reports,
        a.template,
        a.output,
        json.loads(a.selections.read_text()) if a.selections else None,
    )


if __name__ == "__main__":
    main()
