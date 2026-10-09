"""Apply explicit image corrections to saved, source-bound detections without GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from apexveil.backends.image import Detection
from apexveil.image_pipeline import process_image
from apexveil.imaging.compositor import load_template


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--overrides", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    overrides = json.loads(a.overrides.read_text())
    template = load_template(a.template)
    a.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, override in overrides.items():
        source = a.source / name
        cache = a.reports / source.stem
        report_path = cache / "report.json"
        old = json.loads(report_path.read_text())
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if old["source_sha256"] != actual or old["source_name"] != name:
            raise ValueError(f"Detection cache does not belong to source: {name}")
        detections = [
            Detection(
                s["object_id"],
                s["score"],
                (np.array(Image.open(cache / s["visible_mask"])) > 0).astype(np.uint8),
                s.get("detection", {}),
            )
            for s in old["surfaces"]
        ]

        class SavedDetections:
            def detect(self, image, prompt, refine):
                if list(image.size) != old["size"]:
                    raise ValueError("Cache/image dimensions differ")
                return detections, {
                    "cache_reused": True,
                    "inference_ran": False,
                    "cache_report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                    "source_sha256": actual,
                }

        report = process_image(
            source,
            a.output / source.stem,
            SavedDetections(),
            template=template,
            override=override,
            allow_uncertain=bool(override.get("allow_uncertain", False)),
        )
        report["correction_provenance"] = {
            "kind": "explicit_external_override",
            "note": override.get("note", ""),
            "automatic_geometry": not bool(override.get("quads")),
            "manual_corners_are_ground_truth": False,
        }
        (a.output / source.stem / "report.json").write_text(
            json.dumps(report, indent=2)
        )
        summaries.append(
            {
                "file": name,
                "status": report["results"]["replace"]["status"],
                "correction": report["correction_provenance"],
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    (a.output / "report.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
