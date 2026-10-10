import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate_cached_images import evaluate


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cache = tmp_path / "cache" / "image"
    cache.mkdir(parents=True)
    rgb = np.full((100, 200, 3), 100, np.uint8)
    rgb[30:70, 30:170] = [60, 150, 65]
    path = source / "image.png"
    Image.fromarray(rgb).save(path)
    mask = np.zeros((100, 200), np.uint8)
    mask[30:70, 30:170] = 255
    Image.fromarray(mask).save(cache / "mask.png")
    metadata = {
        "source_name": "image.png",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": [200, 100],
        "surfaces": [{"object_id": 0, "score": 0.9, "visible_mask": "mask.png"}],
    }
    (cache / "report.json").write_text(json.dumps(metadata))
    template = tmp_path / "template.png"
    art = np.full((40, 140, 4), 30, np.uint8)
    art[:, :, 3] = 255
    Image.fromarray(art).save(template)
    return source, cache.parent, template, tmp_path / "output"


def test_cached_evaluation_writes_index_and_bound_provenance(tmp_path):
    source, reports, template, out = fixture(tmp_path)
    result = evaluate(source, reports, template, out)
    assert result["errors"] == []
    assert not result["source_changed_during_evaluation"]
    assert (out / "index.html").is_file()
    assert "image/replace.png" in (out / "index.html").read_text()
    assert (out / "source-snapshot.json").is_file()
    report = json.loads((out / "image" / "report.json").read_text())
    assert report["segmentation"]["inference_ran"] is False
    assert report["segmentation"]["native_mask_sha256"]["0"]
    assert report["surfaces"][0]["visible_mask_sha256"]


def test_cached_evaluation_refuses_unbound_input_and_manual_geometry(tmp_path):
    source, reports, template, out = fixture(tmp_path)
    with pytest.raises(ValueError, match="target selection only"):
        evaluate(source, reports, template, out, {"image.png": {"quads": {}}})
    Image.new("RGB", (200, 100), "white").save(source / "image.png")
    with pytest.raises(ValueError, match="Cache/source mismatch"):
        evaluate(source, reports, template, out)
