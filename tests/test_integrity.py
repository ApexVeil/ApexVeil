from pathlib import Path
import hashlib
import pytest
from apexveil.models import verify
from apexveil.config import load_settings


def test_corrupt_checkpoint_is_rejected(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"expected model content")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    assert verify(model, expected)["verified"]
    model.write_bytes(b"changed content")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(model, expected)


def test_model_path_resolves_against_config_not_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "product"
    cfg = root / "configs/default.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('[project]\nroot = ".."\n[model]\ncheckpoint = "models/model.pt"\n')
    monkeypatch.chdir(tmp_path)
    assert load_settings(cfg).checkpoint == root / "models/model.pt"
