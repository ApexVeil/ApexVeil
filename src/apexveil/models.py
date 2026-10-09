"""Official model download and reproducible integrity checks."""

from pathlib import Path
import hashlib
import json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> dict:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": actual, "verified": True}


def model_assets(settings, download: bool = False) -> dict:
    manifest = json.loads((settings.root / "models/manifest.json").read_text())
    if download:
        from huggingface_hub import hf_hub_download
        settings.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for filename in ["LICENSE", "config.json", manifest["filename"]]:
            hf_hub_download(repo_id=manifest["repository"], filename=filename,
                            revision=manifest["revision"], local_dir=settings.checkpoint.parent)
    return verify(settings.checkpoint, manifest["sha256"])
