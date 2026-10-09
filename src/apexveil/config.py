"""Portable project configuration, independent of the execution host."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Settings:
    root: Path
    checkpoint: Path
    confidence_threshold: float
    use_fa3: bool
    compile: bool


def load_settings(path: str | Path = "configs/default.toml") -> Settings:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    root = (path.parent / data["project"].get("root", "..")).resolve()
    model = data["model"]
    checkpoint = Path(model["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    threshold = float(model.get("confidence_threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    return Settings(root, checkpoint, threshold, bool(model.get("use_fa3", False)), bool(model.get("compile", False)))
