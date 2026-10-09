"""Read-only environment diagnostics; never enumerate credentials."""

import importlib.metadata
import json
import shutil
import subprocess
from . import __version__


def doctor(settings) -> dict:
    result = {"apexveil": __version__, "checkpoint_present": settings.checkpoint.is_file(),
              "ffmpeg": bool(shutil.which("ffmpeg")), "codex": bool(shutil.which("codex")),
              "packages": {}}
    for package in ["torch", "torchvision", "numpy", "sam3", "huggingface-hub"]:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        result["visible_gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        result["cuda_available"] = False
    upstream = settings.root / "third_party/sam3"
    proc = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True)
    result["upstream_commit"] = proc.stdout.strip() if proc.returncode == 0 else None
    manifest = json.loads((settings.root / "models/manifest.json").read_text())
    result["upstream_matches_manifest"] = result["upstream_commit"] == manifest["upstream_commit"]
    return result
