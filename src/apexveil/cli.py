"""Small, honest setup CLI. Editing commands will be added as validated."""

import argparse
import json
from pathlib import Path
from . import __version__
from .config import load_settings


def main():
    parser = argparse.ArgumentParser(prog="apexveil")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default="configs/default.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect this installation without model inference")
    models = commands.add_parser("models", help="Download or verify the pinned official checkpoint")
    models.add_argument("action", choices=["download", "verify"])
    smoke = commands.add_parser("smoke", help="Run bounded SAM3.1 segmentation and tracking validation")
    smoke.add_argument("resource", type=Path, help="An image-frame directory or short video")
    smoke.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    smoke.add_argument("--text", default="license plate")
    smoke.add_argument("--max-frames", type=int, default=1)
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.command == "doctor":
        from .diagnostics import doctor
        result = doctor(settings)
    elif args.command == "models":
        from .models import model_assets
        result = model_assets(settings, download=args.action == "download")
    else:
        if args.max_frames < 1 or args.max_frames > 120:
            parser.error("--max-frames must be between 1 and 120 for a smoke test")
        from .backends.sam31 import smoke
        result = smoke(settings, args.resource, args.output, args.text, args.max_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
