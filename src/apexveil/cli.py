"""Setup, diagnostic, and inspectable image-processing commands."""

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
    commands.add_parser(
        "doctor", help="Inspect this installation without model inference"
    )
    models = commands.add_parser(
        "models", help="Download or verify the pinned official checkpoint"
    )
    models.add_argument("action", choices=["download", "verify"])
    smoke = commands.add_parser(
        "smoke", help="Run bounded SAM3.1 segmentation and tracking validation"
    )
    smoke.add_argument(
        "resource", type=Path, help="An image-frame directory or short video"
    )
    smoke.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    smoke.add_argument("--text", default="license plate")
    smoke.add_argument("--max-frames", type=int, default=1)
    image = commands.add_parser(
        "image",
        help="Process an image or directory; save masks, effects and diagnostics",
    )
    image.add_argument("source", type=Path)
    image.add_argument("--output", type=Path, required=True)
    image.add_argument(
        "--mode", choices=["replace", "blur", "mosaic"], default="replace"
    )
    image.add_argument("--template", type=Path, dest="template_path")
    image.add_argument(
        "--target-id",
        type=int,
        help="Image-local candidate ID; required for multi-target replacement",
    )
    image.add_argument(
        "--overrides",
        type=Path,
        dest="overrides_path",
        help="Per-filename target_id, exclude_ids and quads JSON",
    )
    image.add_argument(
        "--comparison",
        action="store_true",
        help="Also export blur/mosaic and local comparison sheets",
    )
    image.add_argument("--strength", type=float, default=0.12)
    image.add_argument(
        "--blur-sigma",
        type=float,
        default=None,
        help="Override automatic edge-softness estimate, in source-image pixels",
    )
    image.add_argument(
        "--allow-uncertain",
        action="store_true",
        help="Render flagged geometry as an explicitly unreviewed preview",
    )
    image.add_argument("--prompt", default="license plate")
    image.add_argument(
        "--no-refine", action="store_true", help="Disable automatic crop refinement"
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.command == "doctor":
        from .diagnostics import doctor

        result = doctor(settings)
    elif args.command == "models":
        from .models import model_assets

        result = model_assets(settings, download=args.action == "download")
    elif args.command == "image":
        if not 0 < args.strength <= 1 or (
            args.blur_sigma is not None and not 0 <= args.blur_sigma <= 20
        ):
            parser.error("strength must be in (0, 1] and blur-sigma in [0, 20]")
        if args.mode == "replace" and args.template_path is None:
            parser.error("--template is required for replacement")
        from .image_pipeline import run_images

        result = run_images(
            settings,
            args.source,
            args.output,
            mode=args.mode,
            template_path=args.template_path,
            target_id=args.target_id,
            overrides_path=args.overrides_path,
            comparison=args.comparison,
            strength=args.strength,
            blur_sigma=args.blur_sigma,
            allow_uncertain=args.allow_uncertain,
            prompt=args.prompt,
            refine=not args.no_refine,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "processed": len(result["records"]),
                    "errors": result["errors"],
                },
                indent=2,
            )
        )
        if result["errors"]:
            raise SystemExit(1)
        return
    else:
        if args.max_frames < 1 or args.max_frames > 120:
            parser.error("--max-frames must be between 1 and 120 for a smoke test")
        from .backends.sam31 import smoke

        result = smoke(settings, args.resource, args.output, args.text, args.max_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
