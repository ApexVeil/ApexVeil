# ApexVeil · 极影

**Precision in motion.**

ApexVeil is an independent project for scene-consistent automotive image and video compositing. Its first workflow is replacing or anonymizing license plates while respecting perspective, visibility, lighting and imaging blur.

[Project page](https://apexveil.github.io) · [Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md)

## Status

**Experimental compositing core / pre-alpha.** Images and videos share deterministic composition with automatic geometry, boundary, appearance and camera-degradation estimation. Offline video uses bidirectional motion consensus within reviewed shots. Estimator interfaces allow future learned modules without regenerating the supplied artwork. Multi-target treatment requires explicit selection. Difficult occlusion, clipping and optical-edge cases still require review.

| Capability | Current scope |
|---|---|
| Still-image processing | CLI, directory batches, visible masks, corner corrections, replacement/blur/mosaic |
| Short video composition | Bounded continuous-shot experiment; native SAM tracking and offline geometry |
| Reviewed montage | Manifest-driven shots and source audio; a 70-second, 30 fps development montage. Automatic shot selection and native VFR are not implemented |
| Optional Codex scene advice | Experimental source-bound proposal → render → visual review loop; bounded retries, existing operator login |
| Interactive Web editor | Planned; diagnostic result pages are available |

SAM3.1 is one model backend; ApexVeil's product and source are independent of upstream model naming and any other project.

## Setup

Prerequisites: Git, Conda, an NVIDIA GPU/compatible driver and FFmpeg on PATH. The reference environment uses Python 3.12 and PyTorch 2.10 with CUDA 12.8. Obtain access to the official [SAM3.1 checkpoint](https://huggingface.co/facebook/sam3.1) and authenticate with your own Hugging Face account.

```sh
git clone --recurse-submodules https://github.com/ApexVeil/ApexVeil.git
cd ApexVeil
conda env create -f environment.yml
conda activate apexveil
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/runtime.txt
python -m pip install --no-deps -e third_party/sam3
python -m pip install -e '.[dev]'
hf auth login
apexveil models download
apexveil doctor
```

The checkpoint lives in `models/sam3.1/`, with pinned provenance in [models/manifest.json](models/manifest.json). Model weights and user media are never committed.

## Image processing

```sh
CUDA_VISIBLE_DEVICES=0 apexveil image /path/to/photo.jpg \
  --template /path/to/artwork.png --output outputs/example --comparison
```

The command saves effects, native candidate masks, declared edit support, geometry previews and a JSON report. With `--comparison`, diagnostic panels expose RGB, raw SAM masks and the latest output.
See [image pipeline documentation](docs/IMAGE_PIPELINE.md) for multi-target
selection, manual corner overrides and current limitations.

## Bounded video composition

```sh
CUDA_VISIBLE_DEVICES=0 python -m apexveil.video_pipeline /path/to/source.mp4 \
  --start 240 --duration 2 --template /path/to/artwork.png \
  --output outputs/video-example
```

This exports original, independent-frame and temporal comparison MP4s with frame
and audio diagnostics. It accepts at most four seconds of an already-selected
continuous shot. Multi-target clips require `--target-id`; visually confirmed
false tracks may be excluded with `--exclude-ids`. See [video pipeline](docs/VIDEO_PIPELINE.md)
and [development experiment](docs/VIDEO_EXPERIMENT_20261010.md) for limitations.

## Scene solver and optional AI guidance

A newer [scene solver](docs/SCENE_SOLVER.md) separates the complete plane from
visible pixels, image clipping and foreground evidence. It shares one deterministic
renderer between still images and video frames. Optional Codex guidance supplies
bounded geometry and lighting data; it does not redraw the supplied artwork or
modify renderer code. This opt-in developer workflow is not yet the default image
or montage command, and the interactive editor remains planned.

## Bounded GPU validation

Place a short video or numerically named JPEG frames in a private data directory. A single frame is a one-image segmentation test. The command writes masks and a JSON report; it does not change the original or replace its pixels.

```sh
CUDA_VISIBLE_DEVICES=0 apexveil smoke data/private/smoke-frames --max-frames 3 --output outputs/smoke
python -m pytest -q
```

## Estimation and evaluation

See the [estimator contract](docs/ESTIMATOR_CONTRACT.md), [offline geometry](docs/offline-geometry-design.md) and [appearance estimator](docs/appearance-estimation.md). These documents distinguish implemented estimation from unresolved visual failures. Tests check compositing and tracking contracts; they do not certify photorealism or anonymization.

## Product direction

- One shared processing core for images and video.
- Transparent custom artwork, blur and mosaic as per-target treatments.
- User selection through clicks or short strokes, with explicit corrections.
- Separate full-surface geometry from the currently visible region.
- Server-side AI review, configurable by the operator.
- A remote Web editor accessed through ordinary SSH forwarding during development.

See [development notes](docs/DEVELOPMENT.md) for the intended forwarding workflow. Public hosting and invitations are deferred until the core is ready.

## License

Original ApexVeil code: Apache-2.0. SAM source and weights retain their separate upstream SAM License. See [third-party notices](THIRD_PARTY_NOTICES.md).
