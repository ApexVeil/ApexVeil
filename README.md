<div align="center">

<img src="docs/assets/apexveil-icon.svg" alt="ApexVeil icon" width="112" height="112" />

# ApexVeil · 极影

**Precision in motion.**

Natural plate replacement and privacy for automotive images and video.

[Website](https://apexveil.github.io) ·
[Watch the demo](https://apexveil.github.io/#video) ·
[Quick start](#quick-start) ·
[Studio guide](docs/STUDIO.md) ·
[Roadmap](docs/ROADMAP.md)

[![Checks](https://github.com/ApexVeil/ApexVeil/actions/workflows/ci.yml/badge.svg)](https://github.com/ApexVeil/ApexVeil/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](environment.yml)
[![Status](https://img.shields.io/badge/Status-pre--alpha-FF694A?style=flat-square)](#current-scope)
[![License](https://img.shields.io/badge/License-Apache--2.0-8B949E?style=flat-square)](LICENSE)

</div>

<p align="center">
  <a href="https://apexveil.github.io/#video">
    <img src="https://apexveil.github.io/assets/demos/video/drive-past-poster.jpg" alt="Real video comparison: original RGB, native SAM 3.1 masks, and ApexVeil plate replacement" width="1000" />
  </a>
  <br />
  <sub>RGB · SAM 3.1 · ApexVeil — <a href="https://apexveil.github.io/#video">watch the real 3.3-second demo</a></sub>
</p>

## What it does

ApexVeil replaces or anonymizes license plates while matching perspective, visibility, lighting and capture softness. Images and video use one deterministic compositing core; your supplied artwork stays intact.

| | Capability |
| --- | --- |
| **Replace, blur, mosaic** | Use transparent custom artwork or anonymize a selected plate. Other detected plates can be blurred. |
| **Two-stage Studio** | Run SAM once, then change the template, treatment or AI settings and regenerate from the cached observations. |
| **Scene matching** | Estimate the full plate plane separately from visible pixels, foreground protection and material appearance. |
| **Offline video** | Combine native tracking with bidirectional geometry stabilization and temporal appearance filtering. |
| **Optional AI guidance** | Request bounded geometry and appearance advice through the operator's Codex login, with explicit model and reasoning choices. |

## Quick start

Requires **Git, Conda, an NVIDIA GPU with a compatible driver, and FFmpeg**. The reference environment uses Python 3.12, PyTorch 2.10 and CUDA 12.8.

<details>
<summary><strong>Install the environment and SAM3.1 checkpoint</strong></summary>

Request access to the official [SAM3.1 checkpoint](https://huggingface.co/facebook/sam3.1), then:

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

Model provenance is pinned in [models/manifest.json](models/manifest.json). Weights, credentials and private media stay outside Git.

</details>

Start the Studio with your transparent PNG artwork:

```sh
conda activate apexveil
python -m apexveil.web --gpu 0 --port 18768 --template /path/to/artwork.png
```

Open **http://127.0.0.1:18768/**. For a remote GPU server, forward that port with ordinary SSH:

```sh
ssh -N -L 127.0.0.1:18768:127.0.0.1:18768 your-gpu-host
```

**Upload → Stage 1: detect & track → choose a target → Stage 2: compose → download.**

Stage 1 does not call an AI advisor. Stage 2 can be repeated without rerunning SAM and keeps earlier result versions. AI guidance is optional; enabling it requires your own working Codex CLI login. See the [Studio guide](docs/STUDIO.md) for model choices, timing and cache behavior.

## Current scope

**Pre-alpha, built for offline experimentation.** The Studio supports still images and continuous SDR video shots up to **8 seconds, 1080p and 30 fps**. It runs on your GPU server and is accessed through a browser; the public website is a pre-rendered showcase.

Difficult occlusion, fast directional blur and complex optical effects still need visual review. A successful render is not a guarantee of photorealism or complete anonymization. The older image/montage CLI paths have not all migrated to the Studio's scene solver.

## Documentation

| Guide | Contents |
| --- | --- |
| [Studio](docs/STUDIO.md) | Interactive workflow, reusable stages, AI settings and downloads |
| [Scene solver](docs/SCENE_SOLVER.md) | Geometry, visibility and deterministic composition |
| [Video workflow](docs/video-workflow.md) | Native tracking, temporal stabilization and current limits |
| [Image CLI](docs/IMAGE_PIPELINE.md) · [Video CLI](docs/VIDEO_PIPELINE.md) | Batch processing and developer experiments |
| [Estimator interfaces](docs/ESTIMATOR_CONTRACT.md) | Boundaries for future learned estimation modules |
| [Development](docs/DEVELOPMENT.md) · [Architecture](docs/ARCHITECTURE.md) | Project structure and contribution setup |

Run the CPU checks without downloading model weights:

```sh
python -m pytest -q
node --test tests/*.test.cjs
ruff check src tests --select E9,F63,F7,F82
```

## License

Original ApexVeil code is licensed under **[Apache-2.0](LICENSE)**. SAM source and weights retain their separate upstream license. See [third-party notices](THIRD_PARTY_NOTICES.md).

---

<p align="center">
  Built by <a href="https://github.com/ApexVeil">ApexVeil</a> · <a href="https://apexveil.github.io">Precision in motion.</a>
</p>
