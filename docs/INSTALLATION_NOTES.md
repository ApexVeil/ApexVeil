# Installation notes

The initial environment was validated on Linux with an NVIDIA H20 and driver 580.105.08. The independent environment is named `apexveil`, using Python 3.12, PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128.

- SAM source is a pinned, unmodified Git submodule.
- Pin setuptools below 81 because the upstream builder still imports pkg_resources.
- The declared upstream dependency requires NumPy below 2; the reference environment uses 1.26.4 and OpenCV 4.11.
- FlashAttention 3 and compilation are disabled for the initial portable validation. Enable and benchmark them explicitly later.
- The model is gated by Meta on Hugging Face. A cached token does not imply access approval; an awaiting-review 403 is not an installation failure. Use an approved account or import an already-authorized checkpoint, then verify its SHA-256.
- `requirements/lock-linux-cu128.txt` records the initial installed dependency versions, excluding editable local packages. CUDA packages require the official PyTorch CUDA 12.8 wheel index.
- Do not copy an unrelated project's virtual environment or reference a checkpoint through an external symlink. Download/import a separate file into this project's ignored model directory.

## Pinned predictor compatibility

The pinned SAM shared predictor passes an optional state-offload argument that its multiplex model does not accept. ApexVeil adapts only the created model instance: a disabled flag is removed; enabling unsupported state offload raises an error. The vendored source is unchanged. Real-valued RoPE conversion is disabled for the initial checkpoint-compatible path. Every learned tensor is checked against the checkpoint for key presence and shape after construction.

## Fresh interactive video output compatibility

The pinned multiplex interactivity implementation returns `{}` from
`_build_sam2_output` for a frame absent from `cached_frame_outputs`, before it
merges `refined_obj_id_to_mask` supplied by that call. Fresh point-seeded sessions
can therefore discard the tracker's current prediction; examples that first
populate every frame through visual grounding may not reveal this behavior.

ApexVeil installs a narrow instance-level wrapper in `backends/sam31.py` to merge
those supplied masks on uncached frames. It does not synthesize masks, change
scores/thresholds, pre-populate a detection cache, force action history, or edit
the vendored repository. Existing cached-frame behavior is preserved. The model
load audit records the compatibility adapter. CPU regression checks cover fresh
and cached outputs, absent predictions, state preservation and repeat install.
The point-prompt experiment documents real-model validation separately.
