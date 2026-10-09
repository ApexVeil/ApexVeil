# Bounded video composition experiments

`python -m apexveil.video_pipeline` processes one already-selected continuous shot
of at most four seconds. It is an experimental entry point, not the long-video
editor. It calls the same `replace_surface` and redaction functions as the image
pipeline and exports original, framewise-composition, and temporal-composition variants.
Both edited variants use the same SAM3.1 temporal masks. The framewise baseline
independently estimates geometry and appearance; it does not independently run
a text detector on every frame. This baseline is an ablation, not the intended
production path.

```sh
conda activate apexveil
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -m apexveil.video_pipeline \
  /path/to/source.mp4 --start 240 --duration 2 \
  --template /path/to/custom-rgba.png --output outputs/video-example
```

A video with multiple detected IDs requires `--target-id`. Other detected tracks
are blurred. Use `--exclude-ids` only after checking false detections; excluded
tracks are left unchanged. No automatic identity classifier decides which car
belongs to the operator.

## Motion and visibility

Each frame supplies a visible SAM mask and an independent image-geometry
observation. The shared template-aspect, small-surface, clipped-surface and dark
appearance gates apply before accepting that observation. A tight mask around
an occluded fragment is not sufficient evidence of a full planar surface.

The temporal estimator tracks textured visible surface features with pyramidal
Lucas–Kanade optical flow. It rejects failed forward/backward tracks, points
outside the image, and points no longer inside the **current** visible mask.
RANSAC fits a homography; insufficient homography support can use a similarity
fallback. Convexity and area guards reject implausible motion.

The motion-predicted plane and the observed plane describe the same current
frame. They are fused using flow confidence; this is not a screen-coordinate
moving average that trails a moving car. Large disagreement, lost geometry,
possible cuts/flashes, and short flow-only predictions remain explicit in frame
reports. Prediction is bounded to five frames without reliable observations.
The appearance state is reset after lost geometry or a possible cut/flash.

Visibility always comes from the current mask. Neither corner prediction nor
rectification fills foreground occluders in the compositing mask. This cannot
protect a foreground object that the segmenter incorrectly included as plate.

## Appearance

The current plane is rectified to template coordinates. Invisible pixels are
excluded and filled only in the separate appearance reference; they never become
visible replacement pixels. Less than 50% visible reference coverage causes a
skip. The shared compositor additionally applies its own stricter appearance
reliability gates and retains current-frame context information.

A short canonical-reference history suppresses small fluctuations. Broad
lighting changes above an experimental threshold use a faster current-frame
response, so a real moving shadow is not frozen merely to reduce variance.
Isotropic softness has its own short state. This is not a physically recovered
lighting model, directional motion blur, or soft foreground matting.

## Timing, audio, and output

The experiment explicitly exports SDR **CFR** at the source's nominal
`r_frame_rate`; an imprecise aggregate `avg_frame_rate` is not used to determine
an extra frame. The selected segment is re-encoded for the experiment. Native
VFR timestamps and HDR processing are not implemented. Each frame records both
output time and `requested_start + frame_index / fps` source sample time.

Audio uses output-side seeking and stream copy. Input-side keyframe seeking can
retain audio preroll and is unsuitable here. Extracted AAC packets are checked
against the original file's hashes within the requested **absolute PTS interval**.
Every final MP4 is probed to verify the encoded frame count and that audio packet
content remains a contiguous copy of that extracted interval. Synchronization is
packet-level, not sample-accurate; reports include the first packet's actual
source PTS and offset. Unsupported audio muxing fails rather than silently
transcoding.

Outputs include:

- `original.mp4`, `per-frame.mp4`, `temporal.mp4`.
- `comparison-detail.mp4` and `comparison-contact.jpg` for enlarged inspection.
- `frames.jsonl`: visibility, observed/fused corners, uncertainty, LK inliers,
  forward/backward error, appearance response, softness and source-time mapping.
- `report.json`: source and implementation hashes, timing policy, track selection,
  manual exclusions, frame counts, audio verification and measured timings.

Rendered frames remain `rendered_unreviewed`; a lack of heuristic warnings does
not constitute visual acceptance. Flow-relative residuals are internal
consistency diagnostics, not ground-truth insertion error or proof of stability.

## Reusing tracking

`--tracking-cache` requires `--cached-clip` and an `input-provenance.json` binding
both clip and original source hashes to the requested interval. New tracking jobs
write it automatically. Cache names alone are insufficient. A cached experiment
records that inference did not run in that job; historical tracking time is not
reported as newly incurred model time.

## Remaining work

Long-job recovery, native VFR/audio editing, shot-level identity transitions,
complete occlusion/reappearance evaluation, directional motion blur, optical
matting, OCR privacy verification, and automatic AI acceptance remain outside
this bounded experiment. Review rights before redistributing any example video;
user-supplied test clips are not included in the public repository.
