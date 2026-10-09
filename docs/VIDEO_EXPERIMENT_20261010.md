# Video composition development experiment — 2026-10-10

Two continuous shots from a user-supplied 720p review video were processed on
H20. The clips and their outputs remain private. They are development examples,
not a held-out benchmark or a claim that the full 70-second montage works.

The shared compositor uses the appearance fixes from `f5e99b6` plus the diagnostic
serialization correction in `9487cc4`. Every final report records exact source
and implementation-file SHA-256 values. No Codex or other model review was run.

## Results

| Measure | Front driving shot | Tree-shadow approach |
|---|---:|---:|
| Original interval | 04:00.000–04:02.000 | 04:37.500–04:40.000 |
| CFR frames / rate | 60 / 30 fps | 75 / 30 fps |
| Frames with a rendered temporal replacement | 60 | 69 |
| Motion-fused frames | 59 | 40 |
| Frames marked `needs_review` | 0 | 32 |
| Frames skipped for insufficient geometry | 0 | 6 |
| LK points rejected by current visibility | 13 | 84 |
| Render plus intermediate PNG time | 44.35 s | 45.51 s |
| Complete cached-mask experiment | 59.23 s | 68.30 s |

The timing includes two rendered variants, intermediate PNGs, comparison images,
and four encoded MP4s. It excludes new SAM inference because both final jobs
reused content-verified masks. These are diagnostic export timings, not
production throughput or real-time performance. Both jobs ran concurrently on
CPU after tracking, so these numbers are not an isolated latency benchmark.

The front shot kept one tracked target and rendered every frame. Static contact
inspection showed the previous green/purple plate contamination removed by the
shared appearance fixes. Absence of heuristic flags does not equal visual
acceptance, and no learned reviewer accepted the clip.

The tree-shadow shot is a **retained failure example**. The vehicle approaches,
changes angle and passes through moving shadows. A second SAM track was a bright
bumper feature, not a plate: target 0 was selected and false track 1 manually
excluded. This manual intervention is recorded. The visible-mask/plane agreement became unreliable under the changing
appearance. That quality flag alone does not establish actual foreground
occlusion or a bad segmentation.

Replacement was skipped at frames 43, 44, 60, 61, 62 and 63 (source times about
04:38.933–04:38.967 and 04:39.500–04:39.600). Those frames retain the original
plate and therefore produce visible discontinuities. This clip must not be
presented as finished privacy redaction or production-quality replacement.
Twenty-five frames used bounded flow-only geometry predictions and remained
flagged. Lowering gates to hide these failures was deliberately avoided.

The temporal appearance state used its fast light-change response on 19 tree
frames and none of the more stable front frames. It still uses heuristic
illumination and current-frame context; this is not proof that moving hard
shadows, glare or all color flicker have been solved.

## Geometry diagnostics

Post-run analysis of the 30 coverage-flagged tree frames found a median internal
mask-hole fraction of zero. Missing coverage was primarily between the external
mask contour and the fitted quadrilateral. Fifteen frames were already below
0.88 coverage after `fit_quad`; fifteen additional frames crossed the threshold
only after image-gradient refinement. Median coverage changed from 0.880 before
refinement to 0.831 after it, with a median maximum corner shift of 2.76 px.
This points first to low-resolution quad expansion/edge attraction, rather than
assuming foreground holes or immediately retraining SAM. The fixed 1 px line
allowance is a candidate for a later ablation on roughly 20 px-tall plates; no
threshold or geometry change was made to hide the failures in these outputs.


Median motion-relative corner residuals changed from 0.524 to 0.106 px in the
front clip, and 1.685 to 0.358 px in the tree clip. These measure agreement with
our own flow prediction. The fusion formula itself reduces this residual, so the
numbers are **not ground-truth alignment errors or independent quality scores**.
The actual observations, predicted transform, inlier counts and per-frame flags
are preserved for inspection.

Forward/backward feature checks alone are insufficient for new occlusion. The
implemented flow filter also rejects destinations outside the image or outside
the current visible mask. A regression test ensures perfectly matching but
newly invisible texture cannot continue to define a hidden plane.

## Frame and audio verification

Every original/per-frame/temporal/comparison MP4 was verified with ffprobe:
exactly 60 frames and 2.0 s, or 75 frames and 2.5 s. The source's slightly
non-integer aggregate average rate had previously produced an extra sample;
explicit nominal 30 fps and output frame limits corrected it.

Audio is copied AAC, not regenerated. Output-side seeking avoids keyframe
preroll. Extracted packet payload hashes were independently matched against the
source video's absolute PTS interval before muxing:

- Front: 87 source packets, first source PTS 240.001678 s; first-packet offset
  1.678 ms. Final outputs retain 86 unmodified packets after bounded muxing.
- Tree: 108 source packets, first source PTS 277.501995 s; first-packet offset
  1.995 ms. Final outputs retain 107 unmodified packets after bounded muxing.

Final encoded audio packet hashes match the copied source prefix. The final
boundary is packet-level, not sample-accurate editing. Native VFR timestamps,
HDR, full soundtrack editing, and long-video recovery remain unfinished.

## Reproduction and next work

Use the bounded command in [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md), selecting
`--target-id` and inspected `--exclude-ids` where needed. No test video is
redistributed in this repository. The first priority after these examples is
better visible-mask/complete-plane handling and correction keyframes for the
failed tree intervals, followed by temporal appearance validation against new
shots. A long edited montage should not be processed as one continuous track.
