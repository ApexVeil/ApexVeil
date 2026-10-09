# Development experiment — 2026-10-09

This is a small development regression set, not a held-out quality benchmark.
Experiments ran on one explicitly selected NVIDIA H20 using the standalone
`apexveil` environment and pinned SAM3.1 model. No LLM, Photoshop, fine-tuning,
private legacy runtime imports, or second machine's model runtime was used.
Private images, templates, weights and generated media are not in Git.

## Still images

Nine 6000×4000 user-selected photos cover darkness, defocus, foreground occlusion,
multiple vehicles, glare and partial out-of-frame plates. The detector found at
least one candidate in every image, with 12 candidates in total. Visual inspection
confirmed the three targets in the multi-vehicle scene, including a distant
motorcycle. These counts are not measured detection precision/recall: there are
no independently annotated pixel or instance ground truths.

The first GPU batch saved masks and original/replacement/blur/mosaic comparisons.
Full-frame plus automatic crop segmentation took 0.31–0.67 s after the first
image; median warm time was 0.317 s. The first image took 0.959 s for segmentation.
Exporting all three full-resolution lossless PNGs plus diagnostics made the
complete per-image job take 20.9–39.1 s. Do not advertise segmentation latency as
end-to-end export latency.

The second compositor iteration reused those verified masks (no new GPU
inference). It added automatic isotropic softness estimation and a conservative
shape guard so a visible sliver is not stretched into a complete wide design.
Primary targets were manually selected in the two multi-target photos. No manual
corners were supplied.

| Case | Final replacement behavior | Outstanding limitation |
|---|---|---|
| Dark front view, `0076` | Rendered, unreviewed | Relighting is heuristic |
| Backlit view, `0092` | Rendered, unreviewed | No calibrated material model |
| Strong defocus, `0093` | Softened preview, `needs_review` | Edge estimate about 8.5 px; no optical matting |
| Motorcyclist obscuring car, `0100` | Replacement skipped; detected other plate blurred | Complete hidden plane needs manual corners or temporal evidence |
| Car and two motorcycles, `0103` | Selected car replaced, both others blurred | Selection was manual, not automatic semantic identity |
| Severe darkness, `0185` | Detected, replacement skipped as `needs_review` | Uncertain appearance in near-black pixels |
| Sun glare, `0205` | Rendered, unreviewed | Lens flare is not separated from material reflection |
| Plate cut by image edge, `0287` | Replacement skipped as `needs_review` | Complete plane lies outside the frame |
| Person obscuring car, `9998` | Replacement skipped as `needs_review` | Visible fragment is insufficient for complete geometry |

All three modes changed zero working-buffer pixels outside the supplied visible
masks in the measured first batch. The renderer retains foreground mask holes.
This property cannot detect an occluder incorrectly included in a SAM mask.
Blur/mosaic are not certified anonymization; OCR unreadability was not measured.

## Short video diagnostic

The supplied review video's continuous 04:00–04:02 segment was extracted as 60
frames at its nominal 30 fps and original 1280×720 dimensions. SAM3.1 propagated
one target ID through all 60 frames; a per-frame quadrilateral diagnostic was
exported. The initialization response and propagation both contain frame zero;
metrics below deduplicate by actual frame index.

- Model load: 15.38 s.
- Prompt/propagation plus mask export: 8.13 s.
- Peak allocated GPU memory: 21.67 GiB.
- Unique frames with a target: 60/60; target IDs: one.

This was **tracking and geometry only**. No pixel replacement, temporal corner
smoothing, blur consistency, occlusion/reappearance benchmark, sound handling or
final-video quality evaluation was performed. A stable ID is not proof of a
stable photorealistic insertion.

## CPU checks

14 tests pass without downloading model weights. Coverage includes alpha
silhouette/opacity, unique quadrilateral corners, preservation of mask holes and
outside pixels (including defocused rendering), multi-target policy, uncertain
and clipped geometry, source integrity/ICC export, and recovering known synthetic
edge softness. Fatal-error lint and `pip check` pass.

## Next experiments

1. Corner/prompt correction for obscured and clipped surfaces, with explicit full
   geometry and visible alpha.
2. Moving hard shadows and glare: do not transfer old characters as lighting.
3. Optical edge matting and directional motion blur.
4. Continuous-shot video composition with temporal geometry/appearance controls.
5. A held-out vehicle/day/source split and human whole-clip acceptance scores.
