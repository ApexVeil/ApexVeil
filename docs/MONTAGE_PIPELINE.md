# Shot-aware full-timeline comparison

The montage runner extends the experimental image/video core to an edited
sequence with a reviewed shot manifest. It does not treat a montage as a single
track and does not reset every arbitrary four-second window. It currently
exports three native 1280×720 panels at 30 fps with a separate 48-pixel title bar.

The delivered comparison uses:

- **RGB:** the unedited frame from the lossless RGB reference.
- **SAM3.1 + Independent:** native per-shot SAM3.1 masks, followed by independent
  frame geometry and appearance composition.
- **ApexVeil:** the same masks and current shared compositor, with motion and
  fitted-photometric temporal state.

SAM initialization is a text prompt at frame zero of each actual shot, followed
by its native forward propagation. There is no per-frame text prompt, point
conversion, Photoshop, or AI acceptance reviewer.

## Reviewed timeline

`shot_manifest.json` contains contiguous half-open frame intervals covering the
entire reference. No time is omitted. Hard cuts reset state; foreground passage
and fast camera motion alone are not cuts. A dissolve is recorded separately
because overlapping shots cannot reliably be represented as one opaque plane.
Reviewed no-plate views, black frames and unresolved transitions retain their
original pixels and timing. A fade-in can remain part of its following continuous
shot, with native detection continuing as objects become visible.

Every tracked shot requires operator review of target IDs and exclusions before
rendering. `primary_ids` may manually associate two IDs known to belong to the
same vehicle: one current visible mask is chosen, masks are never blindly
unioned, and this alias transition does not reset temporal geometry. False
bumper/trim detections can be explicitly excluded. Other non-excluded detected
plates are blurred. A background-only view can have no primary target while
still blurring a detected plate on another vehicle.

## Run stages

The current deliverable root contains a 2100-frame lossless RGB reference,
reviewed manifest, individual lossless shots, tracking masks, candidates, and
frame diagnostics. An example stage sequence is:

```sh
conda activate apexveil
python -m apexveil.montage_pipeline prepare --root /path/to/deliverable
CUDA_VISIBLE_DEVICES=0 python -m apexveil.montage_pipeline track \
  --root /path/to/deliverable
# Review candidate sheets and target/exclusion fields in shot_manifest.json.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m apexveil.montage_pipeline render \
  --root /path/to/deliverable --template /path/to/artwork.png --workers 8
python -m apexveil.montage_pipeline finalize --root /path/to/deliverable \
  --template /path/to/artwork.png
```

`render --only-ready` supports incremental work while remaining candidate
selections are reviewed. Finalization still requires every segment. CPU workers
stream RGB panels directly to FFmpeg; they do not save multiple full-resolution
PNGs for every frame. Cached renders include selection, template, tracking,
renderer and shared-core hashes, so a changed implementation or selection cannot
silently reuse an obsolete chunk.

## Native propagation boundary contract

The pinned upstream processing order clamps an inclusive end, but its batched
grounding feature cache interprets `max_frame_num_to_track` as an exclusive count.
Full-shot requests therefore pass **T**, the number of frames, not T−1. T−1 can
produce an empty final feature batch and a tensor-size exception, even though
processing order still visits the final frame. A regression test covers short
shots and batch-boundary lengths. The upstream source remains unmodified.

## Export verification and limitations

The 04:00–05:10 deliverable is required to contain exactly 2100 frames, 70 seconds,
and three 1280×720 panels. Shot-frame journals are checked for gaps and overlap.
AAC is copied using output-side seeking and its packet payloads are checked
against the requested absolute source-PTS interval. Synchronization is packet
level, not sample-accurate audio editing. The comparison uses explicit CFR;
native VFR and HDR are outside this export contract.

Every frame records replacement status, warnings, selected/associated IDs,
geometry and appearance diagnostics. Native failures and unreliable geometry
retain their place in the timeline. The result is not certified anonymization or
photorealistic acceptance; an absent mask can mean true absence or a missed plate.
`latest.json` points to the current result and source/core provenance. User media,
model weights, raw results and test-video rights are not redistributed with code.


## 2026-10-10 complete-sequence evidence

The reviewed 04:00–05:10 example contains 38 segments: 30 native-tracking shots
and 8 no-plate/black/dissolve passthrough segments. The longest continuous shot
is 4.4 seconds; it is tracked as one shot. Fresh tracking completed without native
errors after correcting the frame-count request contract above. The failed first
attempt is retained privately and was not used in the delivered comparison.

The final file has 2100 decoded frames, video and container duration 70.000 s,
and dimensions 3840×768. First/last decoded PTS are 0.000000 / 69.966667 s. The
largest error against the exact 30 fps timestamp grid is approximately 0.33 μs,
within ffprobe's printed decimal precision; no final timestamp normalization was
needed. Source AAC payloads were verified before and after muxing: 3015 packets
in the requested source interval, 3014 unchanged packets in the bounded final
file. Audio boundary precision remains packet-level.

ApexVeil produced primary-surface replacement on 1255 frames; the independent
composition produced it on 1199. There are 389 `needs_review` frames, including
small/extreme-angle plates and absent/undetected targets; this is not a labeled
error rate. Other retained time includes 306 reviewed no-plate frames, 38 black
frames, 18 dissolve frames and 143 background-only frames with no detections.
Every one of the 2100 frames is present, including failures and original pixels
where a replacement was skipped. Rendered previews are not automatically
accepted or certified private.

The artifact's base core is `063d93c`. All chunk selections, manual target-ID
aliases, exclusions, tracking reports, approved template and shared-core hashes
were checked against the current manifest before publication of `latest.json`.
Private test-video media is not included in Git or the public project website.
