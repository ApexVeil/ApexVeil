# Scene solving and bounded Codex guidance

The experimental scene solver addresses cases where segmentation is usable but
replacement shrinks inside the original surface, treats an occluder as a plate
edge, or rejects a plane because part of it is outside the image. It is an opt-in
developer API and evidence-pack workflow; existing image/montage CLI commands
have not all migrated to it. A personal upload/selection interface is now available via [Studio](STUDIO.md);
it is not a public multi-user editor.

## Shared core

```python
from apexveil.scene_pipeline import solve_scene
from apexveil.imaging.compositor import load_template

output, report, candidates = solve_scene(
    source_rgb,                  # H×W×3 uint8, original pixel coordinates
    native_visible_mask,         # H×W observation, not the complete plane
    load_template("artwork.png"),
    foreground_mask=None,        # optional independent H×W foreground evidence
    geometry_guidance=None,      # source-coordinate candidate quads / edge labels
    appearance_guidance=None,    # bounded linear-RGB / canonical-UV parameters
)
if not report["rendered"]:
    print(report["unresolved_reason"])
```

Geometry proposals distinguish real boundaries, viewport clipping and occlusion.
Complete-plane corners may lie outside the source image. The asset aspect is a
weak declared hypothesis, not a recovered physical measurement. The geometry
fit can omit negligible isolated mask speckles; the raw SAM observation remains
unchanged in both reports and previews.

The compositor separately tracks old-surface removal, transparent artwork alpha,
and foreground protection. An inferred full plane alone is not permission to
paint over unknown foreground. No default inward shrink preserves the old
colored rim. The renderer matches gain, spatial RGB veil, shading, blur and
noise while keeping the approved artwork pixels deterministic.

For video, `select_scene_trajectory` chooses among per-frame hypotheses before
rendering. Supply the selected candidate to `solve_scene`; explicitly passing
`selected_candidate=None` means unresolved and does not restart framewise
selection. Same-shot anchors can supply clearer plane evidence. Passing
`TemporalPhotometry.filter` as `appearance_parameter_filter` smooths global
appearance levels while preserving the current spatial veil residual and shadow
field. The callback runs after optional appearance guidance, exactly once.
This is not yet a universal tracker or a fully automatic montage processor.

## Optional advisor

The operator needs a working `codex` executable and their own existing login.
No API key is injected and no paid provider/model fallback is attempted. The
profiles request `gpt-6-luna`, `gpt-6.1-sol` or `gpt-6-astra`, with low, medium,
high, xhigh or max reasoning. Availability is account/runtime dependent: CLI
0.157.1 on the development host accepted Luna but rejected Sol. After the
operator upgraded to 0.162.1, real image-guidance requests for Luna and Sol at
medium reasoning succeeded with the existing ChatGPT login. Earlier App-subagent
Sol comparisons remain separate provenance.

A source-bound case pack contains native masks, full/context/tight source images,
a coordinate ruler, an approved template, and (for video) a nearby clearer frame
and a temporal strip. `scripts/build_scene_cases.py` builds the project's private
six-case development pack from existing caches; that private media is not shipped.
It is a benchmark utility, not a general upload importer.

```sh
# Deterministic scene candidate from an existing case pack.
python scripts/solve_scene_case.py --case /path/to/case --output outputs/candidate

# Up to two repairs, followed by review of the actual final render.
python scripts/refine_scene_case.py --case /path/to/case --output outputs/refined \
  --profile luna-medium --max-repairs 2
```

`--foreground` accepts separately estimated foreground evidence. `--proxy` is an
optional operator-supplied network route. Neither is hardcoded into product
source. `--initial-advice` can start a reproducible run from a saved recipe.

Each proposal is closed-schema data: edge labels, up to three complete-plane
quads, an optional physical-aspect assumption, and bounded exposure/tint/veil
parameters. Geometry coordinates are tight-crop pixels; the adapter maps them
to source pixels. Lighting control points use complete-plane UV in [0,1], with
linear RGB values. These are different coordinate systems. A null appearance
parameter uses automatic estimation; proposals are complete recipes rather than
unbounded edits to renderer source.

The loop renders before asking the model to assess a candidate, records the
actual image hashes and requested model, and stops on approval, missing target,
unavailable advisor or exhausted repair budget. A model-approved candidate is
`model_reviewed_candidate`, while `quality_accepted` remains false: self-review
is not independent ground truth. Unresolved targets and exhausted jobs return a
nonzero CLI exit status, preserving diagnostic files. Every attempt is retained;
there is no hidden return of an unchanged original as successful replacement.

## Development evidence and limits

The current six-case check includes a sunset image, a clipped still, a
person-occluded still and three video frames covering undersized replacement and
left/right clipping. The clipped cases now produce complete-plane replacements
instead of being skipped. An additional SAM person mask plus scene guidance
substantially improves the occluded case; that mask is an estimate, not a label.
The pure automatic sunset lighting remains visibly weaker than the guided
candidates. One Luna proposal/render/review cycle ran on each of the sunset and
occluded images without case-specific feedback text. This is evidence of a
working loop, not general success across lighting and occlusion.

Short temporal windows check the same renderer with preselected geometry. They
do not establish full-70-second or unseen-scene quality. Tests protect coordinates,
foreground pixels, raw masks, rendering scope, temporal parameter behavior and
bounded advisor execution. They cannot certify natural appearance or privacy.

No learned estimator was trained for this change. Future learned modules can
replace geometry/appearance estimation behind the same renderer contracts. Broad
shot coverage, independent visual acceptance, automated foreground selection and
editor integration remain work to do.
