# Personal remote Studio

The loopback Studio provides upload → Stage 1 native segmentation/tracking →
target selection → Stage 2 rendering → preview/download. It runs on the GPU server.
The local browser does not load SAM or need a Python environment. This is a personal test interface,
not the invitation/public hosting release.

```sh
conda activate apexveil
cd /path/to/ApexVeil
python -m apexveil.web --gpu 0 --port 18768 \
  --template /path/to/approved-transparent-artwork.png \
  --advisor-profile sol-medium
```

Add `--proxy http://127.0.0.1:PORT` only if that server needs an operator-configured
route for Codex. `--advisor-profile` sets the default selection; `--advisor-profile off` disables AI.
When enabled, the browser offers GPT-6 Luna, GPT-6.1 Sol and GPT-6 Astra with
Low, Medium, High, Extra High and Max reasoning. The default is Sol / Medium.
Selections are saved with each render revision. Model, reasoning, AI toggle,
template, treatment and selected target can change before or after completion
without repeating Stage 1. Login tokens and network configuration stay on the server.

On the client, use ordinary SSH forwarding:

```sh
ssh -N -L 127.0.0.1:18768:127.0.0.1:18768 your-gpu-host
```

Open `http://127.0.0.1:18768/`. The HTTP server binds only to loopback and uses one
GPU worker. The first job loads the model; later jobs reuse it. Files and results
stay under `outputs/studio/` unless `--output` changes that directory. They are
not automatically deleted. Finished tasks, uploaded assets and completed Stage 1
caches are restored after a server restart. Images restore native plate/person masks; videos restore their
lossless prepared clip, tracking records and native per-frame masks. Source hashes
and video interval/clip consistency are verified before reuse. A completed Stage 1
needs no SAM model load for Stage 2, including after a restart. Interrupted Stage 2
renders are not automatically retried and their Stage 1 cache remains available.
Incomplete Stage 1 jobs require another detection. The job URL restores the browser
view across refreshes.

## Workflow

1. Upload an image or video. For videos choose the start time and duration.
2. Run **Stage 1 · SAM detection/tracking**. No AI or template is needed. The
   immutable source interval and raw masks are cached on disk. Image preparation
   also caches person detections so later protection toggles do not rerun SAM.
3. Choose the target from the native-mask preview or candidate buttons. Video
   targets may use different representative frames. Other detected plates are blurred.
4. Set Replace/Blur/Mosaic, the template, optional AI model/reasoning and image
   foreground protection. Run **Stage 2 · scene processing** and inspect the result.
5. Change any Stage 2 setting and regenerate directly. Each submission freezes its
   settings and writes a separate `jobs/<id>/renders/<revision>-<id>/` directory.
   The result version selector previews/downloads earlier successful revisions;
   failed later attempts keep earlier downloads and the Stage 1 cache.
6. A different source or video interval requires a new Stage 1 job. The browser
   clears the current selection when these change, and the server rejects a
   mismatched source/interval in a render request. Source content changes also
   invalidate its cache.

Legacy image jobs that lack person masks receive a one-time Stage 1 foreground
cache supplement when protected replacement is first requested. Their original
plate masks remain unchanged. Newly prepared jobs need no such migration.

The image branch reuses the separately cached person masks when foreground
protection is enabled. These masks are predictions, not ground truth. The current
video interface is limited to a continuous shot of up to eight seconds; it does
not silently apply one geometry/light recipe across cuts. Long-montage shot
selection and complete 70-second quality acceptance remain separate work.

## One AI call versus review

Each Stage 2 generation with AI assistance makes at most one Codex request per
selected image/continuous shot for geometry and bounded appearance guidance, then performs deterministic
rendering. It is not a second inspection of the new output. A proposal cannot
certify its own not-yet-rendered result. The older `refine_scene_case.py` developer
loop remains available when a separate rendered-result review and bounded retries
are wanted; that mode can make multiple calls.

Video continues to estimate each frame from native temporal masks, choose a
coherent geometric path and filter appearance over time. A representative-frame
quad is not pasted into unrelated image coordinates. A static AI UV light field
must not override a moving shadow or flare throughout the clip.

With Codex CLI 0.162.1, the development host successfully requested both
`gpt-6-luna` and `gpt-6.1-sol` at medium using the existing ChatGPT login. The older
Sol rejection applied to CLI 0.157.1. Availability is still checked by the actual
request; failures produce warnings, with no paid provider/model substitution.

## Timing and acceptance

The photo overlay shows estimated progress, capped at 99% until the current phase
actually finishes; failed requests never display successful completion. The
elapsed timer is measured wall time. Hover or focus the information icons for help.

The result records current-job cold-load time, cached Stage 1 preparation, the
current Stage 2 wall time, render revision/settings, queue time and processing wall time.
The UI emphasizes the current generation's rendering time; cached detection time
is not counted again in later revisions. Processing wall time excludes
the user's time spent choosing a target. A warm worker and a cold process are
not comparable. Large photos, video resolution, duration, candidate count and AI
context all affect latency.

The experimental renderer improves clipped planes and old-color coverage, but
warm optical glare, difficult occlusion, small edges and temporal brightness
changes still require visual inspection. Successful tests, completed rendering
and an AI assessment are distinct from natural-looking output.
