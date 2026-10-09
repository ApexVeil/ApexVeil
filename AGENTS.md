# ApexVeil development

ApexVeil is an independent product, not an extension of any robotics repository.

- Product and Python package name: ApexVeil / apexveil. Upstream models are adapters under third_party, not the product identity.
- Keep third_party/sam3 unmodified. Pin its commit and record any deliberate update together with model provenance and smoke-test results.
- Use the dedicated apexveil environment. Never depend on another project's environment, checkpoint directory, source tree or cache for runtime behavior.
- Images and videos share surface geometry, visibility, appearance and compositing contracts. A video adds time and identity; do not implement a second unrelated engine.
- Preserve the approved transparent template silhouette. Default automotive workflow is plate_only; do not preserve fasteners unless explicitly requested later.
- Keep visible masks separate from inferred full surface geometry. Do not fill foreground occlusion holes to obtain a compositing mask.
- The current deployment is a personal remote Web workflow. Bind to loopback by default; SSH forwarding belongs in deployment documentation, not core source.
- AI review is an optional server-side adapter. Never put model credentials in a browser, retry through a paid provider automatically, or label a failed review as passed.
- Never commit models, user media, private templates, credentials, local machine configuration, caches or generated results.
- Report tested capabilities separately from planned features. A successful segmentation smoke test is not a video-replacement quality benchmark.
- Keep GPU checks bounded to one explicitly selected device. CPU tests and lint run without model downloads.
