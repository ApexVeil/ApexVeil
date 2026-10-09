# Architecture

ApexVeil is a standalone product with an automotive workflow as its first application.

The target core contract is a surface state: target identity, full planar geometry, visible alpha, illumination estimate, imaging blur, requested effect and confidence. Images use one state; video adds time, motion observations and correction keyframes.

Model backends propose visible regions. They do not define the product API and do not replace the separate geometry or photometric estimation stages. A visible mask must not be treated as the full plane when occluded.

The intended Web editor and API run together on one configurable loopback port. SSH port forwarding is an operator action, not a model or application dependency. Public hosting, accounts and invitations are outside the initial release.

AI review is an optional server-side provider with structured findings. API keys, Codex credentials and model selection never belong in the browser. Reviews will not silently fall back to a paid provider. The current pre-alpha does not implement the review provider or GUI.

The implemented image path is `backends/image.py` → `image_pipeline.py` →
`imaging/{geometry,appearance,focus,compositor}.py`. Visible masks and quadrilaterals
are distinct values. An ambiguous target or under-constrained plane produces
`needs_review`; rendered previews remain `rendered_unreviewed` until a separate
review occurs. See [image pipeline](IMAGE_PIPELINE.md) for operational details.
