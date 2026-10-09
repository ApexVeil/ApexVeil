# Roadmap

## Implemented foundation

- Independent Python package, Conda environment and pinned SAM3.1 source/model provenance.
- Environment diagnostics, integrity verification, bounded real GPU segmentation smoke tests.
- Private evaluation data layout and public project identity.

## Implemented experimental image core

- Independent geometry, local appearance, visibility-aware compositing and image CLI.
- Transparent templates, per-target replacement/blur/mosaic, and blur for other selected candidates.
- Explicit multi-target selection, corner overrides, uncertainty flags and local diagnostics.
- Isotropic edge-softness estimate, with strong defocus flagged for review.

## Image editor (next)
- Browser upload, click/stroke prompts, undo, adjustable corners, diagnostic layers and export.
- Add server-side Codex review after the deterministic core is reproducible.

## Video validation and editor

- Shot boundaries, stable target identities, forward/backward propagation and keyframe corrections.
- Temporal geometry, partial/full occlusion, reappearance, defocus, motion blur and moving shadows.
- Proxy/original coordinate mapping, original timestamps, audio preservation and resumable jobs.
- Evaluate the supplied driving sequence without claiming that it is one continuous shot.

## Public service

Only after output quality and workflow are validated: invitations, isolated user projects, budgets, operational monitoring and a hosted service.
