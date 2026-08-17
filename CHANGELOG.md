# Changelog

## 0.3.1 — 2026-08-17

- Publish the paper and researcher-onboarding improvements as an installable
  PyPI release.
- Make the maintained tutorials and runnable Kaggle recipes install the pinned
  PyPI distribution instead of cloning or importing FoldPipe from a checkout.
- Make the 1QLX case study standalone by downloading the public structure from
  RCSB PDB.
- Keep benchmark provenance by embedding benchmark drivers and manifests while
  loading the library itself from the immutable PyPI artifact.
- Validate distributions with `build` and `twine` in CI before tag-driven
  trusted publication.
