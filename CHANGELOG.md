# Changelog

## 0.3.2 — 2026-08-17

- Install `setuptools` explicitly in the clean release builder before checking
  the tag against package metadata.
- Publish the reviewed PyPI-first documentation and notebook recipes from a
  fresh immutable release tag after the v0.3.1 tag workflow stopped safely
  before building or uploading artifacts.

## 0.3.1 — 2026-08-17

- Prepare the paper and researcher-onboarding improvements for PyPI packaging.
- Make the maintained tutorials and runnable Kaggle recipes install the pinned
  PyPI distribution instead of cloning or importing FoldPipe from a checkout.
- Make the 1QLX case study standalone by downloading the public structure from
  RCSB PDB.
- Keep benchmark provenance by embedding benchmark drivers and manifests while
  loading the library itself from the immutable PyPI artifact.
- Validate distributions with `build` and `twine` in CI before tag-driven
  trusted publication.
