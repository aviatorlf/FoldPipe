# FoldPipe paper workspace

This directory contains the first paper draft for the v0.3.0 research freeze.

## Draft thesis

FoldPipe is a small compatibility and orchestration layer for consuming remote,
native PyTorch/PyG molecular-data shards with a working set bounded independently
of total dataset size. It uses a one-stage background fetch to overlap retrieval
of shard *N+1* with training on shard *N*. The prefetch principle is established
prior art; the paper's contribution is the domain-specific integration,
instrumentation, reproducible artifact, and honest characterization of a noisy
real-network regime.

The v0.3.0 MD17 experiment observed 16.33 s mean I/O--compute overlap, but its
primary paired runtime estimate was 1.0587x with a 95% bootstrap interval of
0.8776x--1.2880x. The paper therefore does **not** claim a statistically
established speed advantage.

## Files

- `foldpipe.tex` -- submission-style manuscript source.
- `references.bib` -- bibliography assembled from primary papers and official
  project documentation.
- `PLAGIARISM_AND_NOVELTY_AUDIT.md` -- provenance, text-overlap, license, and
  claim-boundary audit performed before drafting.

## Build

From this directory, with a standard TeX installation:

```bash
latexmk -pdf foldpipe.tex
```

The current development machine does not have a TeX engine installed, so the
draft is checked structurally in-repository but has not yet been PDF-compiled.

## Before submission

1. Add the author's verified affiliation and ORCID.
2. Add a README/data card and explicit upstream license information to the
   private `aviatorlf/md17-shards` mirror.
3. Run the final PDF through the target venue's template and an institutional
   similarity service such as iThenticate; public-web search is not a substitute
   for a publisher corpus.
4. Decide whether to add a multi-seed controlled D/C crossover experiment. The
   older whitepaper graphic is intentionally not used as paper evidence.
5. Confirm the target venue's disclosure policy for AI-assisted editing.

