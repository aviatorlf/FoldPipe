# FoldPipe plagiarism, provenance, and novelty audit

**Audit date:** 2026-08-17  
**Code release examined:** `v0.3.0`, merge commit
`20c22c80998dd432c6c92ae3a9aebbfacaee6db2`  
**Benchmark source commit:**
`16fdbb26b00f9721ce4034335ce0ee12bda77720`

## Decision

No concrete evidence of copied prose or source code was found in the materials
examined. It is reasonable to begin a paper, but only with deliberately narrow
novelty language.

The largest publication risk is **not textual plagiarism**. It is presenting a
long-established systems mechanism--prefetching the next item while computing
on the current item--as a new algorithm. The draft avoids that error and treats
prefetching, bounded buffering, and the asymptotic I/O/compute crossover as
prior art.

This is an engineering and literature audit, not a legal opinion or a guarantee
that a journal's proprietary similarity checker will find no match.

## Publication-ethics standard used

The audit follows the distinctions in the Committee on Publication Ethics
([COPE plagiarism discussion paper](https://publicationethics.org/files/COPE_plagiarism_discussion_%20doc_26%20Apr%2011.pdf)):
possible misuse is not limited to identical prose and can involve ideas, data,
images, or other original material; assessment depends on the extent,
originality, context, and quality of attribution. The separate
[COPE text-recycling guidance](https://members.publicationethics.org/sites/default/files/Web_A29298_COPE_Text_Recycling.pdf)
also makes clear that reuse from an author's own publication must be assessed
for amount, location, disclosure, originality, and possible duplicate
publication. Accordingly, this audit treats four questions separately:

1. Is source code or prose copied without attribution?
2. Are established ideas presented as new?
3. Are earlier results or text being republished without disclosure?
4. Are data and figures properly attributed and licensed?

## What was checked

### Repository provenance

- Inspected the full Git history and the history of `foldpipe/loader.py`.
- Substantive commits are attributed to Dhiren Khatri; merge/release commits are
  attributed to the repository owner account.
- Inspected non-notebook source for third-party copyright headers, copied-from
  markers, license notices, and embedded source URLs.
- The repository carries an MIT license naming Dhiren Khatri.
- No vendored third-party source files or unexplained copyright headers were
  found in the package implementation.

### Distinctive-language and public-code searches

Exact public-web and GitHub code searches were run for distinctive phrases,
including:

- `True O(1) bounded-memory asynchronous streaming dataloader`
- `Kick off prefetch for Chunk N+1 immediately`
- combinations of `FoldPipe`, `SyntheticLatencySource`, and native PyTorch
  shards

No external exact match was returned. This lowers the likelihood of verbatim
reuse in indexed public material, but it cannot test private repositories,
paywalled full text, student submissions, or proprietary publisher corpora.

### Conceptual prior art

Primary papers and official documentation were reviewed across five groups:

1. classic file-system prefetch and compute/I/O overlap;
2. general ML input-pipeline systems (`tf.data`, CoorDL, Cachew, FFCV, DALI);
3. sharded/streaming formats and services (WebDataset and Hugging Face
   streaming);
4. graph-specific out-of-core systems and PyG storage/loading APIs; and
5. molecular-data/model provenance (MD17, SchNet, and PyTorch Geometric).

The related-work map used for the paper is recorded in `references.bib`.

## Claim-by-claim boundary

| Proposed claim | Nearest prior art or fact | Safe wording | Wording to avoid |
| --- | --- | --- | --- |
| Prefetching overlaps I/O and computation | File-system prefetch literature; `tf.data`; PyTorch/PyG loaders; FFCV | “FoldPipe applies one-stage asynchronous prefetch to remote molecular shards.” | “FoldPipe invents asynchronous prefetch.” |
| Ideal homogeneous runtime approaches `max(D,C)` | Standard two-stage pipeline model; the benefit of overlap is decades old | Present the equation as an explanatory model with citations. | Call the equation a new theorem or algorithm. |
| Memory is independent of total dataset size | A constant number of shards are live, assuming bounded shard size | “The implementation maintains a constant-number shard working set, up to current and prefetched payloads.” | “Uses exactly one shard,” “zero memory,” or “cannot OOM.” |
| Native `.pt` use avoids format migration | WebDataset uses tar shards; PyG `OnDiskDataset` uses a database backend; HF can stream arbitrary files | “Avoids a separate tar/database conversion for already-sharded trusted `.pt` objects.” | “First zero-conversion streaming system” or “zero-copy.” |
| Domain relevance | MD17, SchNet, and PyG are established molecular/graph resources | “A compact integration evaluated on a SchNet energy-and-force workload over MD17-derived shards.” | “A new molecular model” or “new force field.” |
| Real-network performance | 20 paired passes; 1.0587x geometric mean; CI crosses 1 | “The run is inconclusive about a speed advantage; it directly records overlap.” | “FoldPipe accelerates MD17 training” as an unconditional finding. |
| GPU utilization | Coarse 0.5 s `nvidia-smi` samples; 36.17% vs 37.70% means | Report as descriptive telemetry. | Claim a significant utilization improvement. |
| Reproducibility | Revision, code bundle, raw traces, hashes, and order protocol are recorded | “The artifact pins the evaluated code and data revisions.” | “Perfectly reproducible” or deterministic network performance. |

## Code and artifact caveats discovered

1. **Two-shard-scale peak, not exactly one shard.** While the current shard is
   consumed, the future may hold a fully downloaded and deserialized next shard.
   The bound is constant in dataset size, but peak memory can include both.
2. **Both compared pipelines are bounded.** The rigorous MD17 baseline is a
   sequential bounded stream, not eager whole-dataset accumulation. The paper
   does not use boundedness to imply that only FoldPipe avoids total-dataset
   residency.
3. **Deserialization is neither zero-copy nor safe for untrusted data.** The
   Hugging Face source buffers bytes in RAM and calls `torch.load(...,
   weights_only=False)`. Only trusted shards should be consumed.
4. **The old whitepaper figure is not suitable evidence.** Its schematic
   traces and single-run controlled claims are not used in the manuscript.
5. **The private dataset mirror lacks a data card.** The migration code shows
   that `torch_geometric.datasets.MD17(name="aspirin")` was reshaped into 43
   `.pt` shards, but the Hugging Face repository has no README or explicit
   license field. Upstream attribution and verified redistribution terms must
   be added before submission or public release.
6. **The paper is about a systems path, not model quality.** Repeated
   structure-visits measure pipeline behavior. They are not unique molecular
   samples and do not establish predictive accuracy.
7. **Repository-facing claims must match the paper.** The earlier README title,
   “Eliminating GPU Starvation,” was broader than the v0.3.0 evidence. It was
   replaced on the paper branch with a descriptive bounded-streaming title, and
   the package description was narrowed for the same reason.

## Plagiarism safeguards applied to the draft

- All prose was written afresh; no abstract or related-work sentence was copied
  from a source.
- General mechanisms are cited at the point of discussion.
- No external figure, table, or source-code listing is reproduced.
- The included benchmark figure was generated by this repository from the raw
  release artifact and is labeled as a representative trace.
- The results section distinguishes new measurements from prior ideas.
- The same benchmark data are not represented as a new, independent dataset.

## Residual checks required before submission

- Run the rendered manuscript through the target venue's institutional
  similarity service and inspect matches manually. A similarity percentage is
  not itself a plagiarism finding.
- Review all notebooks for copied tutorial prose if any notebook will be part
  of supplementary material; this audit focused on paper-facing source and the
  production package.
- Add an author contribution statement and declare any prior public version,
  preprint, or overlapping manuscript.
- Verify the target venue's rules for text recycling and AI-assisted writing.
- Add the MD17 mirror data card, source URL, citation, transformation steps,
  and verified license.

## Final risk assessment

| Risk | Assessment | Reason |
| --- | --- | --- |
| Verbatim public-code copying | Low based on available evidence | No exact distinctive match or foreign attribution marker found. |
| Verbatim paper-text copying | Low for this new draft | Prose is newly written and cited; proprietary corpora remain unchecked. |
| Unattributed conceptual borrowing | Medium if claims drift; low in current draft | Prefetching is old and must remain explicitly credited. |
| Novelty overclaim | High without the boundaries above | Many mature systems already pipeline and prefetch ML inputs. |
| Dataset attribution/license | Medium-high until fixed | Private mirror currently has no README or license metadata. |
| Duplicate/redundant publication | Unknown | Requires the author to disclose any prior manuscript or submission. |

**Conclusion:** proceed with the paper as a modest systems artifact and empirical
case study. Do not market FoldPipe as a new prefetch algorithm or as proven to
accelerate real MD17 training.
