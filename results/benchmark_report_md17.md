# FoldPipe MD17 + SchNet benchmark

- Generated: 2026-08-16T21:05:13.461132+00:00
- Hardware: Tesla T4
- Dataset: `aviatorlf/md17-shards@f779686deb9217877dd7ddde99b2522bd441492a`
- Source bundle: `0ed7e4c583621edc793234e51dafc3756221f7e2337e1d86708c1d5f25fcad39`
- Base Git commit: `f19735f10935fd44a3fbf4bfee0660adf83a111b`
- Kaggle kernel: [`dhirenkhatri/foldpipe-md17-rigorous-benchmark`, version 6](https://www.kaggle.com/code/dhirenkhatri/foldpipe-md17-rigorous-benchmark)
- Protocol: 10 paired, order-alternating passes; 5 pinned shards per pass; batch size 32
- Warm-up: one untimed training batch from the first pinned shard

| Metric | Sequential | FoldPipe |
| --- | ---: | ---: |
| Mean time (s) | 155.518 | 107.587 |
| 95% bootstrap CI, mean time (s) | [113.641, 203.158] | [65.444, 156.030] |
| Mean peak RSS (GiB) | 1.938 | 2.037 |
| Mean sampled GPU utilization (%) | 20.431 | 37.649 |
| Mean I/O/compute overlap (s) | 0.000 | 15.259 |
| Mean GPU wait time (s) | 128.606 | 79.596 |

Paired mean speedup ratio: **2.3122x** (95% bootstrap CI [1.2835, 3.4650]).

Paired mean time saved: **47.931 s** (95% bootstrap CI [-12.225, 103.440]). FoldPipe was faster in 70% of pairs.

The mean-ratio interval excludes 1 in FoldPipe's favor, but the additive time-saved interval includes 0. These estimands disagree under high run-to-run variability, so this result supports improved observed means and the overlap mechanism but should not be presented as a universal speedup.

The JSON artifact contains every raw paired duration and per-shard download, deserialization, training, payload-byte, overlap, and wait-time trace.

## Artifact checksums

- `benchmark_stats_md17.json`: `56e34a33421eb31e1a67348549c32ecaecbeded2b78116db270a17fc6a23c7f3`
- `benchmark_comparison_md17.png`: `cf26d005fa62788464f196ecbcff82f256ddc81d7d4ab2bce4ef88ce3b307a72`
- `benchmark_source_manifest_md17.json`: `a2880551c8175986cc3207532686442458e179c30423f179e57092033aded9d3`
- `foldpipe-md17-rigorous-benchmark-v6.log`: `f601d6f2f44398845f56102685ff42ce9871a3c6a025d4e7cbbb0ddd1c960a6c`
