# FoldPipe MD17 + SchNet benchmark

- Generated: 2026-08-17T04:59:36.371481+00:00
- Hardware: Tesla T4
- Dataset: `aviatorlf/md17-shards@f779686deb9217877dd7ddde99b2522bd441492a`
- Git commit: `16fdbb26b00f9721ce4034335ce0ee12bda77720`
- Source checkout dirty: `False`
- Source bundle: `4fc3f6d712557822efc0d5d71aaa616b0fe30f578631726ef60d6a0fe73b32c0`
- Base Git commit: `16fdbb26b00f9721ce4034335ce0ee12bda77720`
- Kaggle kernel: [`dhirenkhatri/foldpipe-md17-rigorous-benchmark`, version 8](https://www.kaggle.com/code/dhirenkhatri/foldpipe-md17-rigorous-benchmark)
- Protocol: 20 paired, order-alternating passes; 5 pinned shards per pass; batch size 32
- Warm-up: one untimed training batch from the first pinned shard
- Validation scope: 40 timed pipeline passes, 200 shard traces, and 1,000,000 repeated structure-visits

| Metric | Sequential | FoldPipe |
| --- | ---: | ---: |
| Mean time (s) | 83.372 | 76.780 |
| 95% bootstrap CI, mean time (s) | [67.868, 101.057] | [64.859, 89.440] |
| Mean peak RSS (GiB) | 1.935 | 2.064 |
| Mean sampled GPU utilization (%) | 36.172 | 37.703 |
| Mean I/O/compute overlap (s) | 0.000 | 16.333 |
| Mean GPU wait time (s) | 56.503 | 48.652 |

Geometric mean paired speedup: **1.0587x** (95% paired bootstrap CI in log-ratio space [0.8776, 1.2880]).

Mean paired time saved: **6.591 s** (95% paired bootstrap CI [-7.692, 21.108]).

Median paired time saved: **3.453 s** (95% paired bootstrap CI [-13.477, 22.888]). FoldPipe was faster in 55% of pairs.

For continuity with the earlier artifact, the arithmetic mean of paired speedup ratios was **1.1705x**; it is retained as a supplementary, skew-sensitive summary rather than the headline ratio.

All three paired intervals include their no-effect values; this run is inconclusive about a speed advantage.

The JSON artifact contains every raw paired duration and per-shard download, deserialization, training, payload-byte, overlap, and wait-time trace.

## Artifact checksums

- `benchmark_stats_md17.json`: `0832fa4180a95c78d7a017f0c960f3e4d7273c3e0226428734197d71697df21d`
- `benchmark_comparison_md17.png`: `109e3783fad92e9dace75c80e205ad4ee2c20fb779ef70261951db901a316780`
- `benchmark_source_manifest_md17.json`: `5658ca6676a4eb055d6bb914d54e1918928a82b7e45b2a6150a8f9049be39a68`
- `foldpipe-md17-rigorous-benchmark-v8.log`: `cdbc4b30e85ef5c741d6d404945bd3b6b317d7569ec073b952adcf809f0f3f1d`
