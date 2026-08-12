# Data Directory

This directory is intentionally left empty. The `FoldPipe` dataloader automatically fetches and caches the MD17 dataset (and other specified molecular dynamics datasets) from PyTorch Geometric upon first execution.

**Note:** The dataset will be downloaded and processed into `.pt` (PyTorch tensor) files to dramatically accelerate loading on subsequent runs.
