import os
import json
import shutil

os.makedirs("benchmark_notebook", exist_ok=True)
if os.path.exists("benchmark_notebook/foldpipe"):
    shutil.rmtree("benchmark_notebook/foldpipe")
shutil.copytree("foldpipe", "benchmark_notebook/foldpipe")

# Read the benchmark python script
with open("scripts/benchmark_whitepaper.py", "r") as f:
    benchmark_code = f.read()

# Build the standalone notebook
nb = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install torch==2.3.1 torchvision torchaudio google-api-python-client git+https://github.com/aviatorlf/FoldPipe.git\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [line + "\n" for line in benchmark_code.split("\n")]
  }
 ],
 "metadata": {
  "accelerator": "nvidiaTeslaT4",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("benchmark_notebook/foldpipe_benchmark.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

# Build the Kaggle metadata
meta = {
  "id": "dhirenkhatri/foldpipe-whitepaper-benchmarks",
  "title": "FoldPipe Whitepaper Benchmarks",
  "code_file": "foldpipe_benchmark.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "accelerator": "nvidiaTeslaT4",
  "enable_internet": "true",
  "dataset_sources": ["dhirenkhatri/gcp-secret-dataset"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}

with open("benchmark_notebook/kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=1)

print("Created benchmark_notebook with standalone embedded notebook and metadata.")
