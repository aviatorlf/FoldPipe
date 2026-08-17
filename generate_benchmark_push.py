import os
import json

output_dir = "scripts/kaggle_benchmark"
os.makedirs(output_dir, exist_ok=True)

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
    "%pip install -q foldpipe==0.3.1 psutil matplotlib\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from kaggle_secrets import UserSecretsClient\n",
    "os.environ[\"HF_TOKEN\"] = UserSecretsClient().get_secret(\"HF_TOKEN\")\n"
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

with open(os.path.join(output_dir, "benchmark.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)

print(f"Created {output_dir}/benchmark.ipynb with the PyPI release and embedded benchmark driver.")
