import nbformat as nbf
import os
import json

os.makedirs('scripts/kaggle_molecular', exist_ok=True)
nb = nbf.v4.new_notebook()

code1 = """!pip install -q torch_geometric huggingface_hub matplotlib psutil
import torch
version = torch.__version__.split("+")[0]
# Kaggle sometimes formats cuda as 12.1, we need cu121
cuda_version = torch.version.cuda.replace(".", "")
url = f"https://data.pyg.org/whl/torch-{version}+cu{cuda_version}.html"
print(f"Installing PyG extensions from {url}...")
!pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f {url}"""

code2 = """# Clone the repository dynamically to get the latest FoldPipe library and benchmark script
!git clone https://github.com/aviatorlf/FoldPipe.git
%cd FoldPipe
!pip install -e ."""

code3 = """import os
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
except Exception as e:
    print("Warning: Could not load HF_TOKEN from Kaggle Secrets.")

!python scripts/benchmark_molecular.py"""

nb['cells'] = [nbf.v4.new_code_cell(code1), nbf.v4.new_code_cell(code2), nbf.v4.new_code_cell(code3)]
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    }
}
with open('scripts/kaggle_molecular/benchmark.ipynb', 'w') as f:
    nbf.write(nb, f)

kernel_metadata = {
  "id": "aviatorlf/benchmark-molecular-foldpipe-v7",
  "title": "benchmark-molecular-foldpipe-v7",
  "code_file": "benchmark.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
with open('scripts/kaggle_molecular/kernel-metadata.json', 'w') as f:
    json.dump(kernel_metadata, f, indent=2)
