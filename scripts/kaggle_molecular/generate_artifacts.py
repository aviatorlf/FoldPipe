import nbformat as nbf
import os
import json

os.makedirs('scripts/kaggle_molecular', exist_ok=True)
nb = nbf.v4.new_notebook()

code1 = """!pip install -q torch_geometric huggingface_hub matplotlib psutil"""

code2 = """# Clone the repository dynamically to get the latest FoldPipe library and benchmark script
!git clone https://github.com/aviatorlf/FoldPipe.git
%cd FoldPipe
!pip install -e ."""

code3 = """import os
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
except:
    print("Could not load HF_TOKEN from Kaggle Secrets")

!python scripts/benchmark_molecular.py"""

nb['cells'] = [nbf.v4.new_code_cell(code1), nbf.v4.new_code_cell(code2), nbf.v4.new_code_cell(code3)]
with open('scripts/kaggle_molecular/benchmark.ipynb', 'w') as f:
    nbf.write(nb, f)

kernel_metadata = {
  "id": "aviatorlf/benchmark-molecular-foldpipe",
  "title": "benchmark-molecular-foldpipe",
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
