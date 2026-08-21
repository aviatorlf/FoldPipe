import nbformat as nbf
import os
import json

nb = nbf.v4.new_notebook()

code1 = """!pip install -q torch_geometric huggingface_hub"""

code2 = """import os
import torch
from huggingface_hub import HfApi, login
from torch_geometric.datasets import MD17
from tqdm import tqdm

# We use secret management on Kaggle
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
except:
    # Fallback to env var if running locally
    hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

print("Downloading MD17 Aspirin dataset...")
dataset = MD17(root='/kaggle/working/data', name='aspirin')
print(f"Dataset downloaded: {len(dataset)} samples")

# We want chunks of 5000 Data objects
chunk_size = 5000
chunks_dir = '/kaggle/working/shards'
os.makedirs(chunks_dir, exist_ok=True)

api = HfApi(token=hf_token)
repo_id = "aviatorlf/md17-shards"
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"Created repository {repo_id}")
except Exception as e:
    print(f"Repository already exists or error: {e}")

all_data = [data for data in dataset]
total_chunks = (len(all_data) + chunk_size - 1) // chunk_size

for i in range(total_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(all_data))
    
    chunk = all_data[start_idx:end_idx]
    filename = f"checkpoint_batch_{i:04d}.pt"
    local_path = os.path.join(chunks_dir, filename)
    
    # Save the list of PyG Data objects natively using torch.save
    torch.save(chunk, local_path)
    print(f"Saved {filename} with {len(chunk)} elements.")

# Upload all generated shards directly from the output directory
print("Uploading shards to HuggingFace...")
api.upload_folder(
    folder_path=chunks_dir,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Initial shard migration from MD17"
)
print("Migration completed successfully!")
"""

nb['cells'] = [nbf.v4.new_code_cell(code1), nbf.v4.new_code_cell(code2)]
with open('scripts/md17_migration/migrate_md17.ipynb', 'w') as f:
    nbf.write(nb, f)

kernel_metadata = {
  "id": "aviatorlf/migrate-md17-shards",
  "title": "migrate-md17-shards",
  "code_file": "migrate_md17.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
with open('scripts/md17_migration/kernel-metadata.json', 'w') as f:
    json.dump(kernel_metadata, f, indent=2)
