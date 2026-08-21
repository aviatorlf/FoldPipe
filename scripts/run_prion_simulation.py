import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from foldpipe.prion_loader import PrionStreamer
from torchmdnet.models.model import create_model

def run_prion_test():
    # 1. Hardware Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Prion pipeline on: {device}")
    
    # 2. Setup Dataset Directory
    # We downloaded 1QLX.pdb to the root directory. Let's move it to data/prion/raw/
    raw_dir = os.path.join('data', 'prion', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists('1QLX.pdb'):
        os.rename('1QLX.pdb', os.path.join(raw_dir, '1QLX.pdb'))
        
    # 3. Load Prion Dataset
    print("\nLoading Prion Dataset...")
    dataset = PrionStreamer(raw_dir='./data/prion/raw')
    
    # 4. Initialize stream and pull first batch
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader)).to(device)
    num_atoms = batch.z.shape[0]
    print(f"Structure 0 (1QLX) has {num_atoms} atoms.")
    
    # 5. Initialize TorchMD-Net Equivariant Transformer
    from torchmdnet.scripts.train import get_args
    import argparse
    import sys
    
    original_argv = sys.argv
    sys.argv = ['run_prion_simulation.py']
    model_args = vars(get_args())
    sys.argv = original_argv
    
    model_args.update({
        'model': 'equivariant-transformer',
        'output_model': 'Scalar',
        'derivative': True,
        'embedding_dimension': 64, # Smaller dim for local CPU testing
        'num_layers': 2,           # Fewer layers for local CPU testing
        'max_num_neighbors': 128,  # Increased for massive protein structures
    })
    
    print("\nInitializing TorchMD-Net Equivariant Transformer...")
    model = create_model(model_args).to(device)
    
    # 6. Forward Pass
    print(f"\nRunning Forward Pass with {num_atoms} atoms...")
    try:
        # Pass atomic numbers (z), coordinates (pos), and batch indices
        pred_y, pred_neg_dy = model(batch.z, batch.pos, batch=batch.batch)
        print("✅ FORWARD PASS SUCCESSFUL!")
        print(f"Predicted Energy (Scalar): {pred_y.shape}")
        print(f"Predicted Forces (dY/dPos): {pred_neg_dy.shape} (Should match [num_atoms, 3])")
    except Exception as e:
        print(f"❌ FORWARD PASS FAILED: {e}")
        raise

if __name__ == "__main__":
    run_prion_test()
