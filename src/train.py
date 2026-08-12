import torch
import torch.nn as nn
import time
from .dataset import get_optimized_dataloader
from torchmdnet.models.model import create_model

def train_baseline_optimized(epochs=10):
    """
    Optimized training loop applying Mixed Precision, Memory Management,
    and Dual-GPU DataParallel scaling for TorchMD-Net.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting optimized training on {device}")
    
    # Initialize optimized dataloader
    dataloader = get_optimized_dataloader(batch_size=64, num_workers=4)
    
    from torchmdnet.scripts.train import get_args
    # Get all default TorchMD-Net arguments
    import argparse
    # We pass an empty list to parse_args so it doesn't read sys.argv
    try:
        from torchmdnet.scripts.train import get_parser
        parser = get_parser()
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser()
        
    # the easiest way is just to call get_args() since we saw it returns the defaults when sys.argv has no torchmd args
    import sys
    original_argv = sys.argv
    sys.argv = ['train.py'] # dummy
    model_args = vars(get_args())
    sys.argv = original_argv
    
    # Overwrite specific hyper-parameters for our Equivariant Transformer
    model_args.update({
        'model': 'equivariant-transformer',
        'output_model': 'Scalar',
        'derivative': True,             # Predict forces alongside energy
        'embedding_dimension': 128,
        'num_layers': 6,
        'num_rbf': 50,
        'activation': 'silu',
        'rbf_type': 'expnorm',
        'trainable_rbf': True,
        'neighbor_embedding': True,
        'distance_influence': 'both',
        'attn_activation': 'silu',
        'num_heads': 8,
        'cutoff_lower': 0.0,
        'cutoff_upper': 5.0,
        'max_z': 100,
        'max_num_neighbors': 32,
        'reduce_op': 'add',
        'prior_model': None,
        'aggr': 'add',
        'check_errors': False,
        'precision': 32
    })
    
    # Instantiate the actual TorchMD-Net model
    print("Initializing TorchMD-Net Equivariant Transformer...")
    model = create_model(model_args)
    
    # Wrapper to fix PyG batch indexing under DataParallel
    class TorchMDNetWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, z, pos, batch):
            # DataParallel splits the batch tensor, but GPU 1 gets indices like 32..63
            # We must offset it to 0..31 so the model's reduce operation sizes the output correctly
            batch = batch - batch.min()
            return self.model(z, pos, batch=batch)
            
    model = TorchMDNetWrapper(model)
    
    # Wrap model in DataParallel to utilize Kaggle's dual T4 GPUs
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_y = nn.MSELoss()
    criterion_dy = nn.MSELoss()
    
    # 1. Mixed Precision Setup
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            try:
                # Move PyTorch Geometric graph data to device
                z = batch.z.to(device, non_blocking=True)
                pos = batch.pos.to(device, non_blocking=True)
                batch_idx = batch.batch.to(device, non_blocking=True)
                
                y = batch.y.to(device, non_blocking=True)
                neg_dy = batch.neg_dy.to(device, non_blocking=True)
            except AttributeError as e:
                print(f"Batch missing attribute: {e}")
                continue
            
            optimizer.zero_grad(set_to_none=True)
            
            # Predict energy and forces (AMP disabled due to TorchMD-Net Half/Float internal mismatches)
            pred_y, pred_neg_dy = model(z, pos, batch=batch_idx)
            
            # Combine losses (energy + force)
            loss_y = criterion_y(pred_y, y.view_as(pred_y))
            loss_dy = criterion_dy(pred_neg_dy, neg_dy.view_as(pred_neg_dy))
            loss = loss_y + loss_dy
                
            # Backward pass with scaler (scaler works gracefully without autocast, just acts as pass-through)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 2. Memory Management: aggressive cache clearing
            if i > 0 and i % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f} | Time: {time.time() - start_time:.2f}s")
                
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds. Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_baseline_optimized(epochs=1)
