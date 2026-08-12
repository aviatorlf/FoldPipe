import torch
import time
from .dataset import get_optimized_dataloader

def train_baseline_optimized(epochs=10):
    """
    Optimized training loop applying Mixed Precision and Memory Management.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting optimized training on {device}")
    
    # Initialize optimized dataloader
    dataloader = get_optimized_dataloader(batch_size=128, num_workers=4)
    
    # Mock model for profiling (in reality, you'd initialize a TorchMD-Net model here)
    # Using a simple linear layer just to simulate the memory load of gradient passes
    model = torch.nn.Linear(100, 100).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. Mixed Precision Setup (uses FP16 for speed/memory, keeps FP32 for gradients where needed)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            # Move data to device, non_blocking=True pairs well with pin_memory=True in DataLoader
            # Since torchmd-net batches might not standard tensors initially depending on the version,
            # we handle attributes commonly present.
            try:
                z = batch.z.to(device, non_blocking=True)
                pos = batch.pos.to(device, non_blocking=True)
                y = batch.y.to(device, non_blocking=True)
            except AttributeError:
                pass
            
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            
            # Forward pass with Automatic Mixed Precision
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Dummy forward pass for profiling
                mock_input = torch.randn(128, 100, device=device)
                loss = model(mock_input).sum()
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 2. Memory Management: aggressive cache clearing
            # Prevents memory fragmentation on constrained GPUs like Kaggle's T4
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds.")

if __name__ == "__main__":
    train_baseline_optimized(epochs=1)
