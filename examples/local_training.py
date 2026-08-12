import sys
import os
import torch

# Ensure the parent directory is in the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_baseline_optimized

if __name__ == "__main__":
    print("FoldPipe: Starting Local Training Example")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Run a short training loop (1 epoch) for demonstration
    train_baseline_optimized(epochs=1)
