import torch
from torch.utils.data import DataLoader
from torchmdnet.datasets import MD17

def get_optimized_dataloader(data_dir='./md17_data', dataset_arg='aspirin', batch_size=64, num_workers=4):
    """
    Returns an optimized PyTorch DataLoader for the MD17 dataset.
    
    Optimizations applied:
    - num_workers > 0 (e.g., 4) allows the CPU to fetch data in parallel while GPU computes.
    - pin_memory=True allocates data in page-locked memory, which speeds up transfer to the GPU.
    - prefetch_factor > 1 ensures the data queue is populated before the GPU asks for it.
    """
    dataset = MD17(data_dir, dataset_arg=dataset_arg)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,          # Accelerates CPU to GPU data transfer
        prefetch_factor=2,        # Pre-fetch 2 batches per worker
        persistent_workers=True   # Keep worker processes alive between epochs
    )
    
    return dataloader

if __name__ == "__main__":
    # Test the dataloader setup
    loader = get_optimized_dataloader(num_workers=1)
    print(f"Optimized DataLoader initialized. Batch size: {loader.batch_size}, Num workers: {loader.num_workers}")
