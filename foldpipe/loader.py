import concurrent.futures

class AsyncFoldPipeLoader:
    """
    True O(1) bounded-memory asynchronous streaming dataloader.
    Downloads native PyTorch .pt chunk files via a generic Source backend in the background.
    Hides network I/O latency behind GPU computation.
    """
    def __init__(self, source, batch_size=128):
        self.source = source
        self.batch_size = batch_size
        self.files = self.source.get_files()

    def __iter__(self):
        """Consumer pipeline."""
        if not self.files:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Kick off the prefetch for Chunk 0
            future_chunk = executor.submit(self.source.download_chunk, self.files[0])
            
            for i in range(len(self.files)):
                # Block only if GPU is faster than network
                chunk_tensor = future_chunk.result()
                
                # Kick off prefetch for Chunk N+1 immediately
                if i + 1 < len(self.files):
                    future_chunk = executor.submit(self.source.download_chunk, self.files[i+1])
                
                # Yield batches to the GPU
                for b in range(0, chunk_tensor.size(0), self.batch_size):
                    yield chunk_tensor[b:b+self.batch_size]
                
                # Explicitly delete the chunk (relying on Python GC for O(1) bound)
                del chunk_tensor
