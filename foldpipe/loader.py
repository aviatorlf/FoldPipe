import concurrent.futures

class AsyncFoldPipeLoader:
    """
    Bounded-working-set asynchronous shard iterator.

    Keeps a constant number of shard payloads live with respect to total
    dataset shard count, assuming individual shards are bounded.  Downloads
    native PyTorch .pt chunk files via a generic Source backend in a
    background thread and yields batches to the training loop.
    """
    def __init__(self, source, batch_size=128, batch_fn=None):
        self.source = source
        self.batch_size = batch_size
        self.batch_fn = batch_fn or self._default_batch_fn

    def _default_batch_fn(self, chunk):
        """Default tensor slicing for symmetric tensors."""
        for b in range(0, chunk.size(0), self.batch_size):
            yield chunk[b:b+self.batch_size]

    def __iter__(self):
        """Consumer pipeline."""
        file_iterator = self.source.iter_files()
        try:
            first_file = next(file_iterator)
        except StopIteration:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Kick off the prefetch for Chunk 0
            future_chunk = executor.submit(self.source.download_chunk, first_file)
            
            while True:
                # Block only if GPU is faster than network
                chunk_tensor = future_chunk.result()
                
                try:
                    next_file = next(file_iterator)
                    # Kick off prefetch for Chunk N+1 immediately
                    future_chunk = executor.submit(self.source.download_chunk, next_file)
                    has_next = True
                except StopIteration:
                    has_next = False
                
                # Yield batches to the GPU
                for batch in self.batch_fn(chunk_tensor):
                    yield batch
                
                # Drop the consumer's reference to the completed shard.
                del chunk_tensor
                
                if not has_next:
                    break
