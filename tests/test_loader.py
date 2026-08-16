import pytest
import torch
from unittest.mock import patch, MagicMock
from foldpipe.loader import AsyncFoldPipeLoader
from foldpipe.sources import SyntheticLatencySource

class TestGoogleDriveSource:
    @patch('foldpipe.sources.Credentials.from_authorized_user_info')
    @patch('foldpipe.sources.build')
    def test_lazy_iteration_pagination(self, mock_build, mock_creds):
        from foldpipe.sources import GoogleDriveSource
        
        # Create a mock for the Drive service
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        # Mock the files().list().execute() chaining
        mock_files = mock_service.files.return_value
        mock_list = mock_files.list.return_value
        
        # Setup pagination sequence: two pages
        # Page 1: returns 2 items and a nextPageToken
        # Page 2: returns 1 item and no token
        mock_list.execute.side_effect = [
            {
                'files': [{'id': '1', 'name': 'chunk1'}, {'id': '2', 'name': 'chunk2'}],
                'nextPageToken': 'token_abc'
            },
            {
                'files': [{'id': '3', 'name': 'chunk3'}],
            }
        ]
        
        source = GoogleDriveSource('mock_folder_id', {'dummy': 'creds'})
        iterator = source.iter_files()
        
        # Consumer lazily pulls from iterator
        files = list(iterator)
        
        assert len(files) == 3
        assert files[0]['id'] == '1'
        assert files[2]['id'] == '3'
        
        # Verify that list was called twice (due to pagination)
        assert mock_files.list.call_count == 2
        
        # Verify that the second call used the token
        kwargs = mock_files.list.call_args_list[1][1]
        assert kwargs.get('pageToken') == 'token_abc'

class TestAsyncFoldPipeLoader:
    def test_empty_source(self):
        source = SyntheticLatencySource(num_chunks=0, latency_ms=0)
        loader = AsyncFoldPipeLoader(source=source)
        
        batches = list(loader)
        assert len(batches) == 0

    def test_multi_epoch_iteration(self):
        # We configure 2 chunks, each chunk is size 400.
        # Batch size is 100. So we expect 4 batches per chunk -> 8 batches total per epoch.
        source = SyntheticLatencySource(num_chunks=2, latency_ms=0, chunk_size=400)
        loader = AsyncFoldPipeLoader(source=source, batch_size=100)
        
        # Epoch 1
        batches_epoch1 = list(loader)
        assert len(batches_epoch1) == 8
        assert batches_epoch1[0].shape == (100, 3)
        
        # Epoch 2 - tests the regression where iterator was exhausted in __init__
        batches_epoch2 = list(loader)
        assert len(batches_epoch2) == 8
        assert batches_epoch2[0].shape == (100, 3)

    def test_single_shard(self):
        source = SyntheticLatencySource(num_chunks=1, latency_ms=0, chunk_size=300)
        loader = AsyncFoldPipeLoader(source=source, batch_size=150)
        batches = list(loader)
        assert len(batches) == 2

    def test_non_divisible_batch_size(self):
        source = SyntheticLatencySource(num_chunks=1, latency_ms=0, chunk_size=100)
        loader = AsyncFoldPipeLoader(source=source, batch_size=30)
        batches = list(loader)
        # Should yield sizes: 30, 30, 30, 10
        assert len(batches) == 4
        assert batches[-1].shape == (10, 3)

    def test_custom_batch_fn(self):
        source = SyntheticLatencySource(num_chunks=1, latency_ms=0, chunk_size=100)
        # Custom batch function that returns just a string
        def dummy_batch_fn(chunk):
            yield "batch1"
            yield "batch2"
            
        loader = AsyncFoldPipeLoader(source=source, batch_size=10, batch_fn=dummy_batch_fn)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0] == "batch1"
        assert batches[1] == "batch2"

    def test_source_download_exception_propagates(self):
        class BrokenDownloadSource(SyntheticLatencySource):
            def download_chunk(self, identifier):
                raise ValueError("Simulated network failure")
                
        source = BrokenDownloadSource(num_chunks=2)
        loader = AsyncFoldPipeLoader(source=source, batch_size=10)
        
        with pytest.raises(ValueError, match="Simulated network failure"):
            list(loader)

    def test_source_iteration_exception(self):
        class BrokenIterSource(SyntheticLatencySource):
            def iter_files(self):
                yield "file1"
                raise RuntimeError("Failed to list files")
                
        source = BrokenIterSource()
        loader = AsyncFoldPipeLoader(source=source, batch_size=10)
        
        with pytest.raises(RuntimeError, match="Failed to list files"):
            list(loader)

    def test_prefetch_actually_overlaps(self):
        import time
        
        class TracingSource(SyntheticLatencySource):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.download_start_times = []
                self.download_end_times = []
                
            def download_chunk(self, identifier):
                self.download_start_times.append(time.time())
                time.sleep(0.1) # Simulate 100ms network latency
                chunk = torch.randn(10, 3)
                self.download_end_times.append(time.time())
                return chunk

        source = TracingSource(num_chunks=2)
        loader = AsyncFoldPipeLoader(source=source, batch_size=10)
        iterator = iter(loader)
        
        # 1. Ask for first batch. The background thread will fetch chunk 0, block us until chunk 0 is ready.
        #    Once chunk 0 is yielded to us, the background thread IMMEDIATELY starts fetching chunk 1.
        batch1 = next(iterator)
        batch1_processing_start = time.time()
        
        # Simulate consumer processing taking a very long time
        time.sleep(0.2) 
        batch1_processing_end = time.time()
        
        # 2. Ask for second batch.
        batch2 = next(iterator)
        
        assert len(source.download_start_times) == 2
        
        # KEY ASSERTION: Chunk 1 download MUST have started BEFORE we finished processing batch 1!
        # This proves the background thread is overlapping I/O behind our compute.
        assert source.download_start_times[1] < batch1_processing_end, "Prefetch did not overlap with consumer processing!"
