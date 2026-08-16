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
