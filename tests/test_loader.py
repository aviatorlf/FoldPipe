import unittest
from unittest.mock import patch, MagicMock
from foldpipe.loader import AsyncFoldPipeLoader

class TestAsyncFoldPipeLoader(unittest.TestCase):

    @patch('foldpipe.loader.Credentials.from_authorized_user_info')
    @patch('foldpipe.loader.build')
    def test_drive_pagination(self, mock_build, mock_creds):
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
                # no nextPageToken
            }
        ]
        
        # Initialize the loader (this triggers the listing)
        loader = AsyncFoldPipeLoader('mock_folder_id', {'dummy': 'creds'})
        
        # Verify the accumulated files list has all 3 items
        self.assertEqual(len(loader.files), 3)
        self.assertEqual(loader.files[0]['id'], '1')
        self.assertEqual(loader.files[2]['id'], '3')
        
        # Verify that list was called twice (due to pagination)
        self.assertEqual(mock_files.list.call_count, 2)
        
        # Verify that the second call used the token
        kwargs = mock_files.list.call_args_list[1][1]
        self.assertEqual(kwargs.get('pageToken'), 'token_abc')

if __name__ == '__main__':
    unittest.main()
