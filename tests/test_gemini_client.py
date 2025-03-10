"""Tests for the Gemini client."""

import unittest
from unittest.mock import patch, MagicMock
from intervista_assistant.gemini_client import GeminiClient

class TestGeminiClient(unittest.TestCase):
    """Test cases for GeminiClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = GeminiClient()
    
    @patch('google.generativeai.configure')
    def test_initialize_with_valid_key(self, mock_configure):
        """Test initialization with valid API key."""
        result = self.client.initialize(api_key='test_key')
        self.assertTrue(result)
        mock_configure.assert_called_once_with(api_key='test_key')
    
    def test_initialize_without_key(self):
        """Test initialization without API key."""
        result = self.client.initialize()
        self.assertFalse(result)
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_text(self, mock_model):
        """Test text generation."""
        mock_model.return_value.generate_content.return_value.text = 'Test response'
        self.client.initialized = True
        self.client.api_key = 'test_key'
        
        success, response = self.client.generate_text('Test prompt')
        self.assertTrue(success)
        self.assertEqual(response, 'Test response')

if __name__ == '__main__':
    unittest.main()