#!/usr/bin/env python3
"""
Audio Handler Module for Intervista Assistant.
Handles audio processing and transcription using Gemini API.
"""
import os
import base64
import logging
from typing import Dict, Any, Optional
from ..gemini_client import GeminiClient

# Configure logging
logger = logging.getLogger(__name__)

class AudioHandler:
    """Handles audio processing and transcription using Gemini API."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize the audio handler."""
        self.gemini_client = gemini_client
        
    def process_audio_data(self, audio_data: str, sample_rate: int = 16000, encoding: str = 'LINEAR16') -> Dict[str, Any]:
        """Process audio data using Gemini API.
        
        Args:
            audio_data: Base64 encoded audio data
            sample_rate: Audio sample rate (default: 16000)
            encoding: Audio encoding format (default: LINEAR16)
            
        Returns:
            Dict containing transcription and response
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            logger.info(f"Processing audio data: {len(audio_bytes)} bytes")
            
            # Process with Gemini if available
            if self.gemini_client.is_available():
                # Send to Gemini for processing
                result = self.gemini_client.process_audio(audio_bytes, sample_rate, encoding)
                
                if result and 'transcription' in result:
                    return result
                else:
                    logger.error("Gemini processing failed or returned no transcription")
                    return {'error': 'Gemini processing failed'}
            else:
                logger.error("Gemini API not available")
                return {'error': 'Gemini API not available'}
                
        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}")
            return {'error': str(e)}