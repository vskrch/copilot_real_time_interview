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
        """Process audio data using local Whisper and Gemini API.
        
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
            
            # Use local Whisper for transcription
            from ..modules.whisper_transcriber import WhisperTranscriber
            
            # Initialize the transcriber
            transcriber = WhisperTranscriber()
            
            # Get transcription from Whisper
            transcription_result = transcriber.transcribe_audio(audio_bytes, sample_rate)
            
            if not transcription_result or 'text' not in transcription_result:
                logger.error("Whisper transcription failed")
                return {'error': 'Transcription failed'}
                
            transcription = transcription_result['text']
            
            # Process with Gemini if available
            if self.gemini_client.is_available() and transcription:
                # Generate response with Gemini
                success, response = self.gemini_client.generate_text(
                    f"The user said: '{transcription}'. Provide a helpful response."
                )
                
                if success:
                    return {
                        'transcription': transcription,
                        'response': response
                    }
                else:
                    logger.error(f"Gemini response generation failed: {response}")
                    return {
                        'transcription': transcription,
                        'error': 'Response generation failed'
                    }
            else:
                # Just return the transcription
                return {'transcription': transcription}
                
        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}")
            return {'error': str(e)}