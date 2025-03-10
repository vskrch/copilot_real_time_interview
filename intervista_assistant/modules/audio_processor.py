#!/usr/bin/env python3
"""
Audio Processor Module for Intervista Assistant.
Handles audio processing, transcription with Whisper, and response generation with Gemini.
"""
import os
import base64
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Processes audio data using Whisper for transcription and Gemini for responses."""
    
    def __init__(self, gemini_client):
        """Initialize the audio processor.
        
        Args:
            gemini_client: Initialized Gemini client
        """
        self.gemini_client = gemini_client
        
    def process_audio(self, audio_data: bytes, sample_rate: int = 16000, encoding: str = 'LINEAR16') -> Dict[str, Any]:
        """Process audio data using Whisper and Gemini.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            encoding: Audio encoding format
            
        Returns:
            Dict containing transcription and response
        """
        try:
            # Use Whisper for transcription
            from .whisper_transcriber import WhisperTranscriber
            
            # Initialize the transcriber
            transcriber = WhisperTranscriber()
            
            # Get transcription from Whisper
            transcription_result = transcriber.transcribe_audio(audio_data, sample_rate)
            
            if not transcription_result or 'text' not in transcription_result:
                logger.error("Whisper transcription failed")
                return {'error': 'Transcription failed'}
                
            transcription = transcription_result['text']
            confidence = transcription_result.get('confidence', 0.0)
            
            logger.info(f"Transcription: {transcription[:50]}...")
            
            # Process with Gemini if available and transcription is meaningful
            if self.gemini_client.is_available() and transcription and len(transcription.strip()) > 3:
                # Generate response with Gemini
                success, response = self.gemini_client.generate_text(
                    f"The user said: '{transcription}'. Provide a helpful response."
                )
                
                if success:
                    return {
                        'transcription': transcription,
                        'confidence': confidence,
                        'response': response
                    }
                else:
                    logger.error(f"Gemini response generation failed: {response}")
                    return {
                        'transcription': transcription,
                        'confidence': confidence,
                        'error': 'Response generation failed'
                    }
            else:
                # Just return the transcription
                return {
                    'transcription': transcription,
                    'confidence': confidence
                }
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {'error': str(e)}
    
    def process_base64_audio(self, audio_data_base64: str, sample_rate: int = 16000, encoding: str = 'LINEAR16') -> Dict[str, Any]:
        """Process base64 encoded audio data.
        
        Args:
            audio_data_base64: Base64 encoded audio data
            sample_rate: Audio sample rate
            encoding: Audio encoding format
            
        Returns:
            Dict containing transcription and response
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data_base64)
            
            # Process the raw audio bytes
            return self.process_audio(audio_bytes, sample_rate, encoding)
            
        except Exception as e:
            logger.error(f"Error processing base64 audio: {str(e)}")
            return {'error': str(e)}