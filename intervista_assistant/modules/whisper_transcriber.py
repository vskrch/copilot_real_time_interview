#!/usr/bin/env python3
"""
Whisper Transcriber Module for Intervista Assistant.
Handles local speech-to-text transcription using Whisper.
"""
import os
import io
import logging
import tempfile
import numpy as np
from typing import Dict, Any, Optional
import whisper

# Configure logging
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Handles local speech-to-text transcription using Whisper."""
    
    def __init__(self):
        """Initialize the Whisper transcriber."""
        self.model = None
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        self.device = os.getenv("WHISPER_DEVICE", "auto")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        
    def load_model(self):
        """Loads the Whisper model."""
        try:
            # Clean up the model name - remove any comments or extra text
            model_name = self.model_size.split('#')[0].strip()
            if model_name not in ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 
                                 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 
                                 'large', 'large-v3-turbo', 'turbo']:
                # Default to base if invalid model specified
                logger.warning(f"Invalid model name: {model_name}. Defaulting to 'base'")
                model_name = 'base'
                
            # Clean up device specification
            device = self.device.split('#')[0].strip()
            if device not in ['cpu', 'cuda', 'auto']:
                device = 'auto'
                
            logger.info(f"Loading Whisper model: {model_name} on {device}")
            self.model = whisper.load_model(model_name, device=device)
            logger.info(f"Whisper model {model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            return False
            
    def transcribe_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Whisper.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Dict containing transcription text and confidence
        """
        try:
            # Load the model if not already loaded
            if not self.load_model():
                return None
                
            # Convert audio bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed
            if sample_rate != 16000:
                # Simple resampling for demonstration
                # In production, use a proper resampling library like librosa
                target_length = int(len(audio_np) * 16000 / sample_rate)
                audio_np = np.interp(
                    np.linspace(0, len(audio_np), target_length),
                    np.arange(len(audio_np)),
                    audio_np
                )
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_np, 
                language="en",
                fp16=(self.compute_type == "float16")
            )
            
            # Extract text and confidence
            transcription = result.get("text", "").strip()
            confidence = 0.0
            
            # Some Whisper versions include segment-level confidence
            if "segments" in result and result["segments"]:
                # Average confidence across segments
                confidences = [seg.get("confidence", 0.0) for seg in result["segments"]]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            logger.info(f"Transcription: {transcription[:50]}...")
            
            return {
                "text": transcription,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None