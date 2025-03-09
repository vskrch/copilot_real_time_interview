import os
import time
import logging
import numpy as np
from typing import Optional, Callable, List, Dict, Any, Tuple
from faster_whisper import WhisperModel
import sounddevice as sd

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    Local speech-to-text transcription using Whisper.cpp with GPU acceleration on Mac.
    """
    
    def __init__(self, 
                 model_size: str = "base", 
                 device: str = "auto",
                 compute_type: str = "float16",
                 on_transcription: Optional[Callable[[str], None]] = None):
        """
        Initialize the WhisperTranscriber.
        
        Args:
            model_size: Size of the Whisper model to use (tiny, base, small, medium, large-v2)
            device: Device to use for inference (cpu, cuda, auto)
            compute_type: Compute type for inference (float16, float32, int8)
            on_transcription: Callback function to handle transcription results
        """
        self.on_transcription = on_transcription
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.recording = False
        self.audio_buffer = []
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.silence_threshold = 500  # RMS value to define silence
        self.pause_duration = 0.7     # seconds of pause to trigger transcription
        self.min_audio_duration = 1.0  # minimum audio duration before transcription
        
        # State tracking
        self.is_speaking = False
        self.silence_start_time = 0
        self.audio_start_time = 0
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model with GPU acceleration if available."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device} with {self.compute_type}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def start_recording(self):
        """Start recording audio for transcription."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_buffer = []
        self.audio_start_time = time.time()
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            dtype='int16'
        )
        self.stream.start()
        logger.info("Started recording for local transcription")
    
    def stop_recording(self):
        """Stop recording and process any remaining audio."""
        if not self.recording:
            return
        
        self.recording = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        
        # Process any remaining audio
        if self.audio_buffer:
            self._process_audio()
        
        logger.info("Stopped recording for local transcription")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if not self.recording:
            return
        
        # Convert to float32 for RMS calculation
        audio_float = indata.flatten().astype(np.float32) / 32768.0
        
        # Calculate RMS to detect silence
        rms = np.sqrt(np.mean(audio_float**2)) * 1000
        
        # Detect speaking/silence transitions
        current_time = time.time()
        if rms > self.silence_threshold:
            # User is speaking
            self.is_speaking = True
            self.audio_buffer.append(indata.copy())
        elif self.is_speaking:
            # User was speaking but now is silent
            self.audio_buffer.append(indata.copy())
            
            if not hasattr(self, 'silence_start_time') or self.silence_start_time == 0:
                self.silence_start_time = current_time
            elif current_time - self.silence_start_time > self.pause_duration:
                # Silence duration exceeded threshold, process the audio
                if current_time - self.audio_start_time > self.min_audio_duration:
                    self._process_audio()
                
                # Reset state
                self.is_speaking = False
                self.silence_start_time = 0
                self.audio_buffer = []
                self.audio_start_time = current_time
        else:
            # Still silent, just add to buffer
            self.audio_buffer.append(indata.copy())
    
    def _process_audio(self):
        """Process the accumulated audio buffer and perform transcription."""
        if not self.audio_buffer:
            return
        
        try:
            # Concatenate audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Convert to the format expected by Whisper (float32 normalized to [-1, 1])
            audio_float = audio_data.flatten().astype(np.float32) / 32768.0
            
            # Perform transcription
            segments, info = self.model.transcribe(
                audio_float, 
                beam_size=5,
                language="en",
                vad_filter=True
            )
            
            # Process segments
            transcription = " ".join([segment.text for segment in segments])
            
            # Call the callback with the transcription result
            if self.on_transcription and transcription.strip():
                self.on_transcription(transcription.strip())
                
            logger.info(f"Transcribed: {transcription[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing audio for transcription: {str(e)}")