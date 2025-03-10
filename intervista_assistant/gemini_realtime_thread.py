"""
Gemini Realtime Thread for Intervista Assistant.
Handles communication with Google's Gemini API for real-time text processing.
"""
import os
import time
import json
import logging
import threading
import queue
import pyaudio
import wave
import numpy as np
from datetime import datetime
from typing import Dict, Any, Callable, Optional

# Import the Gemini client
try:
    from .gemini_client import GeminiClient
except ImportError:
    from gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiRealtimeThread(threading.Thread):
    """
    Thread for handling real-time communication with Gemini API.
    Uses callbacks instead of PyQt signals, but maintains similar functionality.
    """
    
    # Audio recording constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    def __init__(self, callbacks=None):
        """
        Initializes the instance with optional callbacks.
        
        Args:
            callbacks: dictionary of callback functions:
                - on_transcription(text)
                - on_response(text)
                - on_error(message)
                - on_connection_status(connected)
        """
        super().__init__()
        self.callbacks = callbacks or {}
        self.running = False
        self.connected = False
        self.transcription_buffer = ""
        self.recording = False
        self.daemon = True  # Thread will exit when main program exits
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.gemini_client.initialize(api_key)
        else:
            logger.error("No Gemini API key found in environment")
        
        # Message queue for sending text to the model
        self.message_queue = queue.Queue()
        
        # Chat history for context
        self.chat_history = []
        
        # System prompt
        self.system_prompt = """You are an expert job interview assistant, specialized in helping candidates in real-time during interviews. Provide useful, clear, and concise advice on both technical and behavioral aspects."""
        
        # Audio recording variables
        self.p = None
        self.stream = None
        self.audio_buffer = []
        self.accumulated_audio = b''
        self.is_speaking = False
        self.silence_start_time = 0
        self.silence_threshold = 500  # Adjust based on testing
        self.silence_duration = 1.5  # seconds
        self.max_audio_duration = 30  # seconds
        self.last_commit_time = 0
        self.lock = threading.Lock()
        
    def run(self):
        """Main thread execution."""
        self.running = True
        logger.info("Gemini realtime thread started")
        
        # Set connection status
        self.connected = self.gemini_client.is_available()
        self._call_callback('on_connection_status', self.connected)
        
        if not self.connected:
            error_msg = "Failed to connect to Gemini API"
            logger.error(error_msg)
            self._call_callback('on_error', error_msg)
            return
        
        # Process messages from the queue
        while self.running:
            try:
                # Check if there are messages to process
                try:
                    message = self.message_queue.get(block=False)
                    self._process_message(message)
                    self.message_queue.task_done()
                except queue.Empty:
                    # No messages in the queue, continue the loop
                    pass
                
                # Process audio if recording
                if self.recording and self.stream:
                    self._process_audio_chunk()
                
                # Sleep to prevent high CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                error_msg = f"Error in Gemini thread: {str(e)}"
                logger.error(error_msg)
                self._call_callback('on_error', error_msg)
                time.sleep(1)  # Prevent rapid error loops
        
        logger.info("Gemini realtime thread stopped")
    
    def stop(self):
        """Stops the thread."""
        logger.info("Stopping Gemini realtime thread")
        self.running = False
        self.connected = False
        self._call_callback('on_connection_status', False)
        
        # Stop recording if active
        if self.recording:
            self.stop_recording()
    
    def send_text(self, text):
        """
        Sends text to the Gemini model for processing.
        
        Args:
            text: The text message to send
            
        Returns:
            bool: True if the message was queued successfully
        """
        if not text or not self.connected:
            return False
        
        try:
            # Add to queue for processing
            self.message_queue.put(text)
            logger.info(f"Text message queued: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error queueing text message: {str(e)}")
            return False
    
    def _process_message(self, text):
        """
        Processes a text message with the Gemini API.
        
        Args:
            text: The text message to process
        """
        try:
            # Add user message to history
            self.chat_history.append({"role": "user", "content": text})
            
            # Call Gemini API
            response = self.gemini_client.process_text(
                text, 
                session_history=self._format_history_for_gemini()
            )
            
            if response:
                # Add assistant response to history
                self.chat_history.append({"role": "assistant", "content": response})
                
                # Send response through callback
                self._call_callback('on_response', response)
                logger.info(f"Received response from Gemini: {response[:50]}...")
            else:
                error_msg = "Gemini API returned empty response"
                logger.error(error_msg)
                self._call_callback('on_error', error_msg)
                
        except Exception as e:
            error_msg = f"Error processing message with Gemini: {str(e)}"
            logger.error(error_msg)
            self._call_callback('on_error', error_msg)
    
    def _format_history_for_gemini(self):
        """
        Formats chat history for Gemini API.
        
        Returns:
            List of formatted messages
        """
        formatted_history = []
        
        # Add system prompt
        if self.system_prompt:
            formatted_history.append({
                "role": "system",
                "parts": [self.system_prompt]
            })
        
        # Add conversation history (limited to last 10 messages)
        for msg in self.chat_history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        return formatted_history
    
    def _call_callback(self, callback_name, *args, **kwargs):
        """
        Calls a callback function if it exists.
        
        Args:
            callback_name: Name of the callback function
            *args, **kwargs: Arguments to pass to the callback
        """
        callback = self.callbacks.get(callback_name)
        if callback and callable(callback):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback_name}: {str(e)}")
    
    def start_recording(self):
        """Starts audio recording from the microphone."""
        with self.lock:
            if self.recording:
                return
            self.recording = True
        
        self.audio_buffer = []
        self.accumulated_audio = b''
        
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
            
            logger.info("Audio recording started")
            self._call_callback('on_transcription', "Recording started...")
            
        except Exception as e:
            logger.error(f"Error initializing PyAudio: {str(e)}")
            self._call_callback('on_error', f"Audio initialization error: {str(e)}")
            self.recording = False
    
    def stop_recording(self):
        """Stops audio recording."""
        with self.lock:
            if not self.recording:
                return
            self.recording = False
        
        # Process any remaining audio
        if self.accumulated_audio:
            self._process_accumulated_audio()
        
        # Clean up audio resources
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {str(e)}")
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {str(e)}")
        
        self.stream = None
        self.p = None
        
        logger.info("Audio recording stopped")
        self._call_callback('on_transcription', "Recording stopped[end]")
    
    def _process_audio_chunk(self):
        """Processes a chunk of audio data."""
        try:
            # Read audio data
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            
            # Convert to numpy array for analysis
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate audio level
            audio_level = np.abs(audio_data).mean()
            
            # Detect speech/silence
            is_speaking_now = audio_level > self.silence_threshold
            
            # Accumulate audio data
            self.accumulated_audio += data
            
            # Handle silence detection
            current_time = time.time()
            if self.is_speaking and not is_speaking_now:
                # Transition from speech to silence
                if self.silence_start_time == 0:
                    self.silence_start_time = current_time
                elif current_time - self.silence_start_time > self.silence_duration:
                    # Silence duration exceeded, process the accumulated audio
                    self._process_accumulated_audio()
                    self.silence_start_time = 0
            elif not self.is_speaking and is_speaking_now:
                # Transition from silence to speech
                self.silence_start_time = 0
            
            # Update speaking state
            self.is_speaking = is_speaking_now
            
            # Check if max duration exceeded
            if len(self.accumulated_audio) > self.RATE * self.max_audio_duration * 2:  # *2 for 16-bit samples
                logger.info("Max audio duration reached, processing accumulated audio")
                self._process_accumulated_audio()
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
    
    def _process_accumulated_audio(self):
        """Processes the accumulated audio buffer."""
        if not self.accumulated_audio:
            return
        
        try:
            # Process with Gemini
            result = self.gemini_client.process_audio(
                self.accumulated_audio, 
                sample_rate=self.RATE, 
                encoding="LINEAR16"
            )
            
            if result and 'transcription' in result:
                transcription = result['transcription']
                
                # Add [fine] marker to indicate this is a complete segment
                if not transcription.endswith("[fine]") and not transcription.endswith("[end]"):
                    transcription += "[fine]"
                
                # Call transcription callback
                self._call_callback('on_transcription', transcription)
                
                # If there's a response, call response callback
                if 'response' in result and result['response']:
                    self._call_callback('on_response', result['response'])
            else:
                logger.warning("No transcription result from Gemini")
                
        except Exception as e:
            logger.error(f"Error processing accumulated audio: {str(e)}")
        
        # Reset accumulated audio
        self.accumulated_audio = b''