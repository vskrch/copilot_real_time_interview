#!/usr/bin/env python3
import sys
import os
import time
import json
import logging
import asyncio
import threading
import base64
import numpy as np
import pathlib

import pyaudio
from PyQt5.QtCore import QThread, pyqtSignal

# Logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log')
logger = logging.getLogger(__name__)

class RealtimeTextThread(QThread):
    """Thread for text (and audio) communication using the Realtime API."""
    transcription_signal = pyqtSignal(str)
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool)
    
    # Add these imports at the top of the file
    from .whisper_transcriber import WhisperTranscriber
    
    # Then modify the __init__ method to include local transcription option
    def __init__(self, parent=None, use_local_transcription=False):
        super().__init__(parent)
        self.running = False
        self.connected = False
        self.transcription_buffer = ""
        self.last_event_time = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        
        # Local transcription option
        self.use_local_transcription = use_local_transcription
        self.whisper_transcriber = None
        
        # Audio configuration
        self.recording = False
        self.audio_buffer = []
        self.accumulated_audio = b''
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.p = None
        self.stream = None
        
        # Configuration for pause detection
        self.last_audio_commit_time = 0
        self.silence_threshold = 500  # RMS value to define silence
        self.pause_duration = 0.7       # seconds of pause to trigger commit
        self.min_commit_interval = 1.5  # minimum interval between commits
        self.is_speaking = False
        self.silence_start_time = 0
        self.last_commit_time = 0
        self.response_pending = False
        
        # Buffer to accumulate audio transcription deltas and response
        self._response_transcript_buffer = ""
        self._response_buffer = ""
        
        # Variable to hold the final transcription
        self.current_text = ""
        
        # Path to the system prompt file
        self.system_prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "system_prompt.json"
        )

    def _load_system_prompt(self):
        """Loads the system prompt from the external JSON file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                system_message = json.load(f)
            logger.info("System prompt loaded from file: %s", self.system_prompt_path)
            return system_message
        except Exception as e:
            logger.error("Error loading system prompt from file: %s", str(e))
            # Fallback to default system prompt
            return {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{
                        "type": "input_text",
                        "text": "You are an AI assistant for job interviews, specialized in questions for software engineers. Respond concisely and structured."
                    }]
                }
            }

    async def realtime_session(self):
        """Manages a session for communication via Realtime API."""
        try:
            self.transcription_signal.emit("Connecting to Realtime API...")
            
            import websocket
            import json
            import threading
            
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
            headers = [
                "Authorization: Bearer " + os.getenv('OPENAI_API_KEY'),
                "OpenAI-Beta: realtime=v1",
                "Content-Type: application/json"
            ]
            
            self.connected = False
            self.websocket = None
            self.websocket_thread = None
            
            def on_open(ws):
                logger.info("WebSocket connection established")
                self.connected = True
                self.connection_status_signal.emit(True)
                self.last_event_time = time.time()
                self.current_text = ""
                
                session_config = {
                    "event_id": "event_123",
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": "You are a helpful assistant.",
                        "voice": "sage",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "whisper-1"
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": True
                        },
                        "tool_choice": "auto",
                        "temperature": 0.8,
                        "max_response_output_tokens": "inf"
                    }
                }
                
                try:
                    ws.send(json.dumps(session_config))
                    logger.info("Session configuration sent (audio and text)")
                except Exception as e:
                    logger.error("Error sending configuration: " + str(e))
                
                # Load system prompt from external file
                system_message = self._load_system_prompt()
                
                try:
                    ws.send(json.dumps(system_message))
                    logger.info("System prompt message sent")
                except Exception as e:
                    logger.error("Error sending system prompt message: " + str(e))
                
                response_request = {
                    "type": "response.create",
                    "response": {"modalities": ["text"]}
                }
                try:
                    ws.send(json.dumps(response_request))
                    logger.info("Response request sent")
                except Exception as e:
                    logger.error("Error sending response request: " + str(e))
                
                self.transcription_signal.emit("Connected! Ready for the interview. Speak to ask questions.")
            
            def on_message(ws, message):
                try:
                    event = json.loads(message)
                    event_type = event.get('type', 'unknown')
                    logger.debug(f"Event received: {event_type}")
                    
                    if event_type == 'response.audio_transcript.delta':
                        delta = event.get('delta', '')
                        self._response_transcript_buffer += delta
                        logger.debug("Accumulated transcript delta: %s", delta)
                    elif event_type == 'response.audio_transcript.done':
                        transcribed_text = self._response_transcript_buffer.strip()
                        self.current_text = transcribed_text
                        self.response_signal.emit(transcribed_text)
                        logger.info("Final audio transcription: %s", transcribed_text)
                        self._response_transcript_buffer = ""
                    elif event_type == 'response.text.delta':
                        delta = event.get('delta', '')
                        self._response_buffer += delta
                        logger.debug("Accumulated delta: %s", delta)
                    elif event_type in ('response.text.done', 'response.done'):
                        if self._response_buffer.strip():
                            self.response_signal.emit(self._response_buffer)
                            logger.info("Response completed: %s", self._response_buffer)
                            self._response_buffer = ""
                    elif event_type == 'error':
                        self.response_pending = False
                        error = event.get('error', {})
                        error_msg = error.get('message', 'Unknown error')
                        self.error_signal.emit(f"API Error: {error_msg}")
                        logger.error("Error received: %s", error_msg)
                    else:
                        logger.debug("Message received of type %s", event_type)
                except Exception as e:
                    logger.error("Exception in on_message: " + str(e))
            
            def on_error(ws, error):
                logger.error("WebSocket Error: " + str(error))
                self.error_signal.emit(f"Connection error: {error}")
                self.connected = False
                self.connection_status_signal.emit(False)
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
                self.connected = False
                self.connection_status_signal.emit(False)
                
                self.reconnect_attempts += 1
                reconnect_msg = f"\n[Connection lost. Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}]"
                self.transcription_buffer += reconnect_msg
                self.transcription_signal.emit(self.transcription_buffer)
                
                if self.running and self.reconnect_attempts <= self.max_reconnect_attempts:
                    self.error_signal.emit("Reconnection needed")
            
            self.websocket = websocket.WebSocketApp(
                url,
                header=headers,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.reconnect_attempts = 0
            
            def run_websocket():
                self.websocket.run_forever()
            
            self.websocket_thread = threading.Thread(target=run_websocket)
            self.websocket_thread.daemon = True
            self.websocket_thread.start()
            
            try:
                while self.running:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
            finally:
                if self.websocket and self.connected:
                    try:
                        self.websocket.close()
                    except Exception as e:
                        logger.error("Error closing websocket: " + str(e))
                logger.info("WebSocket session ended")
        except Exception as e:
            error_msg = f"Critical error: {e}"
            self.error_signal.emit(error_msg)
            logger.error(error_msg)
        finally:
            self.connected = False
            self.connection_status_signal.emit(False)
    
    def run(self):
        """Starts the asynchronous loop for the session."""
        self.running = True
        try:
            asyncio.run(self.realtime_session())
        except Exception as e:
            logger.error("Error in asynchronous thread: " + str(e))
        logger.info("Communication thread ended")
            
    def stop(self):
        """Stops communication and terminates all pending processes."""
        logger.info("Stop communication request received")
        self.running = False
        self.recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error("Error stopping stream: " + str(e))
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                logger.error("Error terminating PyAudio: " + str(e))
        if getattr(self, "websocket", None) and self.connected:
            try:
                logger.info("Closing WebSocket...")
                self.websocket.close()
                logger.info("WebSocket closed")
            except Exception as e:
                logger.error("Error closing websocket: " + str(e))
        logger.info("Termination request completed. Waiting for pending threads to close...")
        time.sleep(0.5)
    
    def start_recording(self):
        """Starts audio recording from the microphone."""
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
        except Exception as e:
            logger.error("Error initializing PyAudio: " + str(e))
            self.error_signal.emit("Audio initialization error")
            return
        threading.Thread(target=self._record_audio, daemon=True).start()
        logger.info("Audio recording started")
        self.transcription_signal.emit("Recording... Speak now.")
    
    # Then modify the start_recording method
    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            return
            
        self.recording = True
        
        if self.use_local_transcription:
            # Use local Whisper.cpp transcription
            if not self.whisper_transcriber:
                self.whisper_transcriber = WhisperTranscriber(
                    model_size="base",  # Can be configured based on performance needs
                    device="auto",      # Will use GPU if available
                    on_transcription=self._handle_local_transcription
                )
            self.whisper_transcriber.start_recording()
            self.transcription_signal.emit("Recording with local transcription...")
        else:
            # Use remote API for transcription
            self.start_recording()
    
    # Add a method to handle local transcription results
    def _handle_local_transcription(self, text):
        """Handle transcription results from local Whisper model."""
        if not text:
            return
            
        # Update the current text
        self.current_text = text
        
        # Emit the transcription signal
        self.transcription_signal.emit(text)
        
        # Add to chat history for context
        if not hasattr(self, 'chat_history'):
            self.chat_history = []
        self.chat_history.append({"role": "user", "content": text})
        
        # Send the text to the model for a response
        self.send_text(text)
    
    # Modify the stop_recording method
    def stop_recording(self):
        """Stop recording audio."""
        if not self.recording:
            return
            
        self.recording = False
        
        if self.use_local_transcription and self.whisper_transcriber:
            self.whisper_transcriber.stop_recording()
        else:
            self.stop_recording()
    
    def _record_audio(self):
        """Records audio in a loop until stopped."""
        try:
            self.last_commit_time = time.time()
            while self.recording and self.stream:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_buffer.append(data)
                self.accumulated_audio += data
                rms = self._calculate_rms(data)
                current_time = time.time()
                if rms > self.silence_threshold:
                    if not self.is_speaking:
                        logger.info(f"Speech start detected (RMS: {rms})")
                        self.is_speaking = True
                    self.silence_start_time = 0
                else:
                    if self.is_speaking:
                        if self.silence_start_time == 0:
                            self.silence_start_time = current_time
                        elif (current_time - self.silence_start_time >= self.pause_duration and 
                              current_time - self.last_commit_time >= self.min_commit_interval):
                            if len(self.accumulated_audio) >= 3200:
                                logger.info(f"Pause detected ({current_time - self.silence_start_time:.2f}s) - sending partial audio")
                                self._send_entire_audio_message()
                                commit_message = {"type": "input_audio_buffer.commit"}
                                self.websocket.send(json.dumps(commit_message))
                                logger.info("Partial audio commit sent")
                                if not self.response_pending:
                                    response_request = {"type": "response.create", "response": {"modalities": ["text"]}}
                                    self.websocket.send(json.dumps(response_request))
                                    logger.info("Response request sent after partial audio")
                                    self.response_pending = True
                                self.last_commit_time = current_time
                                self.silence_start_time = 0
                                self.accumulated_audio = b''
                                self.audio_buffer = []
                                self.is_speaking = False
                            else:
                                logger.info(f"Pause detected but buffer too small ({len(self.accumulated_audio)} bytes), continuing to accumulate")
                if not self.recording:
                    logger.info("Recording stop detected, exiting audio capture loop.")
                    break
        except Exception as e:
            logger.error("Error during recording: " + str(e))
            self.error_signal.emit("Recording error: " + str(e))
            self.recording = False
    
    def _calculate_rms(self, data):
        """Calculates the RMS value for an audio buffer."""
        try:
            shorts = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(np.square(shorts)))
            if np.isnan(rms):
                logger.warning("RMS calculation produced NaN, returning 0")
                return 0
            return rms
        except Exception as e:
            logger.error("Error calculating RMS: " + str(e))
            return 0
    
    def _send_entire_audio_message(self):
        """Sends the entire accumulated audio as a single message."""
        if not self.accumulated_audio or not self.connected:
            return
        try:
            base64_audio = base64.b64encode(self.accumulated_audio).decode('ascii')
            audio_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": base64_audio}]
                }
            }
            self.websocket.send(json.dumps(audio_message))
            logger.info("Single audio message sent")
        except Exception as e:
            logger.error("Error sending single audio message: " + str(e))
            
    # Add imports at the top
    import os
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    def send_text(self, text):
        """Send text to the model."""
        try:
            if not self.connected:
                return False
                
            if self.api_type == "gemini":
                # Gemini API call
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                # Convert chat history to Gemini format
                gemini_history = []
                for msg in self.chat_history:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg["content"]]})
                
                # Add the new message
                gemini_history.append({"role": "user", "parts": [text]})
                
                # Create the chat session
                chat = self.client.GenerativeModel(
                    model="gemini-pro",
                    safety_settings=safety_settings
                ).start_chat(history=gemini_history[:-1])
                
                # Get response
                response = chat.send_message(text)
                response_text = response.text
                
                # Update chat history
                self.chat_history.append({"role": "user", "content": text})
                self.chat_history.append({"role": "assistant", "content": response_text})
                
                # Emit the response signal
                self.response_signal.emit(response_text)
                return True
            else:
                # Other API call
                # ...
                return False
        except Exception as e:
            self.error_signal.emit(f"Error sending text: {str(e)}")
            return False