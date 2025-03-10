#!/usr/bin/env python3
"""
Socket Handlers Module for Intervista Assistant.
Handles Socket.IO events and real-time communication.
"""
import os
import time
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class SocketHandlers:
    """Handles Socket.IO events for the Intervista Assistant."""
    
    def __init__(self, socketio, active_sessions, gemini_client):
        """Initialize the socket handlers.
        
        Args:
            socketio: Socket.IO instance
            active_sessions: Dictionary of active sessions
            gemini_client: Initialized Gemini client
        """
        self.socketio = socketio
        self.active_sessions = active_sessions
        self.gemini_client = gemini_client
        
        # Register socket handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all Socket.IO event handlers."""
        self.socketio.on('connect')(self.handle_connect)
        self.socketio.on('disconnect')(self.handle_disconnect)
        self.socketio.on('audio_data')(self.handle_audio_data)
        self.socketio.on('audio_data_gemini')(self.handle_audio_data_gemini)
        self.socketio.on('set_api_preference')(self.handle_api_preference)
    
    def handle_connect(self):
        """Handles a Socket.IO client connection."""
        from flask import request
        logger.info(f"New Socket.IO client connected: {request.sid}")
    
    def handle_disconnect(self):
        """Handles a Socket.IO client disconnection."""
        from flask import request
        logger.info(f"Socket.IO client disconnected: {request.sid}")
    
    def handle_api_preference(self, data):
        """Handle client preference for AI API provider."""
        session_id = data.get('session_id')
        preference = data.get('preference')
        
        if not session_id or session_id not in self.active_sessions:
            return {'success': False, 'error': 'Invalid session ID'}
        
        session = self.active_sessions[session_id]
        
        logger.info(f"Setting API preference for session {session_id} to {preference}")
        
        # Store the preference in the session
        session.api_preference = preference
        
        # If using Gemini, ensure the client is initialized
        if preference == 'gemini' and not self.gemini_client.is_available():
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                self.gemini_client.initialize(gemini_api_key)
                logger.info("Gemini client initialized for session")
            else:
                logger.error("Gemini API key not found in environment")
                return {'success': False, 'error': 'Gemini API key not configured'}
        
        return {'success': True}
    
    def handle_audio_data_gemini(self, data):
        """Handle audio data specifically for Gemini API processing."""
        session_id = data.get('session_id')
        audio_data_base64 = data.get('audio_data')
        sample_rate = data.get('sample_rate', 16000)
        encoding = data.get('encoding', 'LINEAR16')
        
        if not session_id or session_id not in self.active_sessions:
            return {'success': False, 'error': 'Invalid session ID'}
        
        session = self.active_sessions[session_id]
        
        try:
            # Use the AudioProcessor to process the audio
            from .audio_processor import AudioProcessor
            
            # Create an audio processor if not already created
            if not hasattr(self, 'audio_processor'):
                self.audio_processor = AudioProcessor(self.gemini_client)
            
            # Process the audio
            result = self.audio_processor.process_base64_audio(
                audio_data_base64, 
                sample_rate=sample_rate, 
                encoding=encoding
            )
            
            if result and 'transcription' in result:
                transcription = result['transcription']
                
                # Handle the transcription result
                session.handle_transcription(transcription)
                
                # If there's a response, handle it
                if 'response' in result and result['response']:
                    session.handle_response(result['response'])
                
                return {'received': True, 'samples': len(base64.b64decode(audio_data_base64)) // 2}
            else:
                error_msg = result.get('error', 'Unknown error in audio processing')
                logger.error(f"Audio processing failed: {error_msg}")
                return {'received': False, 'error': error_msg}
                
        except Exception as e:
            error_message = f"Error processing audio data: {str(e)}"
            logger.error(error_message)
            return {'received': False, 'error': error_message}
    
    def handle_audio_data(self, data):
        """Handles audio data received from the client."""
        from flask import request
        from flask_socketio import emit
        
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')
        
        acknowledgement = {
            'received': False, 
            'error': None,
            'timestamp': time.time(),
            'session_active': session_id in self.active_sessions
        }
        
        try:
            # Log only essential metadata about the audio data
            audio_type = type(audio_data).__name__
            if isinstance(audio_data, list):
                audio_length = len(audio_data)
                logger.info(f"[SOCKET.IO:AUDIO] Received audio data for session {session_id}: {audio_length} samples, type={audio_type}")
                acknowledgement['samples'] = audio_length
            elif isinstance(audio_data, bytes):
                logger.info(f"[SOCKET.IO:AUDIO] Received audio data for session {session_id}: {len(audio_data)} bytes, type={audio_type}")
                acknowledgement['bytes'] = len(audio_data)
            else:
                logger.info(f"[SOCKET.IO:AUDIO] Received audio data for session {session_id}: type={audio_type}")
                acknowledgement['type'] = audio_type
            
            if session_id not in self.active_sessions:
                logger.error(f"[SOCKET.IO:AUDIO] Session {session_id} not found")
                emit('error', {'message': 'Session not found'})
                acknowledgement['error'] = 'Session not found'
                return acknowledgement
            
            session = self.active_sessions[session_id]
            acknowledgement['session_recording'] = session.recording
            
            # Check if the session is recording
            if not session.recording or not session.text_thread:
                logger.warning(f"[SOCKET.IO:AUDIO] Session {session_id} not recording")
                emit('error', {'message': 'Session not recording'})
                acknowledgement['error'] = 'Session not recording'
                return acknowledgement
            
            # Check if audio_data is not empty
            if not audio_data:
                logger.warning(f"[SOCKET.IO:AUDIO] Empty audio data received for session {session_id}")
                acknowledgement['error'] = 'Empty audio data'
                return acknowledgement
                
            try:
                # Update the last activity timestamp
                session.last_activity = datetime.now()
                
                # Verify and convert the audio to the correct format
                if isinstance(audio_data, list):
                    # Check for valid values based on 16-bit PCM requirements
                    if any(not isinstance(sample, (int, float)) for sample in audio_data):
                        logger.error(f"[SOCKET.IO:AUDIO] Invalid audio data format: non-numeric samples")
                        acknowledgement['error'] = 'Invalid audio data format'
                        emit('error', {'message': 'Invalid audio data format: non-numeric samples'})
                        return acknowledgement
                    
                    # Ensure there are enough samples (at least 10ms of audio at 24kHz = 240 samples)
                    if len(audio_data) < 240:
                        logger.warning(f"[SOCKET.IO:AUDIO] Audio clip too short: {len(audio_data)} samples")
                        acknowledgement['warning'] = 'Audio clip too short'
                    
                    # Normalize values to ensure compatibility with 16-bit PCM (-32768 to 32767)
                    max_value = max(abs(sample) if isinstance(sample, int) else abs(float(sample)) for sample in audio_data)
                    
                    # If values are too large or too small, normalize them
                    if max_value > 32767 or max_value < 1:
                        logger.info(f"[SOCKET.IO:AUDIO] Normalizing audio samples: max value = {max_value}")
                        if max_value > 0:  # Avoid division by zero
                            scaling_factor = 32767.0 / max_value
                            audio_data = [int(sample * scaling_factor) for sample in audio_data]
                    
                    # Ensure it is of the correct type for processing (16-bit PCM)
                    audio_data = np.array(audio_data, dtype=np.int16)
                    logger.info(f"[SOCKET.IO:AUDIO] Converted list to numpy array for session {session_id}: {len(audio_data)} samples")
                    
                    # Calculate the duration of the received audio (assuming 24kHz)
                    audio_duration_ms = (len(audio_data) / 24000) * 1000
                    logger.info(f"[SOCKET.IO:AUDIO] Approximate audio duration: {audio_duration_ms:.2f}ms at 24kHz")
                    acknowledgement['duration_ms'] = audio_duration_ms
                
                # Check connection status
                websocket_connected = False
                websocket_reconnect_attempts = 0
                if session.text_thread:
                    websocket_connected = session.text_thread.connected
                    if hasattr(session.text_thread, 'reconnect_attempts'):
                        websocket_reconnect_attempts = session.text_thread.reconnect_attempts
                
                acknowledgement['websocket_connected'] = websocket_connected
                acknowledgement['websocket_reconnect_attempts'] = websocket_reconnect_attempts
                
                # Add the audio data to the text thread buffer
                if session.text_thread:
                    # Now we can send the data to the text thread
                    success = session.text_thread.add_audio_data(audio_data)
                    acknowledgement['received'] = success
                    
                    # Add info about the data size
                    if isinstance(audio_data, np.ndarray):
                        acknowledgement['bytes_processed'] = audio_data.nbytes
                    elif isinstance(audio_data, bytes):
                        acknowledgement['bytes_processed'] = len(audio_data)
                    else:
                        acknowledgement['bytes_processed'] = 'unknown'
                    
                    # Update the status for the frontend
                    if success:
                        emit('audio_processed', {'success': True, 'bytes': acknowledgement.get('bytes_processed', 0)})
                
            except Exception as e:
                logger.error(f"[SOCKET.IO:AUDIO] Error processing audio in text thread: {str(e)}")
                acknowledgement['error'] = f"Audio processing error: {str(e)}"
                emit('error', {'message': f'Audio processing error: {str(e)}'})
            
            return acknowledgement
            
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"[SOCKET.IO:AUDIO] Unexpected error: {str(e)}")
            acknowledgement['error'] = f"Unexpected error: {str(e)}"
            return acknowledgement