#!/usr/bin/env python3
"""
Session Manager for Intervista Assistant.
Handles conversation sessions and communication with AI models.
"""
import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages a conversation session, including communication with AI models.
    Each session is associated with a frontend client.
    """
    def __init__(self, session_id, socketio=None):
        """Initializes a new session."""
        self.session_id = session_id
        self.socketio = socketio
        self.recording = False
        self.text_thread = None
        self.last_activity = datetime.now()
        self.chat_history = []
        self.api_preference = os.getenv("DEFAULT_API", "openai").lower()
        
        # Queues for updates
        self.transcription_updates = []
        self.response_updates = []
        self.error_updates = []
        self.connection_updates = []
    
    def update_status(self, status_change=None):
        """Updates and emits the current session status."""
        status = self.get_status()
        if status_change:
            status["status_change"] = status_change
        
        # Emit the status update via Socket.IO to the specific room (session_id)
        if self.socketio:
            self.socketio.emit('session_status_update', status, room=self.session_id)
        
        return status

    def get_status(self):
        """Returns the current status of the session."""
        return {
            "session_id": self.session_id,
            "recording": self.recording,
            "connected": self.text_thread.connected if self.text_thread else False,
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.chat_history)
        }
    
    def start_session(self):
        """Starts the session and initializes resources."""
        logger.info(f"Starting session {self.session_id}")
        
        try:
            # Check if already recording
            if self.recording:
                return True, None
            
            # Import here to avoid circular imports
            try:
                from intervista_assistant.web_realtime_text_thread import WebRealtimeTextThread
                from intervista_assistant.gemini_realtime_thread import GeminiRealtimeThread
                gemini_available = True
            except ImportError:
                from web_realtime_text_thread import WebRealtimeTextThread
                gemini_available = False
                
            # Initialize the appropriate text thread based on API preference
            if self.api_preference == 'gemini' and gemini_available:
                logger.info(f"Using Gemini API for session {self.session_id}")
                # Set environment variable to use Gemini
                os.environ['USE_OPENAI_FOR_TEXT'] = 'false'
                os.environ['USE_GEMINI_FOR_TEXT'] = 'true'
                
                # Create callbacks for the Gemini thread
                callbacks = {
                    'on_transcription': lambda text: self.handle_transcription(text),
                    'on_response': lambda text: self.handle_response(text),
                    'on_error': lambda message: self.handle_error(message),
                    'on_connection_status': lambda connected: self.handle_connection_status(connected)
                }
                
                # Create the Gemini thread
                self.text_thread = GeminiRealtimeThread(callbacks)
            else:
                logger.info(f"Using OpenAI API for session {self.session_id}")
                
                # Create callbacks for the OpenAI thread
                callbacks = {
                    'on_transcription': lambda text: self.handle_transcription(text),
                    'on_response': lambda text: self.handle_response(text),
                    'on_error': lambda message: self.handle_error(message),
                    'on_connection_status': lambda connected: self.handle_connection_status(connected)
                }
                
                # Create the OpenAI thread
                self.text_thread = WebRealtimeTextThread(callbacks)
            
            # Start the thread
            self.text_thread.start()
            
            # Mark the session as recording
            self.recording = True
            
            # Update the last activity timestamp
            self.last_activity = datetime.now()
            
            # Wait for the thread to initialize (max 2 seconds)
            max_wait = 2  # seconds
            wait_interval = 0.1  # seconds
            waited = 0
            
            while waited < max_wait:
                if self.text_thread.connected:
                    logger.info(f"Connection established after {waited:.1f} seconds")
                    break
                waited += wait_interval
                time.sleep(wait_interval)
            
            logger.info(f"Session {self.session_id} started successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Error starting session: {str(e)}"
            logger.error(error_msg)
            self.handle_error(error_msg)
            return False, error_msg
    
    def end_session(self):
        """Ends the current session and frees resources."""
        logger.info(f"Ending session {self.session_id}")
        
        try:
            # Check if the session is already ended
            if not self.recording and not self.text_thread:
                return True
                
            # Handle the real-time communication thread
            if self.text_thread:
                # Stop recording if active
                if self.text_thread.recording:
                    self.text_thread.stop_recording()
                
                # Stop the thread
                self.text_thread.stop()
                time.sleep(1)  # Wait a second for completion
                
                # Cleanup
                self.text_thread = None
                
            # Update the state
            self.recording = False
            logger.info(f"Session {self.session_id} ended successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session {self.session_id}: {str(e)}")
            return False
    
    def handle_transcription(self, text):
        """Handles transcription updates from the real-time API."""
        self.last_activity = datetime.now()
        if not text:
            return
            
        # Update chat history only for complete transcriptions
        if text.endswith("[fine]") or text.endswith("[end]"):
            clean_text = text.replace("[fine]", "").replace("[end]", "").strip()
            if clean_text:
                self.chat_history.append({"role": "user", "content": clean_text})
        
        # Add the update to the queue
        timestamp = datetime.now().isoformat()
        self.transcription_updates.append({
            "timestamp": timestamp,
            "text": text
        })
        
        logger.info(f"Transcription: {text[:50]}...")
    
    def handle_response(self, text, final=False):
        """Handles response updates from the real-time API."""
        self.last_activity = datetime.now()
        if not text:
            return
            
        # Update chat history
        if (not self.chat_history or self.chat_history[-1]["role"] != "assistant"):
            self.chat_history.append({"role": "assistant", "content": text})
        elif self.chat_history and self.chat_history[-1]["role"] == "assistant":
            current_time = datetime.now().strftime("%H:%M:%S")
            previous_content = self.chat_history[-1]["content"]
            self.chat_history[-1]["content"] = f"{previous_content}\n--- Response at {current_time} ---\n{text}"
        
        # Add the update to the queue
        timestamp = datetime.now().isoformat()
        self.response_updates.append({
            "timestamp": timestamp,
            "text": text,
            "final": final
        })
        
        logger.info(f"Response: {text[:50]}...")
        
    def handle_error(self, message):
        """Handles error updates."""
        # Ignore some known errors that do not require notification
        if "buffer too small" in message or "Conversation already has an active response" in message:
            logger.warning(f"Ignored error (log only): {message}")
            return
            
        timestamp = datetime.now().isoformat()
        self.error_updates.append({
            "timestamp": timestamp,
            "message": message
        })
        logger.error(f"Error in session {self.session_id}: {message}")
    
    def handle_connection_status(self, connected):
        """Handles connection status updates."""
        timestamp = datetime.now().isoformat()
        self.connection_updates.append({
            "timestamp": timestamp,
            "connected": connected
        })
        logger.info(f"Connection status for session {self.session_id}: {'connected' if connected else 'disconnected'}")