#!/usr/bin/env python3
"""
Session Manager Module for Intervista Assistant.
Handles conversation sessions and communication with Gemini API.
"""
import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from ..gemini_client import GeminiClient
from .audio_handler import AudioHandler
from .image_processor import ImageProcessor

# Configure logging
logger = logging.getLogger(__name__)

class SessionManager:
    """Manages a conversation session using Gemini API."""
    
    def __init__(self, session_id: str, gemini_client: GeminiClient):
        """Initialize a new session.
        
        Args:
            session_id: Unique identifier for the session
            gemini_client: Initialized Gemini client
        """
        self.session_id = session_id
        self.gemini_client = gemini_client
        self.recording = False
        self.text_thread = None
        self.last_activity = datetime.now()
        self.chat_history = []
        
        # Initialize handlers
        self.audio_handler = AudioHandler(gemini_client)
        self.image_processor = ImageProcessor(
            gemini_client,
            handle_response_callback=self.handle_response
        )
        
        # Queues for updates
        self.transcription_updates = []
        self.response_updates = []
        self.error_updates = []
        self.connection_updates = []
    
    def start_session(self) -> Tuple[bool, Optional[str]]:
        """Starts the session and initializes resources."""
        logger.info(f"Starting session {self.session_id}")
        
        try:
            if self.recording:
                return True, None
            
            # Initialize the text thread with Gemini
            if self.gemini_client.is_available():
                from ..gemini_realtime_thread import GeminiRealtimeThread
                
                callbacks = {
                    'on_transcription': lambda text: self.handle_transcription(text),
                    'on_response': lambda text: self.handle_response(text),
                    'on_error': lambda message: self.handle_error(message),
                    'on_connection_status': lambda connected: self.handle_connection_status(connected)
                }
                
                self.text_thread = GeminiRealtimeThread(callbacks)
                self.text_thread.start()
                
                # Mark as recording
                self.recording = True
                self.last_activity = datetime.now()
                
                # Wait for connection
                max_wait = 2
                wait_interval = 0.1
                waited = 0
                
                while waited < max_wait:
                    if self.text_thread.connected:
                        logger.info(f"Gemini connection established after {waited:.1f} seconds")
                        break
                    waited += wait_interval
                    time.sleep(wait_interval)
                
                return True, None
            else:
                error_msg = "Gemini API not available"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error starting session: {str(e)}"
            logger.error(error_msg)
            self.handle_error(error_msg)
            return False, error_msg
    
    def end_session(self) -> bool:
        """Ends the current session and frees resources."""
        logger.info(f"Ending session {self.session_id}")
        
        try:
            if not self.recording and not self.text_thread:
                return True
            
            if self.text_thread:
                if self.text_thread.recording:
                    self.text_thread.stop_recording()
                
                self.text_thread.stop()
                time.sleep(1)
                self.text_thread = None
            
            self.recording = False
            logger.info(f"Session {self.session_id} ended successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session {self.session_id}: {str(e)}")
            return False
    
    def handle_transcription(self, text: str) -> None:
        """Handles transcription updates from Gemini API."""
        self.last_activity = datetime.now()
        if not text:
            return
        
        if text.endswith("[fine]") or text.endswith("[end]"):
            clean_text = text.replace("[fine]", "").replace("[end]", "").strip()
            if clean_text:
                self.chat_history.append({"role": "user", "content": clean_text})
        
        timestamp = datetime.now().isoformat()
        self.transcription_updates.append({
            "timestamp": timestamp,
            "text": text
        })
        
        logger.info(f"Transcription: {text[:50]}...")
    
    def handle_response(self, text: str, final: bool = False) -> None:
        """Handles response updates from Gemini API."""
        self.last_activity = datetime.now()
        if not text:
            return
        
        if not self.chat_history or self.chat_history[-1]["role"] != "assistant":
            self.chat_history.append({"role": "assistant", "content": text})
        else:
            current_time = datetime.now().strftime("%H:%M:%S")
            previous_content = self.chat_history[-1]["content"]
            self.chat_history[-1]["content"] = f"{previous_content}\n--- Response at {current_time} ---\n{text}"
        
        timestamp = datetime.now().isoformat()
        self.response_updates.append({
            "timestamp": timestamp,
            "text": text,
            "final": final
        })
        
        logger.info(f"Response: {text[:50]}...")
    
    def handle_error(self, message: str) -> None:
        """Handles error updates."""
        if "buffer too small" in message or "Conversation already has an active response" in message:
            logger.warning(f"Ignored error (log only): {message}")
            return
        
        timestamp = datetime.now().isoformat()
        self.error_updates.append({
            "timestamp": timestamp,
            "message": message
        })
        logger.error(f"Error in session {self.session_id}: {message}")
    
    def handle_connection_status(self, connected: bool) -> None:
        """Handles connection status updates."""
        timestamp = datetime.now().isoformat()
        self.connection_updates.append({
            "timestamp": timestamp,
            "connected": connected
        })
        logger.info(f"Connection status for session {self.session_id}: {'connected' if connected else 'disconnected'}")
    
    def get_updates(self, update_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gets all updates of the specified type."""
        if update_type == "transcription":
            return self.transcription_updates
        elif update_type == "response":
            return self.response_updates
        elif update_type == "error":
            return self.error_updates
        elif update_type == "connection":
            return self.connection_updates
        
        # If no type specified, return all updates as a flat list
        all_updates = []
        
        for update in self.transcription_updates:
            update_copy = update.copy()
            update_copy["type"] = "transcription"
            all_updates.append(update_copy)
        
        for update in self.response_updates:
            update_copy = update.copy()
            update_copy["type"] = "response"
            all_updates.append(update_copy)
        
        for update in self.error_updates:
            update_copy = update.copy()
            update_copy["type"] = "error"
            all_updates.append(update_copy)
        
        for update in self.connection_updates:
            update_copy = update.copy()
            update_copy["type"] = "connection"
            all_updates.append(update_copy)
        
        # Sort updates by timestamp
        all_updates.sort(key=lambda x: x["timestamp"])
        return all_updates
    
    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the session."""
        return {
            "session_id": self.session_id,
            "recording": self.recording,
            "connected": self.text_thread.connected if self.text_thread else False,
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.chat_history)
        }