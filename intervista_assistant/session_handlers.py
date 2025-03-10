#!/usr/bin/env python3
"""
Session Handlers for Intervista Assistant.
Contains methods for handling various types of updates and messages.
"""
import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Import this here to avoid circular imports
from intervista_assistant.session_manager import SessionManager

def handle_transcription(self: SessionManager, text: str) -> None:
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

def handle_response(self: SessionManager, text: str, final: bool = False) -> None:
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

#!/usr/bin/env python3
"""
Session Handlers for Intervista Assistant.
Contains methods for handling various types of updates and messages.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

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

def get_updates(self, update_type=None):
    """
    Gets all updates of the specified type.
    If update_type is None, returns all updates.
    """
    if update_type == "transcription":
        return self.transcription_updates
    elif update_type == "response":
        return self.response_updates
    elif update_type == "error":
        return self.error_updates
    elif update_type == "connection":
        return self.connection_updates
    elif update_type is None:
        # If no type is specified, return all updates as a flat list
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
    else:
        return []

def send_text_message(self, text):
    """Sends a text message to the model."""
    self.last_activity = datetime.now()
    
    if not text:
        logger.warning("Empty text message received")
        return False
    
    if not self.text_thread:
        logger.error("No active session for sending text message")
        return False
    
    # Use the thread's send_text method
    try:
        # Make sure the thread is initialized and connected
        if hasattr(self.text_thread, 'send_text'):
            success = self.text_thread.send_text(text)
            if success:
                logger.info(f"Text message sent to model: {text[:50]}...")
                return True
            else:
                logger.error("Failed to send text message")
                return False
        else:
            logger.error("text_thread does not have the method send_text")
            return False
    except Exception as e:
        logger.error(f"Error sending text message: {str(e)}")
        return False