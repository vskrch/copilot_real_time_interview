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
    
    def start_session(self):
        """Starts the session and initializes resources."""
        logger.info(f"Starting session {self.session_id}")
        
        try:
            # Check if already recording
            if self.recording:
                return True, None
            
            # Import Gemini thread
            from ..gemini_realtime_thread import GeminiRealtimeThread
            
            # Create callbacks dictionary
            callbacks = {
                'on_transcription': self.handle_transcription,
                'on_response': self.handle_response,
                'on_error': self.handle_error,
                'on_connection_status': self.handle_connection_status
            }
            
            # Create and start the Gemini thread
            self.text_thread = GeminiRealtimeThread(callbacks=callbacks)
            self.text_thread.start()
            
            # Set recording flag
            self.recording = True
            
            # Update status
            self.update_status("Session started")
            
            return True, None
            
        except Exception as e:
            error_msg = f"Error starting session: {str(e)}"
            logger.error(error_msg)
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

    def start_think_process(self):
        """
        Starts an advanced thinking process that generates a summary
        and a detailed solution based on the conversation.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        logger.info(f"Starting think process for session {self.session_id}")
        
        try:
            # Check if we have enough conversation history
            if not self.chat_history or len(self.chat_history) < 2:
                error_msg = "Not enough conversation history to analyze"
                logger.warning(error_msg)
                return False, error_msg
            
            # Start the thinking process in a separate thread
            threading.Thread(
                target=self._process_thinking_async,
                daemon=True
            ).start()
            
            return True, None
            
        except Exception as e:
            error_msg = f"Error starting thinking process: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _process_thinking_async(self):
        """Performs the thinking process asynchronously using Gemini."""
        try:
            # First generate a summary
            self.handle_response("🧠 Analyzing our conversation...", final=False)
            summary = self._generate_summary()
            
            # Notify the user that the summary is ready
            self.handle_response(f"**CONVERSATION ANALYSIS:**\n\n{summary}", final=False)
            
            # Generate a detailed solution based on the summary
            self.handle_response("🚀 Generating detailed insights...", final=False)
            solution = self._generate_solution(summary)
            
            # Notify the user that the solution is ready
            self.handle_response(f"**DETAILED SOLUTION:**\n\n{solution}", final=True)
            
            logger.info(f"Thinking process completed successfully for session {self.session_id}")
            
        except Exception as e:
            error_message = f"Error in the thinking process: {str(e)}"
            logger.error(error_message)
            self.handle_error(error_message)

    def _generate_summary(self):
        """Generates a summary of the conversation using Gemini API."""
        try:
            # Create a summary prompt
            summary_prompt = """Analyze the conversation history and create a concise summary. 
            Focus on:
            1. Key problems or questions discussed
            2. Important context
            3. Any programming challenges mentioned
            4. Current state of the discussion
            
            Your summary should be comprehensive but brief, highlighting the most important aspects 
            that would help solve any programming or logical problems mentioned."""
            
            # Format history for Gemini
            formatted_history = []
            for msg in self.chat_history:
                if msg["role"] != "system":
                    formatted_history.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [msg["content"]]
                    })
            
            # Process with Gemini
            response = self.gemini_client.process_text(
                summary_prompt,
                session_history=formatted_history
            )
            
            if not response:
                return "Unable to generate summary. Please try again."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _generate_solution(self, summary):
        """
        Generates a detailed solution based on the summary.
        
        Args:
            summary: The conversation summary
            
        Returns:
            str: The generated solution
        """
        try:
            # Build the prompt
            prompt = f"""
            I'm working on a programming or logical task. Here's the context and problem:
            
            # CONTEXT
            {summary}
            
            Please analyze this situation and:
            1. Identify the core problem or challenge
            2. Develop a structured approach to solve it
            3. Provide a detailed solution with code if applicable
            4. Explain your reasoning
            """
            
            # Format history for Gemini
            formatted_history = []
            for msg in self.chat_history:
                if msg["role"] != "system":
                    formatted_history.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [msg["content"]]
                    })
            
            # Add the summary as context
            formatted_history.append({
                "role": "user",
                "parts": [f"Here's a summary of our conversation: {summary}"]
            })
            
            # Process with Gemini
            response = self.gemini_client.process_text(
                prompt,
                session_history=formatted_history
            )
            
            if not response:
                return "Unable to generate solution. Please try again."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating solution: {str(e)}")
            return f"Error generating solution: {str(e)}"