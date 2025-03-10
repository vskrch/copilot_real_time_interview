#!/usr/bin/env python3
"""
Gemini API Client for Intervista Assistant.
Handles communication with Google's Gemini API.
"""
import os
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image
import io

# Configure logging
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.initialized = False
        self.api_key = None
        self.models = {}
    
    def initialize(self, api_key=None):
        """Initialize the Gemini API client.
        
        Args:
            api_key: Gemini API key (optional, will use environment variable if not provided)
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Get API key from parameter or environment
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                logger.error("No Gemini API key provided")
                return False
                
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Initialize models dictionary
            self.models = {}
            
            # Try to initialize models with explicit error handling
            try:
                # First check if models are available
                model_list = genai.list_models()
                available_models = [model.name for model in model_list]
                logger.info(f"Available Gemini models: {available_models}")
                
                # Check for specific models
                if any('gemini-pro' in model for model in available_models):
                    try:
                        self.models['gemini-pro'] = genai.GenerativeModel('gemini-pro')
                        logger.info("Successfully initialized gemini-pro model")
                    except Exception as e:
                        logger.error(f"Failed to initialize gemini-pro model: {str(e)}")
                else:
                    logger.warning("gemini-pro model not available in your API key's allowed models")
                
                if any('gemini-pro-vision' in model for model in available_models):
                    try:
                        self.models['gemini-pro-vision'] = genai.GenerativeModel('gemini-pro-vision')
                        logger.info("Successfully initialized gemini-pro-vision model")
                    except Exception as e:
                        logger.error(f"Failed to initialize gemini-pro-vision model: {str(e)}")
                else:
                    logger.warning("gemini-pro-vision model not available in your API key's allowed models")
                    
            except Exception as e:
                logger.error(f"Error listing models: {str(e)}")
                # Try direct initialization without checking
                try:
                    self.models['gemini-pro'] = genai.GenerativeModel('gemini-pro')
                    logger.info("Directly initialized gemini-pro model")
                except Exception as e2:
                    logger.error(f"Direct initialization of gemini-pro failed: {str(e2)}")
                
                try:
                    self.models['gemini-pro-vision'] = genai.GenerativeModel('gemini-pro-vision')
                    logger.info("Directly initialized gemini-pro-vision model")
                except Exception as e2:
                    logger.error(f"Direct initialization of gemini-pro-vision failed: {str(e2)}")
            
            # Set aliases for backward compatibility
            if 'gemini-pro' in self.models:
                self.models['chat'] = self.models['gemini-pro']
            
            if 'gemini-pro-vision' in self.models:
                self.models['vision'] = self.models['gemini-pro-vision']
            
            # Check if we have at least one working model
            if not self.models:
                logger.error("No Gemini models could be initialized")
                return False
            
            self.initialized = True
            logger.info("Gemini API client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Gemini API client: {str(e)}")
            self.initialized = False
            return False
    
    def is_available(self):
        """Check if the Gemini client is initialized and ready to use.
        
        Returns:
            bool: True if the client is initialized and has at least one model
        """
        return hasattr(self, 'initialized') and self.initialized and hasattr(self, 'models') and bool(self.models)
    
    def process_text(self, prompt, session_history=None):
        """Process text using Gemini API with conversation history.
        
        Args:
            prompt: Text prompt to process
            session_history: Optional conversation history
            
        Returns:
            str: Generated response or None if failed
        """
        try:
            if not self.is_available():
                logger.error("Gemini API not initialized")
                return None
            
            # Try to use gemini-pro model
            model = self.models.get('gemini-pro') or self.models.get('chat')
            
            if not model:
                logger.error("No text generation model available")
                return None
            
            # Generate content with history if provided
            if session_history:
                try:
                    # Create a chat session
                    chat = model.start_chat(history=session_history)
                    response = chat.send_message(prompt)
                    return response.text
                except Exception as e:
                    logger.error(f"Error generating text with history: {str(e)}")
                    # Fall back to simple generation without history
                    response = model.generate_content(prompt)
                    return response.text
            else:
                # Simple generation without history
                response = model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return None
    
    def generate_text(self, prompt, context=None):
        """Generate text using Gemini API."""
        if not self.is_available():
            return False, "Gemini API not initialized"
        
        try:
            # Try to use gemini-pro model first, then fall back to 'chat' alias
            model = self.models.get('gemini-pro') or self.models.get('chat')
            
            # If still no model, try to get any available model
            if not model and self.models:
                model = next(iter(self.models.values()))
                logger.info(f"Using alternative model for text generation: {model}")
            
            if not model:
                return False, "No text generation model available"
            
            # Generate content
            response = model.generate_content(prompt)
            return True, response.text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return False, str(e)
            
    def analyze_image(self, image_data, prompt):
        """Analyze an image using Gemini API.
        
        Args:
            image_data: Base64 encoded image data
            prompt: Text prompt for image analysis
            
        Returns:
            Tuple of (success, response)
        """
        if not self.is_available():
            # Try to initialize with environment variable
            import os
            from dotenv import load_dotenv
            
            # Try to load from .env file first
            try:
                env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from {env_path}")
            except Exception as e:
                logger.warning(f"Could not load .env file: {str(e)}")
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return False, "Gemini API key not found. Please set GEMINI_API_KEY environment variable or add it to .env file."
                
            success = self.initialize(api_key)
            
            # Check again after initialization attempt
            if not self.is_available():
                return False, "Gemini API initialization failed. Please check your API key."
            
        try:
            # Check if we have the vision model
            if 'gemini-pro-vision' not in self.models and 'vision' not in self.models:
                # Try to initialize it specifically
                try:
                    self.models['gemini-pro-vision'] = genai.GenerativeModel('gemini-pro-vision')
                    self.models['vision'] = self.models['gemini-pro-vision']
                    logger.info("Late initialization of gemini-pro-vision model succeeded")
                except Exception as e:
                    logger.error(f"Late initialization of gemini-pro-vision failed: {str(e)}")
                    return False, "The required 'gemini-pro-vision' model is not available with your API key"
            
            # Get the vision model
            model = self.models.get('gemini-pro-vision') or self.models.get('vision')
            if not model:
                return False, "No vision model available"
            
            # Decode base64 image
            try:
                # Handle both formats: with or without data URL prefix
                if ',' in image_data:
                    image_bytes = base64.b64decode(image_data.split(',')[1])
                else:
                    image_bytes = base64.b64decode(image_data)
                    
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                logger.error(f"Error decoding image: {str(e)}")
                return False, f"Error decoding image: {str(e)}"
            
            # Generate content with explicit error handling
            try:
                response = model.generate_content([prompt, image])
                return True, response.text
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating content with image: {error_msg}")
                
                # Check for specific error messages
                if "model_not_found" in error_msg:
                    return False, "The vision model is not available with your API key. Please ensure you have access to gemini-pro-vision."
                elif "permission_denied" in error_msg:
                    return False, "Permission denied. Your API key may not have access to the vision model."
                else:
                    return False, f"Error analyzing image: {error_msg}"
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {str(e)}")
            return False, str(e)
            
    def process_audio(self, audio_bytes, sample_rate=16000, encoding='LINEAR16'):
        """Process audio data using local Whisper transcription and Gemini for response.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Audio sample rate
            encoding: Audio encoding format
            
        Returns:
            Dict containing transcription and response
        """
        if not self.is_available():
            return {"error": "Gemini API not initialized"}
            
        try:
            # Use local Whisper transcription
            from .modules.whisper_transcriber import WhisperTranscriber
            
            # Initialize the transcriber
            transcriber = WhisperTranscriber()
            
            # Get transcription from Whisper
            transcription_result = transcriber.transcribe_audio(audio_bytes, sample_rate)
            
            if not transcription_result or 'text' not in transcription_result:
                return {"error": "Transcription failed"}
                
            transcription = transcription_result['text']
            confidence = transcription_result.get('confidence', 0.0)
            
            # If we have a transcription, generate a response with Gemini
            if transcription and len(transcription.strip()) > 3:  # More than 3 characters
                success, gemini_response = self.generate_text(
                    f"The user said: '{transcription}'. Provide a helpful response."
                )
                
                if success:
                    return {
                        "transcription": transcription,
                        "confidence": confidence,
                        "response": gemini_response
                    }
            
            # Return just the transcription if no response generated
            return {
                "transcription": transcription,
                "confidence": confidence
            }
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {"error": str(e)}