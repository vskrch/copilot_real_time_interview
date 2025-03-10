"""
Gemini API client for Intervista Assistant.
Provides a wrapper around Google's Generative AI API.
"""
import os
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import base64
import json
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.initialized = False
        self.api_key = None
        self.model = None
        
    def initialize(self, api_key=None):
        """Initialize the Gemini API with the provided key."""
        try:
            # Use provided key or get from environment
            self.api_key = api_key or os.getenv('GEMINI_API_KEY')
            
            if not self.api_key:
                logger.error("No Gemini API key provided")
                return False
                
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Set up the model
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 40,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            self.initialized = True
            logger.info("Gemini API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {str(e)}")
            self.initialized = False
            return False
    
    def is_available(self) -> bool:
        """Check if the Gemini API is available."""
        return self.initialized and self.api_key is not None
    
    def process_audio(self, audio_bytes, sample_rate=16000, encoding="LINEAR16"):
        """
        Process audio data with Gemini API.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            
        Returns:
            Dictionary with transcription and optional response
        """
        if not self.is_available():
            logger.error("Gemini API not initialized")
            return None
            
        try:
            # Convert audio bytes to base64 for API
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Create a multimodal content object with audio
            content = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": audio_b64
                            }
                        }
                    ]
                }
            ]
            
            # Send to Gemini for processing
            response = self.model.generate_content(content)
            
            # Extract the transcription and response
            if response and response.text:
                # Parse the response - Gemini typically returns both transcription and analysis
                result = {
                    "transcription": response.text,
                    "response": None  # Will be filled if there's a separate response
                }
                
                # Check if the response contains a structured format with separate transcription and response
                try:
                    parsed = json.loads(response.text)
                    if isinstance(parsed, dict) and "transcription" in parsed:
                        result["transcription"] = parsed["transcription"]
                        if "response" in parsed:
                            result["response"] = parsed["response"]
                except:
                    # Not JSON, use the full text as transcription
                    pass
                    
                logger.info(f"Gemini processed audio successfully: {len(result['transcription'])} chars")
                return result
            else:
                logger.error("Gemini returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error processing audio with Gemini: {str(e)}")
            return None
            
    def process_text(self, text, session_history=None):
        """
        Process text with Gemini API.
        
        Args:
            text: Text to process
            session_history: Optional conversation history
            
        Returns:
            Response text from Gemini
        """
        if not self.is_available():
            logger.error("Gemini API not initialized")
            return None
            
        try:
            # Create a chat session
            chat = self.model.start_chat(history=session_history or [])
            
            # Send the message
            response = chat.send_message(text)
            
            if response and response.text:
                logger.info(f"Gemini processed text successfully: {len(response.text)} chars")
                return response.text
            else:
                logger.error("Gemini returned empty text response")
                return None
                
        except Exception as e:
            logger.error(f"Error processing text with Gemini: {str(e)}")
            return None
    
    def analyze_image(self, image_data: str, prompt: str) -> Tuple[bool, str]:
        """
        Analyze an image using Gemini Pro Vision.
        
        Args:
            image_data: Base64-encoded image data
            prompt: Text prompt to guide the analysis
            
        Returns:
            Tuple of (success, response_or_error)
        """
        if not self.is_available():
            return False, "Gemini API not available"
        
        try:
            # Get the model name from environment variable or use default
            vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-pro-vision")
            logger.info(f"Using Gemini vision model: {vision_model}")
            
            # Create the model - use the configured vision model
            model = self.client.GenerativeModel(vision_model)
            
            # Process the image - handle both base64 strings and raw binary data
            if "base64," in image_data:
                # Extract the actual base64 data if it includes the data URL prefix
                image_data = image_data.split("base64,")[1]
            
            # Create the request with image and prompt
            image_parts = [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }}
            ]
            
            # Generate response
            response = model.generate_content(image_parts)
            
            # Log success
            logger.info("Image analysis completed successfully with Gemini")
            
            return True, response.text
        except Exception as e:
            error_message = f"Error analyzing image with Gemini: {str(e)}"
            logger.error(error_message)
            return False, error_message
    
    def generate_text(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, str]:
        """
        Generate text using Gemini Pro.
        
        Args:
            prompt: Text prompt
            history: Optional conversation history
            
        Returns:
            Tuple of (success, response_or_error)
        """
        if not self.is_available():
            return False, "Gemini API not available"
        
        try:
            # Create the model
            model = self.client.GenerativeModel('gemini-pro')
            
            # If we have history, use it in a chat
            if history:
                # Convert history to Gemini format
                gemini_history = []
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg["content"]
                    gemini_history.append({"role": role, "parts": [content]})
                
                # Start chat with history
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(prompt)
            else:
                # Otherwise just send the prompt directly
                response = model.generate_content(prompt)
            
            return True, response.text
        except Exception as e:
            error_message = f"Error generating text with Gemini: {str(e)}"
            logger.error(error_message)
            return False, error_message