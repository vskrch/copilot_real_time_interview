"""
Gemini API client for Intervista Assistant.
Provides a wrapper around Google's Generative AI API.
"""
import os
import logging
import base64
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.initialized = False
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
                self.initialized = True
                logger.info("Gemini API client initialized successfully")
            except ImportError:
                logger.error("Failed to import google.generativeai. Make sure it's installed.")
            except Exception as e:
                logger.error(f"Error initializing Gemini client: {str(e)}")
        else:
            logger.warning("No Gemini API key found. Gemini features will be unavailable.")
    
    def is_available(self) -> bool:
        """Check if the Gemini API is available."""
        return self.initialized and self.api_key is not None
    
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
            # Create the model - use the latest Gemini vision model
            model = self.client.GenerativeModel('gemini-1.5-pro-vision')
            
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