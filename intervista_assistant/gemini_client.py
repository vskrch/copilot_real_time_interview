"""
Gemini API client for Intervista Assistant.
Provides a wrapper around Google's Generative AI API.
"""
import os
import logging
import base64
import json
from typing import List, Dict, Any, Optional, Tuple

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
        self.vision_model = None
        
    def initialize(self, api_key=None):
        """Initialize the Gemini API with the provided key."""
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI package not installed. Run: pip install google-generativeai")
            return False
            
        try:
            # Use provided key or get from environment
            self.api_key = api_key or os.getenv('GEMINI_API_KEY')
            
            if not self.api_key:
                logger.error("No Gemini API key provided")
                return False
                
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Set up the text model
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
            
            # Set up the vision model for image analysis
            self.vision_model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-vision",
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
        return GEMINI_AVAILABLE and self.initialized and self.api_key is not None
    
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
    
    def analyze_image(self, image_data, prompt=None):
        """
        Analyze an image with Gemini Vision API.
        
        Args:
            image_data: Base64 encoded image data
            prompt: Optional prompt to guide the analysis
            
        Returns:
            Tuple of (success, response_or_error)
        """
        if not self.is_available() or not self.vision_model:
            logger.error("Gemini Vision API not initialized")
            return False, "Gemini Vision API not initialized"
            
        try:
            # Create content with image
            content = []
            
            # Add prompt if provided
            if prompt:
                content.append({"text": prompt})
                
            # Add image
            content.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
            })
            
            # Generate content
            response = self.vision_model.generate_content(content)
            
            if response and response.text:
                logger.info(f"Gemini analyzed image successfully: {len(response.text)} chars")
                return True, response.text
            else:
                logger.error("Gemini returned empty image analysis")
                return False, "Empty response from Gemini"
                
        except Exception as e:
            error_msg = f"Error analyzing image with Gemini: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def generate_summary(self, conversation_history):
        """
        Generate a summary of the conversation.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Summary text
        """
        if not self.is_available():
            logger.error("Gemini API not initialized")
            return None
            
        try:
            # Format conversation for Gemini
            formatted_history = self._format_conversation_for_gemini(conversation_history)
            
            # Add summary request
            prompt = """
            Please provide a concise summary of this interview conversation. 
            Identify the main topics discussed, key questions asked, and important points made.
            """
            
            # Create content with history and prompt
            content = [{"text": prompt}]
            for msg in formatted_history:
                content.append({"text": f"{msg['role']}: {msg['content']}"})
                
            # Generate summary
            response = self.model.generate_content(content)
            
            if response and response.text:
                logger.info(f"Gemini generated summary successfully: {len(response.text)} chars")
                return response.text
            else:
                logger.error("Gemini returned empty summary")
                return None
                
        except Exception as e:
            logger.error(f"Error generating summary with Gemini: {str(e)}")
            return None
    
    def generate_solution(self, summary):
        """
        Generate a detailed solution based on a summary.
        
        Args:
            summary: Summary of the conversation
            
        Returns:
            Solution text
        """
        if not self.is_available():
            logger.error("Gemini API not initialized")
            return None
            
        try:
            # Create prompt for solution
            prompt = f"""
            Based on the following summary of an interview conversation:
            
            {summary}
            
            Please provide a detailed solution that:
            1. Addresses the main technical challenges identified
            2. Offers specific code examples or algorithms where appropriate
            3. Explains key concepts that were discussed
            4. Provides best practices and recommendations
            """
            
            # Generate solution
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                logger.info(f"Gemini generated solution successfully: {len(response.text)} chars")
                return response.text
            else:
                logger.error("Gemini returned empty solution")
                return None
                
        except Exception as e:
            logger.error(f"Error generating solution with Gemini: {str(e)}")
            return None
    
    def _format_conversation_for_gemini(self, conversation):
        """
        Format conversation history for Gemini.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Formatted conversation
        """
        formatted = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                # Handle multimodal content (text + images)
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = " ".join(text_parts)
                
            formatted.append({
                "role": role,
                "content": content
            })
            
        return formatted