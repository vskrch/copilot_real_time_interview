#!/usr/bin/env python3
"""
Image Processing Module for Intervista Assistant.
Handles screenshot analysis using Gemini API.
"""
import os
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from ..gemini_client import GeminiClient

# Configure logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing and analysis using Gemini API."""
    
    def __init__(self, gemini_client: GeminiClient = None, handle_response_callback=None):
        """Initialize the image processor.
        
        Args:
            gemini_client: Initialized Gemini client
            handle_response_callback: Callback for handling responses
        """
        # If no client provided, create one
        if gemini_client is None:
            from ..gemini_client import GeminiClient
            import os
            
            # Check for API key in environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY environment variable not set")
                # Try to look for a .env file and load it
                try:
                    from dotenv import load_dotenv
                    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
                    if os.path.exists(env_path):
                        load_dotenv(env_path)
                        api_key = os.getenv("GEMINI_API_KEY")
                        if api_key:
                            logger.info("Loaded GEMINI_API_KEY from .env file")
                except ImportError:
                    logger.warning("python-dotenv not installed, cannot load from .env file")
            
            self.gemini_client = GeminiClient()
            # Try to initialize with environment variable
            success = self.gemini_client.initialize(api_key)
            if not success:
                logger.error("Failed to initialize Gemini client. Please ensure GEMINI_API_KEY is set correctly.")
        else:
            self.gemini_client = gemini_client
            
        self.handle_response_callback = handle_response_callback
        self.handle_response = handle_response_callback
        
    def process_screenshot(self, image_data: str, chat_history: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Process a screenshot using Gemini API.
        
        Args:
            image_data: Base64 encoded image data
            chat_history: Conversation history
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Prepare messages with history and image
            messages = self._prepare_messages_with_history(chat_history, image_data)
            
            # Perform image analysis asynchronously
            threading.Thread(
                target=self._analyze_image_async,
                args=(messages, image_data, chat_history),
                daemon=True
            ).start()
            
            return True, None
            
        except Exception as e:
            error_message = f"Error processing screenshot: {str(e)}"
            logger.error(error_message)
            return False, error_message
    
    def _analyze_image_async(self, messages: List[Dict[str, Any]], image_data: str, chat_history: List[Dict[str, Any]]):
        """Performs image analysis asynchronously using Gemini API."""
        try:
            # Send a processing notification
            if self.handle_response:
                self.handle_response("Analyzing the screenshot...", final=False)
            
            # Ensure Gemini client is initialized
            if not hasattr(self.gemini_client, 'is_available') or not self.gemini_client.is_available():
                # Try to initialize with environment variable again
                import os
                from dotenv import load_dotenv
                
                # Try to load from .env file first
                try:
                    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
                    if os.path.exists(env_path):
                        load_dotenv(env_path)
                        logger.info(f"Loaded environment from {env_path}")
                except Exception as e:
                    logger.warning(f"Could not load .env file: {str(e)}")
                
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    error_msg = "Gemini API key not found. Please set GEMINI_API_KEY environment variable or add it to .env file."
                    logger.error(error_msg)
                    if self.handle_response:
                        self.handle_response(f"Error: {error_msg}", final=True)
                    return
                    
                success = self.gemini_client.initialize(api_key)
                
                # Check again after initialization attempt
                if not hasattr(self.gemini_client, 'is_available') or not self.gemini_client.is_available():
                    error_msg = "Gemini API initialization failed. Please check your API key."
                    logger.error(error_msg)
                    if self.handle_response:
                        self.handle_response(f"Error: {error_msg}", final=True)
                    return
            
            # Check if vision model is available
            if not hasattr(self.gemini_client, 'models') or ('gemini-pro-vision' not in self.gemini_client.models and 'vision' not in self.gemini_client.models):
                # Try to initialize the model directly
                try:
                    import google.generativeai as genai
                    self.gemini_client.models['gemini-pro-vision'] = genai.GenerativeModel('gemini-pro-vision')
                    self.gemini_client.models['vision'] = self.gemini_client.models['gemini-pro-vision']
                    logger.info("Late initialization of gemini-pro-vision model succeeded")
                except Exception as e:
                    error_msg = f"The required 'gemini-pro-vision' model is not available. Error: {str(e)}"
                    logger.error(error_msg)
                    if self.handle_response:
                        self.handle_response(f"Error: {error_msg}", final=True)
                    return
            
            # Extract text content from messages for context
            context = ""
            for msg in messages:
                if msg["role"] != "system" and isinstance(msg["content"], str):
                    context += f"{msg['role']}: {msg['content']}\n"
            
            # Create prompt for image analysis
            prompt = f"""
            Please analyze this screenshot from a technical interview or coding exercise.
            Describe what you see, identify any code, algorithms, or technical concepts.
            Provide helpful insights about the content.
            
            Previous context:
            {context}
            """
            
            # Use the gemini_client to analyze the image
            success, assistant_response = self.gemini_client.analyze_image(image_data, prompt)
            
            if not success:
                logger.warning(f"Gemini API failed: {assistant_response}")
                if self.handle_response:
                    self.handle_response(f"Error analyzing image: {assistant_response}", final=True)
                return
            
            # Add to chat history
            chat_history.append({
                "role": "assistant", 
                "content": assistant_response
            })
            
            # Log the response
            logger.info(f"Image analysis response generated: {assistant_response[:100]}...")
            
            # Send the response as an update - ensure this is marked as final
            if self.handle_response:
                self.handle_response(assistant_response, final=True)
            
        except Exception as e:
            error_message = f"Error during image analysis: {str(e)}"
            logger.error(error_message)
            if self.handle_response:
                self.handle_response(f"Error analyzing image: {str(e)}. Please ensure GEMINI_API_KEY is set in your environment.", final=True)
    
    def _prepare_messages_with_history(self, chat_history: List[Dict[str, Any]], base64_image: str = None) -> List[Dict[str, Any]]:
        """Prepares messages for the API including chat history."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert job interview assistant, specialized in helping candidates in real-time during interviews. Provide useful, clear, and concise advice on both technical and behavioral aspects."
            }
        ]
        
        # Add chat history (last 10 messages)
        for msg in chat_history[-10:]:
            messages.append(msg.copy())
        
        # Add the image if present
        if base64_image:
            content = [
                {
                    "type": "text",
                    "text": "Analyze this interview screenshot. Describe what you see, what questions/challenges are present, and provide advice on how to respond or solve the problem."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        
        return messages
    
    def process_image(self, image_data, context=None):
        """
        Process an image using Gemini Vision API.
        
        Args:
            image_data: Base64 encoded image data
            context: Optional context from the conversation
            
        Returns:
            dict: Processing result with transcription and response
        """
        try:
            # Ensure Gemini client is initialized
            if not hasattr(self.gemini_client, 'is_available') or not self.gemini_client.is_available():
                # Try to initialize with environment variable
                import os
                from dotenv import load_dotenv
                
                # Try to load from .env file first
                try:
                    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
                    if os.path.exists(env_path):
                        load_dotenv(env_path)
                        logger.info(f"Loaded environment from {env_path}")
                except Exception as e:
                    logger.warning(f"Could not load .env file: {str(e)}")
                
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    error_msg = "Gemini API key not found. Please set GEMINI_API_KEY environment variable or add it to .env file."
                    logger.error(error_msg)
                    return {"error": error_msg}
                    
                success = self.gemini_client.initialize(api_key)
                
                # Check again after initialization attempt
                if not hasattr(self.gemini_client, 'is_available') or not self.gemini_client.is_available():
                    error_msg = "Gemini API initialization failed. Please check your API key."
                    logger.error(error_msg)
                    return {"error": error_msg}
            
            # Create prompt for image analysis
            prompt = """
            Please analyze this screenshot from a technical interview or coding exercise.
            Describe what you see, identify any code, algorithms, or technical concepts.
            Provide helpful insights about the content that would be useful during a job interview.
            If you see code, explain what it does and suggest improvements or potential issues.
            """
            
            if context:
                prompt += f"\n\nConversation context: {context}"
            
            # Use Gemini client to analyze the image
            success, response = self.gemini_client.analyze_image(image_data, prompt)
            
            if not success:
                error_msg = f"Gemini image analysis failed: {response}"
                logger.error(error_msg)
                return {"error": error_msg}
                
            # Send the analysis through the callback
            if self.handle_response:
                self.handle_response(response, final=True)
                
            return {
                "transcription": "[Image analysis]",
                "response": response
            }
                
        except Exception as e:
            error_msg = f"Error processing image with Gemini: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}