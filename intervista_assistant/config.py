#!/usr/bin/env python3
"""
Configuration module for Intervista Assistant.
Centralizes configuration settings and environment variables.
"""
import os
import json
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Intervista Assistant."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # API Preferences
    DEFAULT_API = os.getenv("DEFAULT_API", "openai").lower()
    USE_OPENAI_FOR_TEXT = os.getenv("USE_OPENAI_FOR_TEXT", "true").lower() == "true"
    USE_GEMINI_FOR_TEXT = os.getenv("USE_GEMINI_FOR_TEXT", "false").lower() == "true"
    USE_OPENAI_FOR_IMAGES = os.getenv("USE_OPENAI_FOR_IMAGES", "false").lower() == "true"
    
    # Session Settings
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    
    # Audio Settings
    AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))
    AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_RATE = int(os.getenv("AUDIO_RATE", "16000"))
    
    # Model Settings
    OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-realtime-preview")
    OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-4o")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    # System Prompt
    @staticmethod
    def get_system_prompt():
        """Returns the system prompt from the JSON file."""
        try:
            # Path to the system prompt file
            system_prompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "system_prompt.json"
            )
            
            # Load the system prompt from the JSON file
            with open(system_prompt_path, 'r') as f:
                system_prompt_data = json.load(f)
                
            return system_prompt_data.get("system_prompt", "")
        except Exception as e:
            logger.error(f"Error loading system prompt: {str(e)}")
            return "You are an expert job interview assistant, specialized in helping candidates in real-time during interviews. Provide useful, clear, and concise advice on both technical and behavioral aspects."
    
    @staticmethod
    def validate():
        """Validates the configuration settings."""
        if not Config.OPENAI_API_KEY:
            logger.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")
            return False
        
        return True