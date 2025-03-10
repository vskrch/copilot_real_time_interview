"""Configuration settings for the Intervista Assistant."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Server Configuration
HOST = '0.0.0.0'
PORT = int(os.getenv('PORT', '8000'))
DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096
ENCODING = 'LINEAR16'

# Transcription Configuration
USE_LOCAL_TRANSCRIPTION = os.getenv('USE_LOCAL_TRANSCRIPTION', 'true').lower() == 'true'
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'base')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'auto')
WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')