#!/bin/bash

echo "Installing required dependencies for Gemini API with local Whisper transcription..."

# Install Python dependencies
pip install google-generativeai flask flask-cors flask-socketio python-socketio python-dotenv pillow numpy

# Check if Whisper is already installed
if ! pip show openai-whisper &> /dev/null; then
    echo "Installing Whisper for local transcription..."
    pip install openai-whisper
fi

echo "Dependencies installed successfully!"
echo "Make sure to set your GEMINI_API_KEY in the .env file"