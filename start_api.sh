#!/bin/bash

# Start API Server for Intervista Assistant
echo "Starting Intervista Assistant API Server..."

# Change to the project directory
cd "$(dirname "$0")"

# Run the API launcher
python3 -m intervista_assistant.api_launcher