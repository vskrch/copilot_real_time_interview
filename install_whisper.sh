#!/bin/bash

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Install main dependencies first
echo "Installing main dependencies..."
poetry install --no-root

# Install whisper dependencies manually
echo "Installing whisper dependencies..."
poetry run pip install torch torchaudio
poetry run pip install numba==0.58.1  # This version works with Python 3.11
poetry run pip install llvmlite==0.41.0  # This version works with Python 3.11

# Install whisper from source
echo "Installing whisper from source..."
poetry run pip install git+https://github.com/openai/whisper.git

# Install additional dependencies for Mac GPU support
echo "Installing Mac GPU support dependencies..."
poetry run pip install accelerate

echo "Installation complete!"