#!/usr/bin/env python3
"""
API Launcher for Intervista Assistant.
This script initializes and runs the Flask API server.
"""

import os
import sys
import logging
from pathlib import Path

# Add the main directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_api():
    """
    Initialize and run the Flask API server.
    """
    try:
        # Import the Flask app and socketio from the api module
        from intervista_assistant.api import app, socketio
        
        # Log startup message
        logger.info("Starting Intervista Assistant API server...")
        
        # Get port from environment variables with priority:
        # 1. FLASK_RUN_PORT
        # 2. PORT
        # 3. Default to 8000
        port = int(os.environ.get("FLASK_RUN_PORT") or os.environ.get("PORT") or 8000)
        
        # Log the port being used
        logger.info(f"Starting server on port {port}")
        
        # Run the socketio app
        socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Ensure all required dependencies are installed.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during API server startup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_api()