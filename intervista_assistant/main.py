"""Main entry point for the Intervista Assistant."""

import os
import logging
from .gemini_client import GeminiClient
from .api import app, socketio

def main():
    """Initialize and run the application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check for required environment variables
    if not os.getenv('GEMINI_API_KEY'):
        logger.error('GEMINI_API_KEY environment variable not set')
        return False
    
    try:
        # Initialize Gemini client
        gemini_client = GeminiClient()
        if not gemini_client.initialize():
            logger.error('Failed to initialize Gemini client')
            return False
        
        # Start the server
        port = int(os.getenv('PORT', '8000'))
        debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        
        socketio.run(app, host='0.0.0.0', port=port, debug=debug)
        return True
        
    except Exception as e:
        logger.error(f'Error starting application: {e}')
        return False

if __name__ == '__main__':
    main()