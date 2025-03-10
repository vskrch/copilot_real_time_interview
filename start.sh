#!/bin/bash

# Default variables
WATCH_MODE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-watch|-nw)
      WATCH_MODE=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-watch|-nw]"
      exit 1
      ;;
  esac
done

# Function to handle errors
handle_error() {
  echo "ERROR: $1"
  echo "Check logs for more details: $SCRIPT_DIR/logs/backend.log and $SCRIPT_DIR/logs/frontend.log"
  exit 1
}

# Function to check if service is healthy
check_health() {
  local url=$1
  local service=$2
  local max_attempts=$3
  local attempt=1
  
  echo "Checking $service health..."
  while [ $attempt -le $max_attempts ]; do
    if curl -s "$url" &> /dev/null; then
      echo "$service is healthy!"
      return 0
    fi
    echo "Attempt $attempt/$max_attempts: $service not ready yet, waiting..."
    sleep 2
    attempt=$((attempt+1))
  done
  
  echo "ERROR: $service failed to start properly after $max_attempts attempts."
  return 1
}

# Determine the path of the current directory (where the script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/intervista_assistant"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Check for required Python dependencies - using a more reliable approach
echo "Checking for required Python dependencies..."
# Install dependencies directly to ensure they're available
poetry run pip install werkzeug flask flask-cors flask-socketio python-socketio python-dotenv google-generativeai

# Check that the directories exist
if [ ! -d "$BACKEND_DIR" ]; then
  handle_error "Backend directory not found: $BACKEND_DIR"
fi

if [ ! -d "$FRONTEND_DIR" ]; then
  handle_error "Frontend directory not found: $FRONTEND_DIR"
fi

# Check if npm is installed in the frontend directory
echo "Checking frontend dependencies..."
cd "$FRONTEND_DIR"
if ! command -v npm &> /dev/null; then
  handle_error "npm not found. Please install Node.js and npm."
fi

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

# Kill any previous processes on the requested ports
echo "Checking and cleaning up existing processes..."
if lsof -i:3000 -t &> /dev/null; then
  echo "Terminating processes on port 3000..."
  kill $(lsof -i:3000 -t) 2>/dev/null || true
  sleep 1
fi

if lsof -i:8000 -t &> /dev/null; then
  echo "Terminating processes on port 8000..."
  kill $(lsof -i:8000 -t) 2>/dev/null || true
  sleep 1
fi

# Also check port 5000 which might be used by Flask by default
if lsof -i:5000 -t &> /dev/null; then
  echo "Port 5000 is in use. This might conflict with Flask's default port."
  echo "Attempting to terminate processes on port 5000..."
  kill $(lsof -i:5000 -t) 2>/dev/null || true
  sleep 1
  
  # Check if port was freed
  if lsof -i:5000 -t &> /dev/null; then
    echo "Warning: Could not free port 5000. This might be used by AirPlay Receiver."
    echo "You may need to disable AirPlay Receiver in System Preferences."
  fi
fi

# After the existing port checks, add:
# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(cat .env | grep GEMINI_API_KEY)
    fi
    if [ -z "$GEMINI_API_KEY" ]; then
        handle_error "GEMINI_API_KEY not found in environment or .env file"
    fi
fi

# Create log directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Start the backend
echo "Starting the backend API..."
cd "$BACKEND_DIR" 
# Set PYTHONPATH to include the current directory
export PYTHONPATH="$SCRIPT_DIR:$BACKEND_DIR:$PYTHONPATH"

# Set environment variables for watch mode
if [ "$WATCH_MODE" = true ]; then
  export FLASK_DEBUG=1
  export FLASK_RELOADER=1
  # Explicitly set Flask port to 8000
  export FLASK_RUN_PORT=8000
  echo "Backend running in watch mode - will automatically reload on file changes"
else
  export FLASK_DEBUG=0
  export FLASK_RELOADER=0
  # Explicitly set Flask port to 8000
  export FLASK_RUN_PORT=8000
fi

# Run backend with output to console instead of background
poetry run python api_launcher.py 2>&1 | tee "$SCRIPT_DIR/logs/backend.log" &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for the backend to be ready
echo "Waiting for the backend to be ready..."
sleep 5

# Check backend health
check_health "http://localhost:8000/health" "Backend API" 10 || handle_error "Backend failed to start properly"

# Start the frontend in another terminal window on macOS
echo "Starting the frontend (Next.js)..."
cd "$FRONTEND_DIR" 

# Run frontend with output to console instead of background
npm run dev 2>&1 | tee "$SCRIPT_DIR/logs/frontend.log" &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Check frontend health
check_health "http://localhost:3000" "Frontend" 10 || handle_error "Frontend failed to start properly"

# Function to terminate all processes
cleanup() {
  echo 'Closing processes...'
  # Terminate all child processes
  pkill -P $$ || true
  # Explicitly terminate known processes
  kill $BACKEND_PID 2>/dev/null || true
  kill $FRONTEND_PID 2>/dev/null || true
  exit 0
}

# Handle shutdown with Ctrl+C and other signals
trap cleanup INT TERM

# Display mode information
if [ "$WATCH_MODE" = true ]; then
  echo "====================================================="
  echo "Application running in WATCH MODE"
  echo "Backend and frontend will reload automatically when files change"
  echo "====================================================="
fi

echo "====================================================="
echo "Logs are being written to:"
echo "  Backend: $SCRIPT_DIR/logs/backend.log"
echo "  Frontend: $SCRIPT_DIR/logs/frontend.log"
echo "====================================================="

# Keep the script running
echo "The application is running. Press Ctrl+C to terminate."
wait