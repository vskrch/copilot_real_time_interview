#!/usr/bin/env python3
"""
API Routes for Intervista Assistant.
Defines the Flask routes for the backend API.
"""
import os
import json
import uuid
import logging
from datetime import datetime
from functools import wraps
from flask import request, jsonify, Response, stream_with_context

# Configure logging
logger = logging.getLogger(__name__)

# Import the session manager
from intervista_assistant.session_manager import SessionManager

# Dictionary to store active sessions
active_sessions = {}

def require_session(f):
    """Decorator to require a valid session for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # For OPTIONS requests, just return success
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200
            
        # Get the session ID from the request
        if request.is_json:
            session_id = request.json.get('session_id')
        else:
            session_id = request.args.get('session_id')
            
        # Check if the session exists
        if not session_id or session_id not in active_sessions:
            return jsonify({
                "success": False,
                "error": "Invalid or expired session"
            }), 401
            
        # Add the session manager to the request context
        request.session_manager = active_sessions[session_id]
        
        # Update the last activity timestamp
        request.session_manager.last_activity = datetime.now()
        
        return f(*args, **kwargs)
    return decorated

def format_sse(data, event=None):
    """Formats a server-sent event."""
    msg = f"data: {data}\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    return f"{msg}\n"

def register_routes(app, socketio):
    """Registers the API routes with the Flask app."""
    
    @app.route('/api/sessions/create', methods=['POST', 'OPTIONS'])
    def create_session():
        """Creates a new session."""
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200
            
        # Generate a new session ID
        session_id = str(uuid.uuid4())
        
        # Create a new session manager
        session = SessionManager(session_id, socketio)
        
        # Start the session
        success, error = session.start_session()
        
        if success:
            # Add the session to the active sessions
            active_sessions[session_id] = session
            
            # Join the Socket.IO room for this session
            if request.sid:
                socketio.join_room(session_id, request.sid)
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "status": session.get_status()
            }), 201
        else:
            return jsonify({
                "success": False,
                "error": error or "Failed to create session"
            }), 500
    
    @app.route('/api/sessions/end', methods=['POST', 'OPTIONS'])
    @require_session
    def end_session():
        """Ends an existing session."""
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200
            
        session_id = request.json.get('session_id')
        session = request.session_manager
        
        # End the session
        success = session.end_session()
        
        if success:
            # Save the conversation and remove the session
            save_success, filename = session.save_conversation()
            
            # Safely remove the session from active_sessions
            if session_id in active_sessions:
                del active_sessions[session_id]
            
            return jsonify({
                "success": True,
                "message": "Session ended successfully",
                "conversation_saved": save_success,
                "filename": filename if save_success else None
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Unable to end the session"
            }), 500
    
    @app.route('/api/sessions/stream', methods=['GET'])
    @require_session
    def stream_session_updates():
        """Stream session updates using Server-Sent Events."""
        def generate():
            session = request.session_manager
            while True:
                # Check for transcription updates
                if session.transcription_updates:
                    update = session.transcription_updates.pop(0)
                    yield format_sse(json.dumps(update), event='transcription')
                
                # Check for response updates
                if session.response_updates:
                    update = session.response_updates.pop(0)
                    yield format_sse(json.dumps(update), event='response')
                
                # Check for error updates
                if session.error_updates:
                    update = session.error_updates.pop(0)
                    yield format_sse(json.dumps(update), event='error')
                
                # Check for connection status updates
                if session.connection_updates:
                    update = session.connection_updates.pop(0)
                    yield format_sse(json.dumps(update), event='connection')
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    
    @app.route('/api/sessions/text', methods=['POST'])
    @require_session
    def send_text_message():
        """Sends a text message to the session."""
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200
            
        message = request.json.get('message')
        if not message:
            return jsonify({
                "success": False,
                "error": "No message provided"
            }), 400
            
        session = request.session_manager
        session.handle_transcription(message)
        
        return jsonify({
            "success": True,
            "message": "Message sent successfully"
        }), 200
    
    @app.route('/api/sessions/status', methods=['GET', 'OPTIONS'])
    @require_session
    def get_session_status():
        """Gets the current status of a session."""
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200
            
        session = request.session_manager
        return jsonify({
            "success": True,
            "status": session.get_status()
        }), 200
    
    @app.route('/api/sessions/save', methods=['POST'])
    @require_session
    def save_conversation():
        """Saves the current conversation."""
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200
            
        session = request.session_manager
        success, filename = session.save_conversation()
        
        if success:
            return jsonify({
                "success": True,
                "message": "Conversation saved successfully",
                "filename": filename
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save conversation"
            }), 500