# #!/usr/bin/env python3
# """
# API Routes Module for Intervista Assistant.
# Defines the Flask routes for the backend API.
# """
# import os
# import json
# import uuid
# import logging
# from datetime import datetime
# from functools import wraps
# from flask import request, jsonify, Response, stream_with_context

# # Configure logging
# logger = logging.getLogger(__name__)

# class ApiRoutes:
#     """Handles API routes for the Intervista Assistant."""
    
#     def __init__(self, app, socketio, active_sessions):
#         """Initialize the API routes.
        
#         Args:
#             app: Flask application
#             socketio: Socket.IO instance
#             active_sessions: Dictionary of active sessions
#         """
#         self.app = app
#         self.socketio = socketio
#         self.active_sessions = active_sessions
        
#         # Register routes
#         self._register_routes()
    
#     def _register_routes(self):
#         """Register all API routes."""
#         # OPTIONS handler for CORS
#         self.app.route('/api/<path:path>', methods=['OPTIONS'])(
#             self.handle_options
#         )
        
#         # Session management routes
#         self.app.route('/api/sessions', methods=['POST', 'OPTIONS'])(
#             self.create_session
#         )
        
#         self.app.route('/api/sessions/start', methods=['POST', 'OPTIONS'])(
#             self.require_session(self.start_session)
#         )
        
#         self.app.route('/api/sessions/end', methods=['POST', 'OPTIONS'])(
#             self.require_session(self.end_session)
#         )
        
#         self.app.route('/api/sessions/stream', methods=['GET'])(
#             self.stream_session_updates
#         )
        
#         self.app.route('/api/sessions/text', methods=['POST'])(
#             self.require_session(self.send_text_message)
#         )
        
#         self.app.route('/api/sessions/screenshot', methods=['POST'])(
#             self.require_session(self.process_screenshot)
#         )
        
#         self.app.route('/api/sessions/analyze-screenshot', methods=['POST'])(
#             self.require_session(self.analyze_screenshot)
#         )
        
#         self.app.route('/api/sessions/think', methods=['POST'])(
#             self.require_session(self.start_think_process)
#         )
        
#         self.app.route('/api/sessions/status', methods=['GET', 'OPTIONS'])(
#             self.get_session_status
#         )
        
#         self.app.route('/api/sessions/save', methods=['POST'])(
#             self.require_session(self.save_conversation)
#         )
    
#     def require_session(self, f):
#         """Decorator to require a valid session for API endpoints."""
#         @wraps(f)
#         def decorated_function(*args, **kwargs):
#             # Skip session check for OPTIONS requests
#             if request.method == 'OPTIONS':
#                 return jsonify({"success": True}), 200
                
#             session_id = request.json.get('session_id')
#             if not session_id or session_id not in self.active_sessions:
#                 return jsonify({"success": False, "error": "Session not found"}), 404
                
#             # Store the session in the request context to avoid race conditions
#             request.session_manager = self.active_sessions.get(session_id)
#             if not request.session_manager:
#                 return jsonify({"success": False, "error": "Session disappeared during request processing"}), 404
                
#             return f(*args, **kwargs)
#         return decorated_function
    
#     def handle_options(self, path):
#         """Handle OPTIONS requests for all API endpoints."""
#         response = jsonify({"success": True})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#         response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
#         return response, 200
    
#     # Add this method to the ApiRoutes class if it doesn't exist
    
#     def create_session(self):
#         """Create a new session or return an existing one."""
#         if request.method == 'OPTIONS':
#             return self.handle_options('sessions')
            
#         try:
#             # Generate a new session ID if not provided
#             session_id = request.json.get('session_id')
#             if not session_id:
#                 session_id = str(uuid.uuid4())
                
#             # Check if session already exists
#             if session_id in self.active_sessions:
#                 logger.info(f"Returning existing session: {session_id}")
#                 return jsonify({
#                     "success": True,
#                     "session_id": session_id,
#                     "message": "Using existing session"
#                 }), 200
                
#             # Create a new session manager
#             from ..gemini_client import GeminiClient
#             gemini_client = GeminiClient()
#             gemini_client.initialize()
            
#             # Create session manager
#             session_manager = SessionManager(session_id, gemini_client)
#             self.active_sessions[session_id] = session_manager
            
#             logger.info(f"Created new session: {session_id}")
#             return jsonify({
#                 "success": True,
#                 "session_id": session_id,
#                 "message": "New session created"
#             }), 201
            
#         except Exception as e:
#             error_message = f"Error creating session: {str(e)}"
#             logger.error(error_message)
#             return jsonify({
#                 "success": False,
#                 "error": error_message
#             }), 500
    
#     def start_session(self):
#         """Starts a session with the specified session_id."""
#         if request.method == 'OPTIONS':
#             return jsonify({"success": True}), 200
            
#         session_id = request.json.get('session_id')
#         session = request.session_manager
#         success, error = session.start_session()
        
#         if success:
#             # Add a small delay to give the connection time to establish
#             import time
#             max_wait = 3
#             wait_interval = 0.1
#             waited = 0
            
#             while waited < max_wait:
#                 if session.text_thread and session.text_thread.connected:
#                     logger.info(f"Connection established after {waited:.1f} seconds")
#                     break
#                 time.sleep(wait_interval)
#                 waited += wait_interval
            
#             return jsonify({
#                 "success": True,
#                 "message": "Session started successfully",
#                 "connected": session.text_thread and session.text_thread.connected
#             }), 200
#         else:
#             return jsonify({
#                 "success": False,
#                 "error": error
#             }), 500
    
#     def end_session(self):
#         """Ends an existing session."""
#         if request.method == 'OPTIONS':
#             return jsonify({"success": True}), 200
            
#         session_id = request.json.get('session_id')
#         session = request.session_manager
        
#         # End the session
#         success = session.end_session()
        
#         if success:
#             # Save the conversation and remove the session
#             save_success, filename = session.save_conversation()
            
#             # Safely remove the session from active_sessions
#             if session_id in self.active_sessions:
#                 del self.active_sessions[session_id]
            
#             return jsonify({
#                 "success": True,
#                 "message": "Session ended successfully",
#                 "conversation_saved": save_success,
#                 "filename": filename if save_success else None
#             }), 200
#         else:
#             return jsonify({
#                 "success": False,
#                 "error": "Unable to end the session"
#             }), 500
    
#     def stream_session_updates(self):
#         """SSE stream for session updates."""
#         session_id = request.args.get('session_id')
        
#         if not session_id or session_id not in self.active_sessions:
#             return jsonify({"success": False, "error": "Session not found"}), 404
        
#         logger.info(f"Starting SSE stream for session {session_id}")
        
#         # Set necessary CORS headers
#         headers = {
#             'Cache-Control': 'no-cache',
#             'X-Accel-Buffering': 'no',
#             'Connection': 'keep-alive',
#             'Content-Type': 'text/event-stream',
#             'Access-Control-Allow-Origin': '*',
#             'Access-Control-Allow-Headers': 'Content-Type',
#             'Access-Control-Allow-Methods': 'GET'
#         }
        
#         return Response(
#             stream_with_context(self._session_sse_generator(session_id)),
#             mimetype='text/event-stream',
#             headers=headers
#         )
    
#     def _session_sse_generator(self, session_id):
#         """Generator for SSE events from a session."""
#         session = self.active_sessions.get(session_id)
        
#         if not session:
#             logger.error(f"SSE generator: Session {session_id} not found")
#             yield self._format_sse(json.dumps({
#                 "type": "error",
#                 "message": "Session not found",
#                 "session_id": session_id
#             }))
#             return
        
#         # Send initial connection status
#         session.handle_connection_status(True)
        
#         # Keep track of the last update processed
#         last_update_index = 0
        
#         try:
#             # Initial heartbeat
#             yield self._format_sse(json.dumps({
#                 "type": "heartbeat",
#                 "timestamp": datetime.now().isoformat(),
#                 "session_id": session_id
#             }))
            
#             # Main loop for sending updates
#             while session_id in self.active_sessions:
#                 # Get all updates of any type
#                 all_updates = session.get_updates()
#                 new_updates = all_updates[last_update_index:]
                
#                 # Send any new updates
#                 for update in new_updates:
#                     # Convert the update to an SSE event
#                     event_data = {}
                    
#                     if "text" in update:
#                         if "transcription" in update["type"]:
#                             event_data = {
#                                 "type": "transcription",
#                                 "text": update["text"],
#                                 "session_id": session_id
#                             }
#                         elif "response" in update["type"]:
#                             final = update.get("final", True)
#                             event_data = {
#                                 "type": "response",
#                                 "text": update["text"],
#                                 "session_id": session_id,
#                                 "final": final
#                             }
#                     elif "message" in update:
#                         event_data = {
#                             "type": "error",
#                             "message": update["message"],
#                             "session_id": session_id
#                         }
#                     elif "connected" in update:
#                         event_data = {
#                             "type": "connection",
#                             "connected": update["connected"],
#                             "session_id": session_id
#                         }
                    
#                     if event_data:
#                         yield self._format_sse(json.dumps(event_data))
                
#                 # Update the last update index
#                 last_update_index = len(all_updates)
                
#                 # Send a heartbeat every 5 seconds to keep the connection alive
#                 yield self._format_sse(json.dumps({
#                     "type": "heartbeat",
#                     "timestamp": datetime.now().isoformat(),
#                     "session_id": session_id
#                 }))
                
#                 # Sleep to prevent CPU overuse
#                 import time
#                 time.sleep(1)
                
#         except GeneratorExit:
#             logger.info(f"SSE connection closed for session {session_id}")
#             # Notify the session that the connection is closed
#             if session_id in self.active_sessions:
#                 self.active_sessions[session_id].handle_connection_status(False)
#         except Exception as e:
#             logger.error(f"Error in SSE generator for session {session_id}: {str(e)}")
#             yield self._format_sse(json.dumps({
#                 "type": "error",
#                 "message": f"Server error: {str(e)}",
#                 "session_id": session_id
#             }))
#         finally:
#             logger.info(f"SSE generator exiting for session {session_id}")
    
#     def _format_sse(self, data):
#         """Format data as SSE event."""
#         return f"data: {data}\n\n"