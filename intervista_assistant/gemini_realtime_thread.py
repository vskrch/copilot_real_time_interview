"""
Gemini Realtime Thread for Intervista Assistant.
Handles communication with Google's Gemini API for real-time text processing.
"""
import os
import time
import json
import logging
import threading
import queue
import pyaudio
import wave
import numpy as np
from datetime import datetime
from typing import Dict, Any, Callable, Optional

# Import the Gemini client
try:
    from .gemini_client import GeminiClient
except ImportError:
    from gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiRealtimeThread(threading.Thread):
    """
    Thread for handling real-time communication with Gemini API.
    Uses callbacks instead of PyQt signals, but maintains similar functionality.
    """
    
    # Audio recording constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    def __init__(self, callbacks=None):
        """
        Initializes the instance with optional callbacks.
        
        Args:
            callbacks: dictionary of callback functions:
                - on_transcription(text)
                - on_response(text)
                - on_error(message)
                - on_connection_status(connected)
        """
        super().__init__()
        self.callbacks = callbacks or {}
        self.running = False
        self.connected = False
        self.transcription_buffer = ""
        self.recording = False
        self.daemon = True  # Thread will exit when main program exits
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.gemini_client.initialize(api_key)
        else:
            logger.error("No Gemini API key found in environment")
        
        # Message queue for sending text to the model
        self.message_queue = queue.Queue()
        
        # Chat history for context
        self.chat_history = []
        
        # System prompt
        self.system_prompt = """You are an expert job interview assistant, specialized in helping candidates in real-time during interviews. Provide useful, clear, and concise advice on both technical and behavioral aspects."""
        
        # Audio recording variables
        self.p = None
        self.stream = None
        self.audio_buffer = []
        self.accumulated_audio = b''
        self.is_speaking = False
        self.silence_start_time = 0
        self.silence_threshold = 500  # Adjust based on testing
        self.silence_duration = 1.5  # seconds
        self.max_audio_duration = 30  # seconds
        self.last_commit_time = 0
        self.lock = threading.Lock()
        
    def run(self):
        """Main thread execution."""
        self.running = True
        logger.info("Gemini realtime thread started")
        
        # Set connection status
        self.connected = self.gemini_client.is_available()
        self._call_callback('on_connection_status', self.connected)
        
        if not self.connected:
            error_msg = "Failed to connect to Gemini API"
            logger.error(error_msg)
            self._call_callback('on_error', error_msg)
            return
        
        # Process messages from the queue
        while self.running:
            try:
                # Check if there are messages to process
                try:
                    message = self.message_queue.get(block=False)
                    self._process_message(message)
                    self.message_queue.task_done()
                except queue.Empty:
                    # No messages in the queue, continue the loop
                    pass
                
                # Process audio if recording
                if self.recording and self.stream:
                    self._process_audio_chunk()
                
                # Sleep to prevent high CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                error_msg = f"Error in Gemini thread: {str(e)}"
                logger.error(error_msg)
                self._call_callback('on_error', error_msg)
                time.sleep(1)  # Prevent rapid error loops
        
        logger.info("Gemini realtime thread stopped")
    
    def stop(self):
        """Stops the thread."""
        logger.info("Stopping Gemini realtime thread")
        self.running = False
        self.connected = False
        self._call_callback('on_connection_status', False)
        
        # Stop recording if active
        if self.recording:
            self.stop_recording()
    
    def send_text(self, text):
        """
        Sends text to the Gemini model for processing.
        
        Args:
            text: The text message to send
            
        Returns:
            bool: True if the message was queued successfully
        """
        if not text or not self.connected:
            return False
        
        try:
            # Add to queue for processing
            self.message_queue.put(text)
            logger.info(f"Text message queued: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error queueing text message: {str(e)}")
            return False
    
    def _process_message(self, text):
        """
        Processes a text message with the Gemini API.
        
        Args:
            text: The text message to process
        """
        try:
            # Add user message to history
            self.chat_history.append({"role": "user", "content": text})
            
            # Call Gemini API
            response = self.gemini_client.process_text(
                text, 
                session_history=self._format_history_for_gemini()
            )
            
            if response:
                # Add assistant response to history
                self.chat_history.append({"role": "assistant", "content": response})
                
                # Send response through callback
                self._call_callback('on_response', response)
                logger.info(f"Received response from Gemini: {response[:50]}...")
            else:
                error_msg = "Gemini API returned empty response"
                logger.error(error_msg)
                self._call_callback('on_error', error_msg)
                
        except Exception as e:
            error_msg = f"Error processing message with Gemini: {str(e)}"
            logger.error(error_msg)
            self._call_callback('on_error', error_msg)
    
    def _format_history_for_gemini(self):
        """
        Formats chat history for Gemini API.
        
        Returns:
            List of formatted messages
        """
        formatted_history = []
        
        # Add system prompt
        if self.system_prompt:
            formatted_history.append({
                "role": "system",
                "parts": [self.system_prompt]
            })
        
        # Add conversation history (limited to last 10 messages)
        for msg in self.chat_history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        return formatted_history
    
    def _call_callback(self, callback_name, *args, **kwargs):
        """
        Calls a callback function if it exists.
        
        Args:
            callback_name: Name of the callback function
            *args, **kwargs: Arguments to pass to the callback
        """
        callback = self.callbacks.get(callback_name)
        if callback and callable(callback):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback_name}: {str(e)}")
    
    def start_recording(self):
        """Starts audio recording from the microphone."""
        with self.lock:
            if self.recording:
                return
            self.recording = True
        
        self.audio_buffer = []
        self.accumulated_audio = b''
        
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
            
            logger.info("Audio recording started")
            self._call_callback('on_transcription', "Recording started...")
            
        except Exception as e:
            logger.error(f"Error initializing PyAudio: {str(e)}")
            self._call_callback('on_error', f"Audio initialization error: {str(e)}")
            self.recording = False
    
    def stop_recording(self):
        """Stops audio recording."""
        with self.lock:
            if not self.recording:
                return
            self.recording = False
        
        # Process any remaining audio
        if self.accumulated_audio:
            self._process_accumulated_audio()
        
        # Clean up audio resources
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {str(e)}")
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {str(e)}")
        
        self.stream = None
        self.p = None
        
        logger.info("Audio recording stopped")
        self._call_callback('on_transcription', "Recording stopped[end]")
    
    def _process_audio_chunk(self):
        """Processes a chunk of audio data."""
        try:
            # Read audio data
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            
            # Convert to numpy array for analysis
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate audio level
            audio_level = np.abs(audio_data).mean()
            
            # Detect speech/silence
            is_speaking_now = audio_level > self.silence_threshold
            
            # Accumulate audio data
            self.accumulated_audio += data
            
            # Handle silence detection
            current_time = time.time()
            if self.is_speaking and not is_speaking_now:
                # Transition from speech to silence
                if self.silence_start_time == 0:
                    self.silence_start_time = current_time
                elif current_time - self.silence_start_time > self.silence_duration:
                    # Silence duration exceeded, process the accumulated audio
                    self._process_accumulated_audio()
                    self.silence_start_time = 0
            elif not self.is_speaking and is_speaking_now:
                # Transition from silence to speech
                self.silence_start_time = 0
            
            # Update speaking state
            self.is_speaking = is_speaking_now
            
            # Check if max duration exceeded
            if len(self.accumulated_audio) > self.RATE * self.max_audio_duration * 2:  # *2 for 16-bit samples
                logger.info("Max audio duration reached, processing accumulated audio")
                self._process_accumulated_audio()
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
    
    def _process_accumulated_audio(self):
        """Processes the accumulated audio buffer."""
        if not self.accumulated_audio:
            return
        
        try:
            # Use the AudioProcessor to process the audio
            from .modules.audio_processor import AudioProcessor
            
            # Create an audio processor if not already created
            if not hasattr(self, 'audio_processor'):
                self.audio_processor = AudioProcessor(self.gemini_client)
            
            # Process the audio
            result = self.audio_processor.process_audio(
                self.accumulated_audio, 
                sample_rate=self.RATE, 
                encoding="LINEAR16"
            )
            
            if result and 'transcription' in result:
                transcription = result['transcription']
                
                # Add [fine] marker to indicate this is a complete segment
                if not transcription.endswith("[fine]") and not transcription.endswith("[end]"):
                    transcription += "[fine]"
                
                # Call transcription callback
                self._call_callback('on_transcription', transcription)
                
                # If there's a response, call response callback
                if 'response' in result and result['response']:
                    self._call_callback('on_response', result['response'])
            else:
                logger.warning("No transcription result from audio processing")
                
        except Exception as e:
            logger.error(f"Error processing accumulated audio: {str(e)}")
        
        # Reset accumulated audio
        self.accumulated_audio = b''
    
    def process_image(self, image_data):
        """
        Process an image using Gemini Vision API.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            str: Analysis result or error message
        """
        try:
            # Create prompt for image analysis
            prompt = """
            Please analyze this screenshot from a technical interview or coding exercise.
            Describe what you see, identify any code, algorithms, or technical concepts.
            Provide helpful insights about the content that would be useful during a job interview.
            If you see code, explain what it does and suggest improvements or potential issues.
            """
            
            # Use Gemini client to analyze the image
            success, response = self.gemini_client.analyze_image(image_data, prompt)
            
            if success:
                # Add response to chat history
                self.chat_history.append({
                    "role": "user", 
                    "content": "[Screenshot shared for analysis]"
                })
                self.chat_history.append({
                    "role": "assistant", 
                    "content": response
                })
                
                return response
            else:
                logger.error(f"Gemini image analysis failed: {response}")
                return f"Error analyzing image with Gemini: {response}"
                
        except Exception as e:
            error_msg = f"Error processing image with Gemini: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def start_think_process(self):
        """
        Starts an advanced thinking process that generates a summary
        and a detailed solution based on the conversation.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not self.chat_history:
            return False, "No conversation to analyze."
        
        # Start the thinking process in a separate thread
        threading.Thread(
            target=self._process_thinking_async,
            daemon=True
        ).start()
        
        return True, None
    
    def _process_thinking_async(self):
        """Performs the thinking process asynchronously."""
        try:
            # First generate a summary
            summary = self._generate_summary()
            
            # Notify the user that the summary is ready
            self._call_callback('on_response', "**🧠 CONVERSATION ANALYSIS:**\n\n" + summary)
            
            # Generate a detailed solution based on the summary
            solution = self._generate_solution(summary)
            
            # Notify the user that the solution is ready
            self._call_callback('on_response', "**🚀 DETAILED SOLUTION:**\n\n" + solution)
            
            # Also send a message through the real-time thread
            if self.connected:
                context_msg = "[I have completed an in-depth analysis of our conversation, identified key issues, and generated detailed solutions. If you have specific questions, I am here to help!]"
                self.send_text(context_msg)
            
            logger.info("Thinking process completed successfully")
            
        except Exception as e:
            error_message = f"Error in the thinking process: {str(e)}"
            logger.error(error_message)
            self._call_callback('on_error', error_message)
    
    def _generate_summary(self):
        """
        Generates a summary of the conversation using Gemini API.
        
        Returns:
            str: The generated summary
        """
        try:
            # Create a summary prompt
            summary_prompt = """Analyze the conversation history and create a concise summary. 
            Focus on:
            1. Key problems or questions discussed
            2. Important context
            3. Any programming challenges mentioned
            4. Current state of the discussion
            
            Your summary should be comprehensive but brief, highlighting the most important aspects 
            that would help solve any programming or logical problems mentioned."""
            
            # Format history for Gemini
            formatted_history = self._format_history_for_gemini()
            
            # Add the summary request as a user message
            formatted_history.append({
                "role": "user",
                "parts": [summary_prompt]
            })
            
            # Process with Gemini
            response = self.gemini_client.process_text(
                summary_prompt,
                session_history=formatted_history
            )
            
            if not response:
                return "Unable to generate summary. Please try again."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _generate_solution(self, summary):
        """Generates a detailed solution based on the summary."""
        try:
            # Build the prompt
            prompt = f"""
            I'm working on a programming or logical task. Here's the context and problem:
            
            # CONTEXT
            {summary}
            
            Please analyze this situation and:
            1. Identify the core problem or challenge
            2. Develop a structured approach to solve it
            3. Provide a detailed solution with code if applicable
            4. Explain your reasoning
            """
            
            # Use Gemini to generate the solution
            if not self.gemini_client.is_available():
                # Try to initialize with environment variable
                self.gemini_client.initialize()
                
                # Check again after initialization attempt
                if not self.gemini_client.is_available():
                    raise Exception("Gemini API not configured")
            
            # Use a valid Gemini model for generating the solution
            # First try to get the gemini-pro model, which is commonly available
            model = self.gemini_client.models.get('gemini-pro')
            
            # If gemini-pro is not available, try to get any available model
            if not model:
                # Get the first available model or default to None
                available_models = list(self.gemini_client.models.keys())
                if available_models:
                    model = self.gemini_client.models.get(available_models[0])
                    logger.info(f"Using alternative model: {available_models[0]}")
                else:
                    raise Exception("No Gemini models available")
                
            if not model:
                raise Exception("Gemini chat model not available")
                
            # Generate content
            response = model.generate_content(prompt)
            if not response or not hasattr(response, 'text'):
                raise Exception("Invalid response from Gemini API")
                
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating solution: {str(e)}")
            raise