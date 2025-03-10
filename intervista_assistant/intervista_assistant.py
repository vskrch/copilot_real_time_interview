#!/usr/bin/env python3
import sys
import os
import time
import json
import logging
from datetime import datetime
import base64

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QMessageBox, QFileDialog)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from openai import OpenAI
from dotenv import load_dotenv

from .realtime_text_thread import RealtimeTextThread
from .utils import ScreenshotManager
from .ui import IntervistaAssistantUI
from .whisper_transcriber import WhisperTranscriber  # Add this import

# Logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log')
logger = logging.getLogger(__name__)

class IntervistaAssistant(QMainWindow):
    """Main application for the interview assistant."""
    
    def __init__(self):
        super().__init__()
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            QMessageBox.critical(self, "API Key Error", 
                                 "OpenAI API Key not found. Set the environment variable OPENAI_API_KEY.")
            sys.exit(1)
        self.client = OpenAI(api_key=api_key)
        self.recording = False
        self.text_thread = None
        self.chat_history = []
        self.shutdown_in_progress = False
        self.screenshot_manager = ScreenshotManager()
        
        # Enhanced configuration for local transcription
        self.use_local_transcription = os.getenv("USE_LOCAL_TRANSCRIPTION", "false").lower() == "true"
        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        self.whisper_device = os.getenv("WHISPER_DEVICE", "auto")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        
        # Log transcription configuration
        if self.use_local_transcription:
            logger.info(f"Using local transcription with Whisper.cpp: model={self.whisper_model_size}, device={self.whisper_device}, compute_type={self.whisper_compute_type}")
        else:
            logger.info("Using OpenAI API for transcription")
        
        # Initialize the UI using the IntervistaAssistantUI class
        self.ui = IntervistaAssistantUI(self)
        self.central_widget = self.ui.central_widget
        self.transcription_text = self.central_widget.transcription_text
        self.response_text = self.central_widget.response_text
        self.record_button = self.central_widget.record_button
        self.screenshot_button = self.central_widget.screenshot_button
        self.share_button = self.central_widget.share_button
        self.think_button = self.central_widget.think_button
        self.save_button = self.central_widget.save_button
        
        # Set up the UI connections
        self.setup_connections()
        
        # Set window properties
        self.setWindowTitle("Intervista Assistant")
        self.resize(1200, 800)
        
        # Show the window
        self.show()

    def toggle_recording(self):
        """Toggle the connection to the model and immediately start recording."""
        if not self.recording:
            self.recording = True
            self.record_button.setText("End Session")
            self.record_button.setStyleSheet("background-color: #ff5555;")
            
            # Temporarily disable text input controls
            self.central_widget.text_input_field.setEnabled(False)
            self.central_widget.send_button.setEnabled(False)
            
            # Create the text thread with local transcription configuration
            self.text_thread = RealtimeTextThread(
                use_local_transcription=self.use_local_transcription,
                whisper_config={
                    "model_size": self.whisper_model_size,
                    "device": self.whisper_device,
                    "compute_type": self.whisper_compute_type
                } if self.use_local_transcription else None
            )
            
            # Connect signals
            self.text_thread.transcription_signal.connect(self.update_transcription)
            self.text_thread.response_signal.connect(self.update_response)
            self.text_thread.error_signal.connect(self.show_error)
            self.text_thread.connection_status_signal.connect(self.update_connection_status)
            self.text_thread.finished.connect(self.recording_finished)
            self.text_thread.start()
            
            # Show appropriate message based on transcription mode
            if self.use_local_transcription:
                self.transcription_text.setText("Initializing local transcription with Whisper.cpp...")
                self.transcription_text.append(f"Model: {self.whisper_model_size}, Device: {self.whisper_device}")
            else:
                self.transcription_text.setText("Connecting to OpenAI API...")
            
            # Automatically start recording immediately after session initialization
            while not self.text_thread.connected:
                time.sleep(0.1)
                QApplication.processEvents()  # Keep UI responsive
            self.text_thread.start_recording()
        else:
            if self.shutdown_in_progress:
                return
                
            self.shutdown_in_progress = True
            self.record_button.setText("Terminating...")
            self.record_button.setEnabled(False)
            
            # Disable text input controls
            self.central_widget.text_input_field.setEnabled(False)
            self.central_widget.send_button.setEnabled(False)
            
            if hasattr(self.text_thread, 'recording') and self.text_thread.recording:
                try:
                    self.text_thread.stop_recording()
                except Exception as e:
                    logger.error("Error during stop_recording: " + str(e))
            
            try:
                if self.text_thread:
                    self.text_thread.stop()
                    self.text_thread.wait(2000)
            except Exception as e:
                logger.error("Error during session termination: " + str(e))
                self.recording_finished()
    
    def recording_finished(self):
        """Called when the thread has finished."""
        self.recording = False
        self.shutdown_in_progress = False
        self.record_button.setText("Start Session")
        self.record_button.setStyleSheet("")
        self.record_button.setEnabled(True)
        self.transcription_text.append("\n[Session ended]")
    
    def update_connection_status(self, connected):
        """Update the interface based on the connection status."""
        if connected:
            self.record_button.setStyleSheet("background-color: #55aa55;")
            # Enable text input field and send button
            self.central_widget.text_input_field.setEnabled(True)
            self.central_widget.send_button.setEnabled(True)
        else:
            if self.recording:
                self.record_button.setStyleSheet("background-color: #ff5555;")
            # Disable text input field and send button
            self.central_widget.text_input_field.setEnabled(False)
            self.central_widget.send_button.setEnabled(False)
    
    def update_transcription(self, text):
        """Update the transcription field."""
        if text == "Recording in progress...":
            self.transcription_text.setText(text)
            return
        
        if text.startswith('\n[Audio processed at'):
            formatted_timestamp = f"\n--- {text.strip()} ---\n"
            current_text = self.transcription_text.toPlainText()
            if current_text == "Recording in progress...":
                self.transcription_text.setText(formatted_timestamp)
            else:
                self.transcription_text.append(formatted_timestamp)
        else:
            self.transcription_text.append(text)
        
        self.transcription_text.verticalScrollBar().setValue(
            self.transcription_text.verticalScrollBar().maximum()
        )
        
        if text != "Recording in progress...":
            if not self.chat_history or self.chat_history[-1]["role"] != "user" or self.chat_history[-1]["content"] != text:
                self.chat_history.append({"role": "user", "content": text})
    
    def update_response(self, text):
        """Update the response field with markdown formatting."""
        if not text:
            return
        current_time = datetime.now().strftime("%H:%M:%S")
        html_style = """
        <style>
            body, p, div, span, li, td, th {
                font-family: 'Arial', sans-serif !important;
                font-size: 14px !important;
                line-height: 1.6 !important;
                color: #333333 !important;
            }
            h1 { font-size: 20px !important; margin: 20px 0 10px 0 !important; font-weight: bold !important; }
            h2 { font-size: 18px !important; margin: 18px 0 9px 0 !important; font-weight: bold !important; }
            h3 { font-size: 16px !important; margin: 16px 0 8px 0 !important; font-weight: bold !important; }
            h4 { font-size: 15px !important; margin: 14px 0 7px 0 !important; font-weight: bold !important; }
            pre {
                background-color: #f5f5f5 !important;
                border: 1px solid #cccccc !important;
                border-radius: 4px !important;
                padding: 10px !important;
                margin: 10px 0 !important;
                overflow-x: auto !important;
                font-family: Consolas, 'Courier New', monospace !important;
                font-size: 14px !important;
                line-height: 1.45 !important;
                tab-size: 4 !important;
                white-space: pre !important;
            }
            code {
                font-family: Consolas, 'Courier New', monospace !important;
                font-size: 14px !important;
                background-color: #f5f5f5 !important;
                padding: 2px 4px !important;
                border-radius: 3px !important;
                border: 1px solid #cccccc !important;
                color: #333333 !important;
                white-space: pre !important;
            }
            ul, ol { margin: 10px 0 10px 20px !important; padding-left: 20px !important; }
            li { margin-bottom: 6px !important; }
            p { margin: 10px 0 !important; }
            strong { font-weight: bold !important; }
            em { font-style: italic !important; }
            table {
                border-collapse: collapse !important;
                width: 100% !important;
                margin: 15px 0 !important;
                font-size: 14px !important;
            }
            th, td {
                border: 1px solid #dddddd !important;
                padding: 8px !important;
                text-align: left !important;
            }
            th { background-color: #f2f2f2 !important; font-weight: bold !important; }
            .response-header {
                color: #666666 !important;
                font-size: 13px !important;
                margin: 15px 0 10px 0 !important;
                border-bottom: 1px solid #eeeeee !important;
                padding-bottom: 5px !important;
                font-weight: bold !important;
            }
        </style>
        """
        header = f'<div class="response-header">--- Response at {current_time} ---</div>'
        
        def process_code_blocks(text):
            import re
            def replace_code_block(match):
                language = match.group(1).strip() if match.group(1) else ""
                code = match.group(2)
                code_html = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return f'<pre><code class="language-{language}">{code_html}</code></pre>'
            def replace_inline_code(match):
                code = match.group(1)
                code_html = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return f'<code>{code_html}</code>'
            processed = re.sub(r'```([^\n]*)\n([\s\S]*?)\n```', replace_code_block, text)
            processed = re.sub(r'`([^`\n]+?)`', replace_inline_code, processed)
            return processed
        
        def custom_markdown(text):
            import re
            text = process_code_blocks(text)
            text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
            text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
            text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
            text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
            text = re.sub(r'^(\s*)-\s+(.+)$', r'\1<li>\2</li>', text, flags=re.MULTILINE)
            pattern_ul = r'(<li>.*?</li>)(\n<li>.*?</li>)*'
            text = re.sub(pattern_ul, r'<ul>\g<0></ul>', text)
            text = re.sub(r'^(\s*)\d+\.\s+(.+)$', r'\1<li>\2</li>', text, flags=re.MULTILINE)
            pattern_ol = r'(<li>.*?</li>)(\n<li>.*?</li>)*'
            text = re.sub(pattern_ol, r'<ol>\g<0></ol>', text)
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
            text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
            text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
            lines = text.split('\n')
            in_html_block = False
            for i, line in enumerate(lines):
                if not line.strip() or line.strip().startswith('<'):
                    continue
                if '<pre>' in line or '<ul>' in line or '<ol>' in line or '<h' in line:
                    in_html_block = True
                elif '</pre>' in line or '</ul>' in line or '</ol>' in line or '</h' in line:
                    in_html_block = False
                    continue
                if not in_html_block:
                    lines[i] = f'<p>{line}</p>'
            text = '\n'.join(lines)
            text = re.sub(r'<p>\s*</p>', '', text)
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
            return text
        
        html_content = custom_markdown(text)
        formatted_html = f"{html_style}{header}{html_content}"
        self.response_text.setAcceptRichText(True)
        current_html = self.response_text.toHtml()
        if not self.response_text.toPlainText():
            self.response_text.setHtml(formatted_html)
        else:
            closing_index = current_html.rfind("</body>")
            if closing_index > 0:
                new_html = current_html[:closing_index] + f"{header}{html_content}" + current_html[closing_index:]
                self.response_text.setHtml(new_html)
            else:
                self.response_text.setHtml(current_html + formatted_html)
        self.response_text.verticalScrollBar().setValue(
            self.response_text.verticalScrollBar().maximum()
        )
        if (not self.chat_history or self.chat_history[-1]["role"] != "assistant"):
            self.chat_history.append({"role": "assistant", "content": text})
        elif self.chat_history and self.chat_history[-1]["role"] == "assistant":
            previous_content = self.chat_history[-1]["content"]
            self.chat_history[-1]["content"] = f"{previous_content}\n--- Response at {current_time} ---\n{text}"
    
    def take_and_send_screenshot(self):
        """Capture screenshot and send it to the OpenAI model."""
        try:
            # Check if realtime thread is active
            if not self.recording or not self.text_thread or not self.text_thread.connected:
                QMessageBox.warning(self, "Not Connected", 
                                    "You need to start a session first before analyzing images.")
                return
            
            # Get the selected monitor index from the dropdown menu
            # First check if the attribute exists
            selected_monitor = 0  # Default to primary monitor
            if hasattr(self, 'screen_selector_combo') and self.screen_selector_combo is not None:
                selected_monitor = self.screen_selector_combo.currentData()
            logger.info(f"Capturing screenshot of monitor: {selected_monitor}")
                
            self.showMinimized()
            time.sleep(0.5)
            screenshot_path = self.screenshot_manager.take_screenshot(monitor_index=selected_monitor)
            self.showNormal()
            
            # Update UI to show we're sending the image
            self.transcription_text.append("\n[Screenshot sent for analysis]\n")
            self.response_text.append("<p><i>Image analysis in progress... The app is still responsive.</i></p>")
            QApplication.processEvents()  # Update UI
            
            # Prepare messages for gpt-4o including chat history
            messages = self._prepare_messages_with_history(base64_image=None)
            
            # Update the last message to include the image (will be replaced in the worker)
            messages[-1]["content"][1]["image_url"]["url"] = "placeholder"  # Will be replaced in the worker
            
            # Create a thread and worker for asynchronous analysis
            self.analysis_thread = QThread()
            self.analysis_worker = ImageAnalysisWorker(self.client, messages, screenshot_path)
            
            # Move the worker to the thread
            self.analysis_worker.moveToThread(self.analysis_thread)
            
            # Connect signals and slots
            self.analysis_thread.started.connect(self.analysis_worker.analyze)
            self.analysis_worker.analysisComplete.connect(self.on_analysis_complete)
            self.analysis_worker.error.connect(self.show_error)
            self.analysis_worker.analysisComplete.connect(self.analysis_thread.quit)
            self.analysis_worker.error.connect(self.analysis_thread.quit)
            self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
            self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
            
            # Start the thread
            self.analysis_thread.start()
            
            logger.info(f"Screenshot capture initiated: {screenshot_path}")
            
        except Exception as e:
            error_msg = f"Error during screenshot capture: {str(e)}"
            self.show_error(error_msg)
            logger.error(error_msg)
    
    def on_analysis_complete(self, assistant_response):
        """Callback invoked when image analysis is completed."""
        try:
            # Update the UI with the response
            self.update_response(assistant_response)
            
            # Send the analysis as text to the realtime thread to maintain conversation flow
            if self.text_thread and self.text_thread.connected:
                # Send a reduced version of the response to the realtime thread
                context_msg = f"[I've analyzed the screenshot of a coding exercise/technical interview question. Here's what I found: {assistant_response[:500]}... Let me know if you need more specific details or have questions about how to approach this problem.]"
                success = self.text_thread.send_text(context_msg)
                if success:
                    logger.info("Image analysis context sent to realtime thread")
                else:
                    logger.error("Failed to send image analysis context to realtime thread")
            
            logger.info("Screenshot analysis completed successfully")
            
        except Exception as e:
            error_msg = f"Error handling analysis response: {str(e)}"
            self.show_error(error_msg)
            logger.error(error_msg)
    
    def _prepare_messages_with_history(self, base64_image=None):
        """Prepare messages array for gpt-4o including chat history and image."""
        messages = []
        
        # Add system message
        messages.append({
            "role": "system", 
            "content": "You are a specialized assistant for technical interviews, analyzing screenshots of coding exercises and technical problems. Help the user understand the content of these screenshots in detail. Your analysis should be particularly useful for a candidate during a technical interview or coding assessment."
        })
        
        # Add previous conversation history (excluding the last few entries which might be UI updates)
        history_to_include = self.chat_history[:-2] if len(self.chat_history) > 2 else []
        messages.extend(history_to_include)
        
        # Add the image message
        image_url = f"data:image/jpeg;base64,{base64_image}" if base64_image else ""
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this screenshot of a potential technical interview question or coding exercise. Describe what you see in detail, extract any visible code or problem statement, explain the problem if possible, and suggest approaches or ideas to solve it."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })
        
        return messages
    
    def clear_text(self):
        """Clear the text fields."""
        self.transcription_text.clear()
        self.response_text.clear()
        self.chat_history = []
    
    def save_conversation(self):
        """Save the conversation to a JSON file."""
        try:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Conversation", "", 
                "JSON Files (*.json);;Text Files (*.txt);;All Files (*)", 
                options=options)
            
            if filename:
                if not filename.endswith('.json'):
                    filename += '.json'
                conversation_data = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": self.chat_history
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Save Completed", 
                                          f"Conversation saved in: {filename}")
        except Exception as e:
            error_msg = f"Error during save: {str(e)}"
            self.show_error(error_msg)
            logger.error(error_msg)
    
    def show_error(self, message):
        """Show an error message."""
        if "buffer too small" in message or "Conversation already has an active response" in message:
            logger.warning(f"Ignored error (log only): {message}")
        else:
            QMessageBox.critical(self, "Error", message)
    
    def closeEvent(self, event):
        """Handle application closure."""
        if self.recording and self.text_thread:
            self.transcription_text.append("\n[Application closing...]")
            try:
                self.text_thread.stop()
                self.text_thread.wait(2000)
            except Exception as e:
                logger.error("Error during application closure: " + str(e))
        event.accept()
    
    def toggle_speaking(self):
        """Toggle voice recording."""
        if not self.recording or not self.text_thread or not self.text_thread.connected:
            self.show_error("You are not connected. Start a session first.")
            return
        if not hasattr(self.text_thread, 'recording') or not self.text_thread.recording:
            self.text_thread.start_recording()
        else:
            self.text_thread.stop_recording()
    
    def stop_recording(self):
        """Method preserved for compatibility (stop handling is done in toggle_recording)."""
        logger.info("IntervistaAssistant: Stopping recording")
        pass 

    def show_think_dialog(self):
        """
        Avvia il processo di pensiero avanzato in modo parallelo e mostra i risultati nella chat principale.
        """
        try:
            logger.info("Starting Think process in background")
            
            # Se non c'è una conversazione, mostra un messaggio
            if not self.chat_history:
                QMessageBox.information(self, "No conversation", 
                                      "There's no conversation to analyze. "
                                      "Please start a conversation session first.")
                return
            
            # Verifica che siamo connessi alla sessione
            if not self.recording or not self.text_thread or not self.text_thread.connected:
                QMessageBox.warning(self, "Session not active", 
                                    "You need to start a session before using the Think function.")
                return
            
            # Aggiorna l'interfaccia per mostrare che stiamo elaborando
            self.transcription_text.append("\n[Deep analysis of the conversation in progress...]\n")
            self.response_text.append("<p><i>Advanced thinking process in progress... The application remains responsive.</i></p>")
            QApplication.processEvents()  # Aggiorna l'UI
            
            # Crea un thread per l'elaborazione asincrona
            self.think_thread = QThread()
            
            # Prepara i messaggi per l'elaborazione
            messages_for_processing = []
            for msg in self.chat_history:
                messages_for_processing.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Crea il worker per l'elaborazione
            self.think_worker = ThinkWorker(self.client, messages_for_processing)
            
            # Sposta il worker nel thread
            self.think_worker.moveToThread(self.think_thread)
            
            # Connetti i segnali e gli slot
            self.think_thread.started.connect(self.think_worker.process)
            self.think_worker.summaryComplete.connect(self.on_summary_complete)
            self.think_worker.solutionComplete.connect(self.on_solution_complete)
            self.think_worker.error.connect(self.show_error)
            self.think_worker.solutionComplete.connect(self.think_thread.quit)
            self.think_worker.error.connect(self.think_thread.quit)
            self.think_thread.finished.connect(self.think_worker.deleteLater)
            self.think_thread.finished.connect(self.think_thread.deleteLater)
            
            # Avvia il thread
            self.think_thread.start()
            
        except Exception as e:
            logger.error(f"Error during Think process startup: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
    
    def on_summary_complete(self, summary):
        """Callback invocato quando il riassunto è completato."""
        try:
            # Aggiorna l'UI con la risposta del riassunto
            self.update_response("**🧠 CONVERSATION SUMMARY (GPT-4o-mini):**\n\n" + summary)
            
            logger.info("Conversation summary completed")
            
        except Exception as e:
            error_msg = f"Error in summary handling: {str(e)}"
            self.show_error(error_msg)
            logger.error(error_msg)
    
    def on_solution_complete(self, solution):
        """Callback invocato quando la soluzione è completata."""
        try:
            # Aggiorna l'UI con la risposta della soluzione
            self.update_response("**🚀 IN-DEPTH ANALYSIS AND SOLUTION (o1-preview):**\n\n" + solution)
            
            # Invia la soluzione come testo al thread realtime per mantenere il flusso della conversazione
            if self.text_thread and self.text_thread.connected:
                context_msg = f"[I've completed an in-depth analysis of our conversation. I've identified the key problems and generated detailed solutions. If you have specific questions about any part of the solution, let me know!]"
                success = self.text_thread.send_text(context_msg)
                if success:
                    logger.info("Analysis context sent to realtime thread")
                else:
                    logger.error("Unable to send analysis context to realtime thread")
            
            logger.info("In-depth analysis process completed successfully")
            
        except Exception as e:
            error_msg = f"Error in solution handling: {str(e)}"
            self.show_error(error_msg)
            logger.error(error_msg)

    def send_text_message(self):
        """Sends a text message to the model."""
        if not self.recording or not self.text_thread or not self.text_thread.connected:
            self.show_error("You are not connected. Please start a session first.")
            return
        
        # Get text from input field
        text = self.central_widget.text_input_field.text().strip()
        
        # Check that there is text to send
        if not text:
            return
            
        # Display text in the transcription window
        current_time = datetime.now().strftime("%H:%M:%S")
        self.transcription_text.append(f"\n[Text message sent at {current_time}]\n{text}\n")
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": text})
        
        # Send text through the realtime thread
        success = self.text_thread.send_text(text)
        
        if not success:
            self.show_error("Unable to send message. Please try again.")
        else:
            # Clear input field
            self.central_widget.text_input_field.clear()

class ImageAnalysisWorker(QObject):
    """Worker class for asynchronous image analysis."""
    
    analysisComplete = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, client, messages, screenshot_path, use_local_transcription=False):
        super().__init__()
        self.client = client
        self.messages = messages
        self.screenshot_path = screenshot_path
        self.use_local_transcription = use_local_transcription
        self.base64_image = None
    
    def analyze(self):
        """Performs image analysis in the background."""
        try:
            # Convert image to base64 if not already done
            if not self.base64_image:
                with open(self.screenshot_path, "rb") as image_file:
                    self.base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Update the image URL in the message
            image_url = f"data:image/jpeg;base64,{self.base64_image}"
            self.messages[-1]["content"][1]["image_url"]["url"] = image_url
            
            # Call GPT-4o to analyze the image
            logger.info("Sending image to gpt-4o-mini for analysis")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                max_tokens=1000
            )
            
            # Get the assistant's response
            assistant_response = response.choices[0].message.content
            logger.info(f"Received response from gpt-4o: {assistant_response[:100]}...")
            
            # Emit the signal with the response
            self.analysisComplete.emit(assistant_response)
            
        except Exception as e:
            error_msg = f"Error during image analysis: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)

class ThinkWorker(QObject):
    """Worker class for asynchronous advanced thinking processing."""
    
    # Signals to communicate with the UI
    summaryComplete = pyqtSignal(str)  # Emitted when summary is completed
    solutionComplete = pyqtSignal(str)  # Emitted when solution is completed
    error = pyqtSignal(str)  # Emitted in case of error
    
    def __init__(self, client, messages):
        super().__init__()
        self.client = client
        self.messages = messages
        
    @pyqtSlot()
    def process(self):
        """Performs the advanced thinking process in background."""
        try:
            # Step 1: Generate summary with GPT-4o-mini
            logger.info("Generating summary with GPT-4o-mini")
            summary = self.generate_summary()
            self.summaryComplete.emit(summary)
            
            # Step 2: Perform in-depth analysis with o1-preview
            logger.info("Performing in-depth analysis with o1-preview")
            solution = self.generate_solution(summary)
            self.solutionComplete.emit(solution)
            
        except Exception as e:
            error_msg = f"Error during thinking process: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
    
    def generate_summary(self):
        """Generates a conversation summary using GPT-4o-mini."""
        try:
            # Create a summary prompt
            summary_prompt = {
                "role": "system",
                "content": """Analyze the conversation history and create a concise summary in English. 
                Focus on:
                1. Key problems or questions discussed
                2. Important context
                3. Any programming challenges mentioned
                4. Current state of the discussion
                
                Your summary should be comprehensive but brief, highlighting the most important aspects 
                that would help another AI model solve any programming or logical problems mentioned."""
            }
            
            # Clone messages and add system prompt
            summary_messages = [summary_prompt] + self.messages
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=summary_messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def generate_solution(self, summary):
        """Generates a detailed solution using o1-preview based on the summary."""
        try:
            # Build the prompt
            prompt = """
            I'm working on a programming or logical task. Here's the context and problem:
            
            # CONTEXT
            {}
            
            Please analyze this situation and:
            1. Identify the core problem or challenge
            2. Develop a structured approach to solve it
            3. Provide a detailed solution with code if applicable
            4. Explain your reasoning
            """.format(summary)
            
            response = self.client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            raise