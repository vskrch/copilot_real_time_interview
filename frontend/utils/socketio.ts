/**
 * Module for handling audio streaming via Google Gemini API
 */
import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

// Connection URL for Socket.IO (will be used as fallback or for signaling)
const SOCKET_URL = process.env.NEXT_PUBLIC_SOCKET_URL || 'http://127.0.0.1:8000';

// Gemini API URL
const GEMINI_API_URL = process.env.NEXT_PUBLIC_GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta';

// Interface for audio stream control
export interface AudioStreamControl {
  start: () => void;
  stop: () => void;
  isActive: boolean;
}

/**
 * Custom hook for handling audio streaming with Gemini API
 * @param sessionId Session ID
 * @returns Audio stream controls (start, stop, isActive) and session status
 */
export function useAudioStream(sessionId: string): AudioStreamControl & { sessionStatus: any } {
  const socketRef = useRef<Socket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<any>(null);
  const [isActive, setIsActive] = useState(false);
  const [sessionStatus, setSessionStatus] = useState<any>(null);
  const sessionIdRef = useRef<string>(sessionId);
  const isActiveRef = useRef<boolean>(false);
  
  // Update the sessionId reference when it changes
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  
  // Initialize Socket.IO (for signaling and fallback)
  useEffect(() => {
    if (!socketRef.current) {
      console.log(`[SOCKET.IO] Initializing Socket.IO for session ${sessionId}`);
      socketRef.current = io(SOCKET_URL);
      
      socketRef.current.on('connect', () => {
        console.log(`[SOCKET.IO] Successfully connected [ID: ${socketRef.current?.id}] for session ${sessionId}`);
      });
      
      socketRef.current.on('disconnect', () => {
        console.log('[SOCKET.IO] Disconnected');
        setIsActive(false);
        isActiveRef.current = false;
      });
    
      socketRef.current.on('error', (error: any) => {
        console.error('[SOCKET.IO] Error:', error);
      });
    
      socketRef.current.on('connect_error', (error: any) => {
        console.error('[SOCKET.IO] Connection error:', error);
      });
      
      // Join the session room to receive updates for this specific session
      if (sessionId) {
        socketRef.current.emit('join', { session_id: sessionId });
        
        // Inform the server to use Gemini API for this session
        socketRef.current.emit('set_api_preference', { 
          session_id: sessionId,
          preference: 'gemini'
        });
      }
    }
    
    // Update room when sessionId changes
    if (socketRef.current && sessionId) {
      socketRef.current.emit('join', { session_id: sessionId });
    }
    
    // Cleanup when component unmounts
    return () => {
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
      }
      
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [sessionId]);
  
  // Add session status update handler in a separate useEffect
  useEffect(() => {
    if (socketRef.current) {
      socketRef.current.on('session_status_update', (status: any) => {
        console.log('[SOCKET.IO] Session status update:', status);
        setSessionStatus(status);
      });
    }
    
    return () => {
      if (socketRef.current) {
        socketRef.current.off('session_status_update');
      }
    };
  }, []);
  
  // Register timestamp of last audio processing
  const lastProcessTimestampRef = useRef<number>(Date.now());
  
  /**
   * Start audio recording and streaming to the server using Gemini API
   */
  const startRecording = async () => {
    try {
      if (!socketRef.current) {
        throw new Error('[AUDIO] Socket.IO not initialized');
      }

      console.log(`[AUDIO] Starting audio recording for session ${sessionIdRef.current} with Gemini API`);
      
      // Set active flags
      setIsActive(true);
      isActiveRef.current = true;
      console.log(`[AUDIO] isActive flag set to true before starting recording`);
      
      // Request microphone permission
      try {
        console.log('[AUDIO] Requesting microphone permissions...');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          } 
        });
        
        // Verify audio tracks
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) {
          throw new Error('No audio tracks available');
        }
        
        mediaStreamRef.current = stream;
        console.log('[AUDIO] Microphone stream activated successfully');
        
      } catch (err) {
        console.error('[AUDIO] Error obtaining microphone permissions:', err);
        alert('Please grant microphone permissions to continue with audio recording.');
        throw new Error('Microphone permission denied or device not available');
      }
      
      // Set up AudioContext for Gemini API
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000  // Gemini prefers 16kHz
      });
      
      // Ensure audioContext is running
      if (audioContext.state !== 'running') {
        await audioContext.resume();
      }
      
      console.log(`[AUDIO] AudioContext configured with sampling rate: ${audioContext.sampleRate}Hz for Gemini API`);
      
      const source = audioContext.createMediaStreamSource(mediaStreamRef.current);
      
      // Create processor node with buffer size optimized for Gemini
      processorRef.current = audioContext.createScriptProcessor(4096, 1, 1);
      
      // Buffer for accumulating audio data
      let audioAccumulatorRef: Int16Array[] = [];
      const minimumAudioLength = 3200; // 200ms of audio at 16kHz
      
      // Audio processing callback
      processorRef.current.onaudioprocess = (e: AudioProcessingEvent) => {
        const now = Date.now();
        if (now - lastProcessTimestampRef.current > 3000) {
          console.log(`[AUDIO DEBUG] onaudioprocess active, last process: ${new Date(lastProcessTimestampRef.current).toISOString()}`);
          lastProcessTimestampRef.current = now;
        }
        
        if (!socketRef.current || !isActiveRef.current) {
          return;
        }
        
        // Get audio data
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Check for significant audio
        const hasAudioData = inputData.some(value => Math.abs(value) > 0.01);
        if (!hasAudioData && Math.random() < 0.05) {
          console.log('No significant audio data detected');
          return;
        }
        
        try {
          // Convert to Int16Array for Gemini API
          const scaledData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            scaledData[i] = Math.max(-32768, Math.min(32767, Math.round(inputData[i] * 32767)));
          }
          
          // Verify non-zero values
          const nonZeroCount = Array.from(scaledData).filter(v => v !== 0).length;
          if (nonZeroCount < 10) {
            return;
          }
          
          // Add to accumulation buffer
          audioAccumulatorRef.push(scaledData);
          
          // Calculate total samples
          const totalSamples = audioAccumulatorRef.reduce((sum, array) => sum + array.length, 0);
          
          // Check if we have enough data
          if (totalSamples < minimumAudioLength) {
            return;
          }
          
          // Merge audio fragments
          const mergedArray = new Int16Array(totalSamples);
          let offset = 0;
          
          for (const array of audioAccumulatorRef) {
            mergedArray.set(array, offset);
            offset += array.length;
          }
          
          // Final check for sufficient data
          if (mergedArray.length < minimumAudioLength) {
            return;
          }
          
          // Convert to base64 for Gemini API
          const audioBuffer = new ArrayBuffer(mergedArray.length * 2);
          const view = new DataView(audioBuffer);
          for (let i = 0; i < mergedArray.length; i++) {
            view.setInt16(i * 2, mergedArray[i], true);
          }
          const base64Audio = btoa(
            Array.from(new Uint8Array(audioBuffer))
              .map(b => String.fromCharCode(b))
              .join('')
          );
          
          // Send to server with Gemini API preference
          console.log(`[AUDIO] Sending ${mergedArray.length} samples to Gemini API`);
          
          socketRef.current.emit('audio_data_gemini', {
            session_id: sessionIdRef.current,
            audio_data: base64Audio,
            sample_rate: audioContext.sampleRate,
            encoding: 'LINEAR16'
          }, (acknowledgement: any) => {
            if (acknowledgement && acknowledgement.received) {
              console.log(`[AUDIO] Server confirmed receipt for Gemini processing`);
            } else {
              console.error(`[AUDIO] Server error or no acknowledgement`, acknowledgement);
            }
          });
          
          // Reset accumulator
          audioAccumulatorRef = [];
          
        } catch (error) {
          console.error('Error sending audio data to Gemini API:', error);
        }
      };

      // Connect audio nodes
      source.connect(processorRef.current);
      processorRef.current.connect(audioContext.destination);
      
    } catch (error) {
      console.error('Error during Gemini audio recording:', error);
      setIsActive(false);
      isActiveRef.current = false;
    }
  };

  return {
    start: startRecording,
    stop: () => {
      // Stop audio recording
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
      }
      
      // Notify server about stopping with Gemini
      if (socketRef.current && sessionIdRef.current) {
        socketRef.current.emit('stop_audio_gemini', { session_id: sessionIdRef.current });
      }
      
      setIsActive(false);
      isActiveRef.current = false;
      console.log('Gemini audio recording stopped');
    },
    isActive,
    sessionStatus
  };
}