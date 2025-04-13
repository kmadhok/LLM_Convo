import os
import argparse
import tempfile
import wave
import sys
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from collections import deque

class TranscriptionManager:
    """Manages transcriptions and provides access to the stored text."""
    
    def __init__(self):
        self.full_transcript = ""
        self.session_transcripts = []
        self.current_transcript = ""
    
    def add_transcription(self, text):
        """Add new transcription to the history."""
        self.current_transcript = text
        self.session_transcripts.append(text)
        self.full_transcript += text + " "
        return text
    
    def get_full_transcript(self):
        """Get the full transcript from the entire session."""
        return self.full_transcript.strip()
    
    def get_current_transcript(self):
        """Get the most recent transcription."""
        return self.current_transcript
    
    def get_all_transcripts(self):
        """Get all individual transcriptions as a list."""
        return self.session_transcripts
    
    def clear_transcript(self):
        """Clear all stored transcriptions."""
        self.full_transcript = ""
        self.session_transcripts = []
        self.current_transcript = ""
    
    def save_transcript(self, filename="transcript.txt"):
        """Save the full transcript to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.get_full_transcript())
        return filename

class WakeWordDetector:
    def __init__(self, wake_word="listen", model_size="tiny", device="cpu"):
        """
        Initialize wake word detector.
        
        Args:
            wake_word (str): Word to trigger recording
            model_size (str): Model size for wake word detection
            device (str): Device for inference
        """
        self.wake_word = wake_word.lower().strip()
        self.model = WhisperModel(model_size, device=device)
        self.listening_for_wake_word = False
        self.audio_buffer = deque(maxlen=3)  # Store ~3 seconds of audio
        self.is_detected = False
    
    def start_listening(self):
        """Start listening for wake word in the background."""
        self.listening_for_wake_word = True
        self.is_detected = False
        
        # Start background thread to listen for wake word
        self.listen_thread = threading.Thread(target=self._listen_for_wake_word)
        self.listen_thread.daemon = True
        self.listen_thread.start()
    
    def stop_listening(self):
        """Stop listening for wake word."""
        self.listening_for_wake_word = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=1)
    
    def _listen_for_wake_word(self):
        """Background thread that listens for wake word."""
        sample_rate = 16000
        buffer_duration = 3  # seconds
        chunk_duration = 1.0  # Process in 1-second chunks
        
        # Setup audio buffer
        buffer = np.zeros(int(sample_rate * buffer_duration), dtype=np.float32)
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream to collect audio data."""
            if status:
                print(f"Audio callback status: {status}")
            # Add new audio to buffer (shift old audio out)
            buffer[:-frames] = buffer[frames:]
            buffer[-frames:] = indata[:, 0]
            
            # Store in deque for later use if wake word is detected
            self.audio_buffer.append(indata.copy())
        
        # Start audio stream
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback,
                           blocksize=int(sample_rate * chunk_duration), dtype='float32'):
            print(f"Listening for wake word: '{self.wake_word}'...")
            
            while self.listening_for_wake_word:
                # Every second, check if wake word is present
                time.sleep(chunk_duration)
                
                # Save buffer to temporary file
                temp_file = self._save_audio_to_temp(buffer, sample_rate)
                
                # Check for wake word
                segments, _ = self.model.transcribe(temp_file, language="en")
                transcription = " ".join([segment.text for segment in segments]).lower()
                
                # Clean up temp file
                os.remove(temp_file)
                
                # Check if wake word in transcription
                if self.wake_word in transcription:
                    print(f"Wake word detected: '{transcription}'")
                    self.is_detected = True
                    self.listening_for_wake_word = False
                    break
    
    def _save_audio_to_temp(self, audio_data, sample_rate):
        """Save audio data to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            # Convert float32 array to int16
            audio_data_int = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_data_int.tobytes())
        
        return temp_file.name
    
    def get_recent_audio(self):
        """Get the most recent audio from buffer."""
        if not self.audio_buffer:
            return None, 16000
        
        # Combine all chunks in the buffer
        audio_data = np.vstack(self.audio_buffer)
        return audio_data, 16000

def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    Record audio from microphone.
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Sample rate in Hz
        channels (int): Number of audio channels
    
    Returns:
        numpy.ndarray: Recorded audio data
        int: Sample rate
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    return audio_data, sample_rate

def save_audio_to_temp_file(audio_data, sample_rate, channels=1):
    """
    Save recorded audio to a temporary WAV file.
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sample_rate (int): Sample rate in Hz
        channels (int): Number of audio channels
        
    Returns:
        str: Path to the temporary WAV file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Convert float32 array to int16
        audio_data_int = (audio_data * 32767).astype(np.int16)
        wf.writeframes(audio_data_int.tobytes())
    
    return temp_file.name

def transcribe_audio(audio_path, model_size="base", device="cpu", language=None):
    """
    Transcribe audio using faster-whisper.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device (str): Device to use for inference ('cpu', 'cuda', or 'auto')
        language (str, optional): Language code for transcription (e.g. 'en', 'fr'). 
                                 If None, language will be detected automatically.
    
    Returns:
        str: Transcribed text
    """
    model = WhisperModel(model_size, device=device)
    
    print(f"Transcribing audio...")
    segments, info = model.transcribe(audio_path, language=language)
    
    if language is None and hasattr(info, 'language'):
        print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    
    transcription = ""
    print("Transcription:")
    for segment in segments:
        text = segment.text
        transcription += text
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")
    
    return transcription
def no_wake_transcribe_from_microphone_simplified(duration=5, model_size="base", device="cpu"):
    """
    Record audio from microphone, transcribe it, and return the transcription.
    
    Args:
        duration (int): Recording duration in seconds
        model_size (str): Size of the Whisper model to use
        device (str): Device to use for inference
    
    Returns:
        str: Transcribed speech or None if no speech detected
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 👈 Prevent GPU queries
    # Load transcription model
    print(f"Loading transcription model: {model_size}")
    transcription_model = WhisperModel(model_size, device=device)
    
    try:
        print("Recording audio...")
        # Record audio for transcription
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save to temporary file
        temp_file = save_audio_to_temp_file(audio_data, sample_rate)
        
        # Transcribe with main model
        print(f"Transcribing audio...")
        segments, info = transcription_model.transcribe(temp_file)
        
        # Extract transcription
        transcription = ""
        for segment in segments:
            transcription += segment.text
        
        # Clean up temporary file
        os.remove(temp_file)
        
        return transcription.strip() if transcription.strip() else None
        
    except KeyboardInterrupt:
        print("\nExiting...")
        return None
def transcribe_from_microphone_simplified(duration=5, model_size="tiny", device="cpu", wake_word="listen"):
    """
    Record audio from microphone, transcribe it, and return the transcription.
    
    Args:
        duration (int): Recording duration in seconds
        model_size (str): Size of the Whisper model to use
        device (str): Device to use for inference
        wake_word (str): Word to trigger recording
    
    Returns:
        str: Transcribed speech or None if no speech detected
    """
    # Initialize wake word detector (using tiny model for speed)
    wake_detector = WakeWordDetector(wake_word=wake_word, model_size="tiny", device=device)
    
    # Load transcription model
    print(f"Loading transcription model: {model_size}")
    transcription_model = WhisperModel(model_size, device=device)
    
    try:
        print(f"Listening for wake word: '{wake_word}'")
        
        # Start listening for wake word
        wake_detector.start_listening()
        
        # Wait for wake word to be detected
        while wake_detector.listening_for_wake_word:
            time.sleep(0.1)
        
        # If wake word wasn't detected (e.g., interrupted), return None
        if not wake_detector.is_detected:
            return None
            
        # Record audio for transcription
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save to temporary file
        temp_file = save_audio_to_temp_file(audio_data, sample_rate)
        
        # Transcribe with main model
        print(f"Transcribing audio...")
        segments, info = transcription_model.transcribe(temp_file)
        
        # Extract transcription
        transcription = ""
        for segment in segments:
            transcription += segment.text
        
        # Clean up temporary file
        os.remove(temp_file)
        
        return transcription.strip() if transcription.strip() else None
        
    except KeyboardInterrupt:
        wake_detector.stop_listening()
        print("\nExiting...")
        return None
    finally:
        # Ensure wake detector is stopped
        if hasattr(wake_detector, 'stop_listening'):
            wake_detector.stop_listening()
def transcribe_from_microphone(duration=5, model_size="base", device="cpu", language=None, wake_word="listen", 
                               save_transcript=True, transcript_file="transcript.txt", commands=None,
                               use_groq=False, groq_script_path="groq_integration.py"):
    """
    Record audio from microphone and transcribe it.
    
    Args:
        duration (int): Recording duration in seconds
        model_size (str): Size of the Whisper model to use
        device (str): Device to use for inference
        language (str, optional): Language code for transcription
        wake_word (str): Word to trigger recording
        save_transcript (bool): Whether to save the transcript to file periodically
        transcript_file (str): Path to save the transcript
        commands (dict): Dictionary of voice commands and their actions
        use_groq (bool): Whether to send transcriptions to Groq LLM
        groq_script_path (str): Path to the Groq integration script
    
    Returns:
        TranscriptionManager: Manager object with all transcriptions
    """
    # Initialize transcript manager
    transcript_manager = TranscriptionManager()
    
    # Default commands if none provided
    if commands is None:
        commands = {
            "clear transcript": lambda: transcript_manager.clear_transcript(),
            "save transcript": lambda: transcript_manager.save_transcript(transcript_file),
            "exit program": lambda: sys.exit(0),
            "quit program": lambda: sys.exit(0)
        }
    
    print(f"Loading main transcription model: {model_size}")
    transcription_model = WhisperModel(model_size, device=device)
    
    # Initialize wake word detector (using tiny model for speed)
    wake_detector = WakeWordDetector(wake_word=wake_word, model_size="tiny", device=device)
    
    try:
        print(f"Available commands: {', '.join(commands.keys())}")
        print(f"Transcript will be saved to: {transcript_file}")
        
        if use_groq:
            print(f"Groq LLM integration enabled - responses will be generated for your speech")
        
        while True:
            # Start listening for wake word
            wake_detector.start_listening()
            
            # Wait for wake word to be detected
            while wake_detector.listening_for_wake_word:
                time.sleep(0.1)
            
            # If wake word wasn't detected (e.g., interrupted), break
            if not wake_detector.is_detected:
                break
            
            # Record audio for transcription
            audio_data, sample_rate = record_audio(duration=duration)
            
            # Save to temporary file
            temp_file = save_audio_to_temp_file(audio_data, sample_rate)
            
            # Transcribe with main model
            print(f"Transcribing audio...")
            segments, info = transcription_model.transcribe(temp_file, language=language)
            
            if language is None and hasattr(info, 'language'):
                print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
            
            transcription = ""
            print("Transcription:")
            for segment in segments:
                text = segment.text
                transcription += text
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")
            
            # Add to transcript manager
            transcript_manager.add_transcription(transcription)
            
            # Get the latest speech for potential processing
            latest_speech = transcript_manager.get_current_transcript()
            print(f"Latest speech: {latest_speech}")
            
            # Send to Groq if enabled
            if use_groq and latest_speech.strip():
                try:
                    print("\nSending to Groq LLM...")
                    groq_response = send_to_groq(latest_speech, groq_script_path)
                    print("\n=== Groq LLM Response ===")
                    print(groq_response)
                    print("=========================\n")
                except Exception as e:
                    print(f"Error sending to Groq: {str(e)}")
            
            # Check for commands in transcription
            lower_transcription = transcription.lower()
            for cmd, action in commands.items():
                if cmd.lower() in lower_transcription:
                    print(f"Executing command: {cmd}")
                    action()
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # Periodically save transcript if enabled
            if save_transcript and len(transcript_manager.get_all_transcripts()) % 5 == 0:
                transcript_manager.save_transcript(transcript_file)
                print(f"Transcript saved to: {transcript_file}")
            
            # Display current word count
            word_count = len(transcript_manager.get_full_transcript().split())
            print(f"\nTotal transcript word count: {word_count}")
            print(f"--- Say '{wake_word}' to record again or press Ctrl+C to exit ---\n")
            
    except KeyboardInterrupt:
        wake_detector.stop_listening()
        print("\nExiting...")
        
        # Save transcript before exiting
        if save_transcript:
            transcript_manager.save_transcript(transcript_file)
            print(f"Final transcript saved to: {transcript_file}")
    
    return transcript_manager

def send_to_groq(text, groq_script_path):
    """
    Send text to Groq LLM via the separate Python script.
    
    Args:
        text (str): Text to send to Groq
        groq_script_path (str): Path to the Groq integration script
    
    Returns:
        str: Response from Groq
    """
    import subprocess
    
    try:
        # Call the Groq script as a subprocess
        result = subprocess.run(
            [sys.executable, groq_script_path, text],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the stdout result
        return result.stdout.strip()
        
    except subprocess.CalledProcessError as e:
        # If the subprocess failed, return the error
        return f"Error from Groq script: {e.stderr.strip()}"
    except Exception as e:
        return f"Failed to call Groq script: {str(e)}"

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Transcribe speech from microphone using faster-whisper")
    parser.add_argument("--duration", type=int, default=5,
                        help="Recording duration in seconds")
    parser.add_argument("--model-size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
                        help="Size of the Whisper model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"],
                        help="Device to use for inference")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (e.g. 'en', 'fr') or None for auto-detection")
    parser.add_argument("--file", type=str, default=None,
                        help="Optional: Path to audio file (if provided, microphone will not be used)")
    parser.add_argument("--wake-word", type=str, default="listen",
                        help="Wake word to trigger recording (default: 'listen')")
    parser.add_argument("--transcript-file", type=str, default="transcript.txt",
                        help="File to save transcript (default: transcript.txt)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the transcript to file")
    parser.add_argument("--use-groq", action="store_true",
                        help="Send transcriptions to Groq LLM for responses")
    parser.add_argument("--groq-script", type=str, default="groq_integration.py",
                        help="Path to the Groq integration script (default: groq_integration.py)")
    
    args = parser.parse_args()
    
    if args.file:
        # Transcribe from file
        if not os.path.exists(args.file):
            print(f"Error: Audio file '{args.file}' does not exist")
            exit(1)
        
        transcription = transcribe_audio(
            args.file, 
            model_size=args.model_size,
            device=args.device,
            language=args.language
        )
        
        # Save transcription to file
        output_path = os.path.splitext(args.file)[0] + ".txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print(f"\nTranscription saved to: {output_path}")
    else:
        # Define voice commands
        transcript_manager = TranscriptionManager()
        
        def save_transcript():
            filename = transcript_manager.save_transcript(args.transcript_file)
            print(f"Transcript saved to: {filename}")
            return filename
        
        commands = {
            "clear transcript": lambda: transcript_manager.clear_transcript() and print("Transcript cleared!"),
            "save transcript": save_transcript,
            "exit program": lambda: sys.exit(0),
            "quit program": lambda: sys.exit(0),
            "show transcript": lambda: print(f"\n--- FULL TRANSCRIPT ---\n{transcript_manager.get_full_transcript()}\n--------------------")
        }
        
        # Check if Groq integration script exists
        if args.use_groq and not os.path.exists(args.groq_script):
            print(f"Warning: Groq integration script '{args.groq_script}' not found. Creating default script...")
            
            # Create default Groq script
            with open(args.groq_script, "w", encoding="utf-8") as f:
                f.write("""import os
import sys
import json
from groq import Groq

def get_llm_response(input_text, model="llama3-8b-8192"):
    \"\"\"
    Get a response from Groq LLM.
    
    Args:
        input_text (str): The text to send to the LLM
        model (str): The Groq model to use
        
    Returns:
        str: The LLM response
    \"\"\"
    # Get Groq API key from environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set. Please set it with your Groq API key."
    
    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond concisely to the user's input."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            model=model,
        )
        
        # Extract and return the response
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"Error: Failed to get response from Groq: {str(e)}"

if __name__ == "__main__":
    # Check if input is provided via command line arguments
    if len(sys.argv) > 1:
        # The input text is the combined arguments (to handle spaces)
        input_text = " ".join(sys.argv[1:])
        response = get_llm_response(input_text)
        print(response)
    
    # Check if input is provided via stdin (pipe)
    elif not sys.stdin.isatty():
        # Read from stdin (for piping)
        input_text = sys.stdin.read().strip()
        if input_text:
            response = get_llm_response(input_text)
            print(response)
    
    else:
        print("Error: No input provided. Please provide text via command line arguments or pipe.")
        sys.exit(1)
""")
            print(f"Default Groq integration script created at '{args.groq_script}'")
            print("NOTE: You need to install the groq Python package and set the GROQ_API_KEY environment variable")
            print("Install with: pip install groq")
            print("Set API key with: export GROQ_API_KEY='your-api-key'")
        
        # Transcribe from microphone
        transcribe_from_microphone(
            duration=args.duration,
            model_size=args.model_size,
            device=args.device,
            language=args.language,
            wake_word=args.wake_word,
            save_transcript=not args.no_save,
            transcript_file=args.transcript_file,
            commands=commands,
            use_groq=args.use_groq,
            groq_script_path=args.groq_script
        )
