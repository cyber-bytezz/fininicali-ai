"""
Voice Agent Module.

This module implements an agent that handles speech-to-text and text-to-speech operations.
"""
import os
import logging
import tempfile
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import speech_recognition as sr
import pyttsx3
import numpy as np
import whisper

from agents.base_agent import BaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceAgent(BaseAgent):
    """Agent for handling speech-to-text and text-to-speech operations."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None,
        whisper_model: str = "base",
        voice_rate: int = 150,
        voice_volume: float = 1.0
    ):
        """
        Initialize the voice agent.
        
        Args:
            agent_id: Optional unique identifier for the agent
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium')
            voice_rate: Speech rate for text-to-speech
            voice_volume: Volume for text-to-speech
        """
        super().__init__(agent_id)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize whisper model
        self.whisper_model_name = whisper_model
        self.whisper_model = None  # Lazy-load to save memory
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', voice_rate)
        self.tts_engine.setProperty('volume', voice_volume)
        
        # Cache directory for audio files
        self.cache_dir = os.path.join(tempfile.gettempdir(), "voice_agent_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.log_activity(f"Voice Agent initialized (whisper_model={whisper_model})")
    
    def _load_whisper_model(self):
        """Lazy-load the Whisper model to save memory."""
        if self.whisper_model is None:
            self.log_activity(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            self.log_activity("Whisper model loaded")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data for voice operations.
        
        Args:
            input_data: Dictionary containing request parameters
                - action: Type of voice operation to perform
                - params: Parameters for the specific action
            
        Returns:
            Dictionary containing operation results
        """
        if not self.validate_input(input_data):
            self.log_activity("Invalid input data", "error")
            return {"error": "Invalid input data"}
        
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        self.log_activity(f"Processing action: {action} with params: {params}")
        
        try:
            if action == "speech_to_text":
                audio_file = params.get("audio_file")
                use_whisper = params.get("use_whisper", True)
                
                if not audio_file:
                    return {"error": "Audio file is required"}
                
                text = await self.speech_to_text(audio_file, use_whisper)
                return {
                    "text": text,
                    "metadata": {
                        "audio_file": audio_file,
                        "use_whisper": use_whisper
                    }
                }
                
            elif action == "text_to_speech":
                text = params.get("text")
                output_file = params.get("output_file")
                
                if not text:
                    return {"error": "Text is required"}
                
                if not output_file:
                    # Generate a default output file
                    output_file = os.path.join(self.cache_dir, f"tts_output_{self.agent_id}.mp3")
                
                success = await self.text_to_speech(text, output_file)
                return {
                    "success": success,
                    "output_file": output_file if success else None,
                    "metadata": {
                        "text_length": len(text)
                    }
                }
                
            elif action == "set_voice_properties":
                rate = params.get("rate")
                volume = params.get("volume")
                voice_id = params.get("voice_id")
                
                success = self.set_voice_properties(rate, volume, voice_id)
                return {
                    "success": success,
                    "current_properties": {
                        "rate": self.tts_engine.getProperty('rate'),
                        "volume": self.tts_engine.getProperty('volume'),
                        "voice_id": self.tts_engine.getProperty('voice')
                    }
                }
                
            else:
                self.log_activity(f"Unknown action: {action}", "warning")
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.log_activity(f"Error processing action {action}: {str(e)}", "error")
            return {
                "error": str(e),
                "action": action,
                "params": params
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, dict):
            return False
            
        if "action" not in input_data:
            return False
            
        # Validate specific actions
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        if action == "speech_to_text" and "audio_file" not in params:
            return False
            
        if action == "text_to_speech" and "text" not in params:
            return False
            
        return True
    
    async def speech_to_text(
        self, 
        audio_file: str, 
        use_whisper: bool = True
    ) -> str:
        """
        Convert speech to text.
        
        Args:
            audio_file: Path to the audio file
            use_whisper: Whether to use Whisper for STT
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            self.log_activity(f"Audio file not found: {audio_file}", "error")
            return "Error: Audio file not found."
        
        try:
            if use_whisper:
                # Use Whisper for speech recognition
                self._load_whisper_model()
                
                self.log_activity(f"Transcribing {audio_file} with Whisper")
                result = self.whisper_model.transcribe(audio_file)
                text = result["text"].strip()
                
                self.log_activity(f"Whisper transcription completed: {len(text)} chars")
                return text
            else:
                # Use SpeechRecognition library
                self.log_activity(f"Transcribing {audio_file} with SpeechRecognition")
                
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    
                self.log_activity(f"SpeechRecognition transcription completed: {len(text)} chars")
                return text
                
        except Exception as e:
            self.log_activity(f"Error in speech-to-text: {str(e)}", "error")
            return f"Error in speech-to-text: {str(e)}"
    
    async def text_to_speech(self, text: str, output_file: str) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            self.log_activity(f"Converting text to speech: {len(text)} chars")
            
            # Save to a temporary file first
            temp_file = os.path.join(self.cache_dir, f"temp_{self.agent_id}.wav")
            
            # Convert text to speech
            self.tts_engine.save_to_file(text, temp_file)
            self.tts_engine.runAndWait()
            
            # Check if the file was created
            if not os.path.exists(temp_file):
                self.log_activity("TTS engine did not create the audio file", "error")
                return False
            
            # Copy or convert the file to the desired format if needed
            # For simplicity, we'll just copy it
            import shutil
            shutil.copy2(temp_file, output_file)
            
            self.log_activity(f"Text-to-speech completed, saved to {output_file}")
            return True
            
        except Exception as e:
            self.log_activity(f"Error in text-to-speech: {str(e)}", "error")
            return False
    
    def set_voice_properties(
        self, 
        rate: Optional[int] = None, 
        volume: Optional[float] = None,
        voice_id: Optional[str] = None
    ) -> bool:
        """
        Set properties for the text-to-speech engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            voice_id: Voice identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if rate is not None:
                self.tts_engine.setProperty('rate', rate)
                self.log_activity(f"Set speech rate to {rate}")
            
            if volume is not None:
                self.tts_engine.setProperty('volume', volume)
                self.log_activity(f"Set speech volume to {volume}")
            
            if voice_id is not None:
                # Check if the voice exists
                voices = self.tts_engine.getProperty('voices')
                voice_exists = any(voice.id == voice_id for voice in voices)
                
                if voice_exists:
                    self.tts_engine.setProperty('voice', voice_id)
                    self.log_activity(f"Set voice to {voice_id}")
                else:
                    available_voices = [voice.id for voice in voices]
                    self.log_activity(
                        f"Voice {voice_id} not found. Available voices: {available_voices}",
                        "warning"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.log_activity(f"Error setting voice properties: {str(e)}", "error")
            return False
    
    def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Get a list of available voices.
        
        Returns:
            List of dictionaries containing voice information
        """
        try:
            voices = self.tts_engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_list.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': voice.gender,
                    'age': voice.age
                })
            
            self.log_activity(f"Found {len(voice_list)} available voices")
            return voice_list
            
        except Exception as e:
            self.log_activity(f"Error getting available voices: {str(e)}", "error")
            return []
