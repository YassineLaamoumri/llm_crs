import os
import requests
from openai import OpenAI
from fastapi_app.config import OPENAI_API_KEY  # Import the API key from config

client = OpenAI()


class WhisperClient:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set the 'OPENAI_API_KEY' environment variable or pass it directly."
            )
        client.api_key = self.api_key

    def transcribe(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="srt"
            )
        return transcript


# Usage example (can be imported in your FastAPI endpoints):
# client = WhisperClient()
# transcription = client.transcribe_audio("path/to/audio.wav")
