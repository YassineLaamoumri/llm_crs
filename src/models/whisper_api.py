import os
import requests


class WhisperClient:
    def __init__(self):
        self.api_key = os.getenv("WHISPER_API_KEY")
        self.endpoint = "https://api.whisper.example.com/v1/transcribe"  # Replace with the actual endpoint

    def transcribe_audio(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            files = {"file": audio_file}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(self.endpoint, headers=headers, files=files)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()  # Adjust based on actual API response structure


# Usage example (can be imported in your FastAPI endpoints):
# client = WhisperClient()
# transcription = client.transcribe_audio("path/to/audio.wav")
