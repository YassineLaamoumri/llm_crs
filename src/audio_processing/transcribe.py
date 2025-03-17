import os
from src.models.whisper_api import WhisperClient


def main():
    # Define the paths for cleaned audio and transcripts
    cleaned_audio_dir = "data/processed/cleaned_audio"
    transcripts_dir = "data/processed/transcripts"

    # Ensure the transcripts directory exists
    os.makedirs(transcripts_dir, exist_ok=True)

    # Initialize the Whisper client
    client = WhisperClient()

    # Process each audio file in the cleaned_audio directory
    for audio_filename in os.listdir(cleaned_audio_dir):
        audio_path = os.path.join(cleaned_audio_dir, audio_filename)

        # Transcribe the audio file
        transcript = client.transcribe(audio_path)

        # Determine the output filename (change extension to .txt)
        transcript_filename = os.path.splitext(audio_filename)[0] + ".txt"
        transcript_path = os.path.join(transcripts_dir, transcript_filename)

        # Save the transcript to a text file
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"Transcribed '{audio_filename}' to '{transcript_filename}'")


if __name__ == "__main__":
    main()
