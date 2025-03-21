import os
import json
import numpy as np
from pathlib import Path
import sys
import os
import logging
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
import soundfile as sf


def find_project_root(marker: str = "README.md") -> Path:
    """
    Walks up the directory tree to find the project root by searching for a marker file.

    Parameters:
        marker (str): The filename to look for that indicates the project root.

    Returns:
        Path: The project root directory.

    Raises:
        FileNotFoundError: If the marker file is not found in any parent directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root marker: {marker}")


def write_content(file_path, content):
    """
    Write content to a file.
    - If content is a string, write it directly.
    - If content is a dict or list, write it as pretty-printed JSON.
    - Otherwise, convert the content to a string and write it.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        if isinstance(content, str):
            f.write(content)
        else:
            try:
                formatted = json.dumps(content, ensure_ascii=False, indent=4)
                f.write(formatted)
            except (TypeError, ValueError):
                f.write(str(content))


def extract_json_from_transcript(transcript):
    """
    Extracts a JSON substring from the transcript string by finding the first '{'
    and the last '}'.
    """
    start_index = transcript.find("{")
    end_index = transcript.rfind("}") + 1
    return transcript[start_index:end_index]


def file_needs_processing(input_path, output_path):
    """
    Check if a file needs processing by verifying if the output file exists
    and if it's older than the input file.

    Parameters:
        input_path (str/Path): Path to the input file.
        output_path (str/Path): Path to the output file.

    Returns:
        bool: True if the file needs processing, False otherwise.
    """
    # If output file doesn't exist, it needs processing
    if not os.path.exists(output_path):
        return True

    # Compare modification times - if input is newer than output, reprocess
    input_mtime = os.path.getmtime(input_path)
    output_mtime = os.path.getmtime(output_path)

    return input_mtime > output_mtime


def process_audio(input_audio_path, output_audio_path, sr=16000):
    """
    Process an audio file by reducing noise and applying audio effects using Pedalboard.

    Parameters:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str): Path to save the processed audio file.
        sr (int): Target sampling rate.

    Returns:
        bool: True if processing was performed, False if skipped.
    """
    # Check if processing is needed
    if not file_needs_processing(input_audio_path, output_audio_path):
        print(
            f"Skipping audio cleaning for {os.path.basename(input_audio_path)} - already processed"
        )
        return False

    print(f"Cleaning audio: {os.path.basename(input_audio_path)}")

    # Load audio using Pedalboard's AudioFile and resample to desired sampling rate
    with AudioFile(input_audio_path).resampled_to(sr) as f:
        audio = f.read(f.frames)

    # Reduce noise
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

    # Set up the Pedalboard with desired effects
    board = Pedalboard(
        [
            NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
            Compressor(threshold_db=-16, ratio=4),
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
            Gain(gain_db=2),
        ]
    )

    effected_audio = board(reduced_noise, sr)

    # If the audio is 2D and the first dimension represents channels, transpose it.
    if effected_audio.ndim == 2 and effected_audio.shape[0] == 1:
        effected_audio = effected_audio.T

    # Save the processed audio file
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    sf.write(output_audio_path, effected_audio, sr, format="WAV")
    print(f"Processed audio saved to: {output_audio_path}")
    return True


def process_transcript(audio_path, transcripts_dir, knowledge_base_dir, client):
    """
    Processes an individual audio file to generate transcripts:
    - Generates the transcript via the GeminiClient.
    - Extracts and loads the JSON data.
    - Separates Arabic and English data.
    - Writes both the transcript and knowledge base entries to the appropriate file.

    Returns:
        bool: True if processing was performed, False if skipped.
    """
    # Determine the output filename (change extension to .txt)
    audio_filename = os.path.basename(audio_path)
    base_filename = os.path.splitext(audio_filename)[0] + ".txt"

    # Define file paths
    transcript_path_arabic = os.path.join(transcripts_dir, "arabic", base_filename)

    # Check if transcript already exists and is up to date
    if not file_needs_processing(audio_path, transcript_path_arabic):
        print(
            f"Skipping transcript generation for {audio_filename} - already processed"
        )
        return False

    print(f"Generating transcript for {audio_filename}...")

    try:
        transcript = client.generate_audio_processing(audio_path)
        clean_json_str = extract_json_from_transcript(transcript)
        transcript_dict = json.loads(clean_json_str)

        # Extract language-specific data
        arabic_data = transcript_dict.get("arabic", {})
        english_data = transcript_dict.get("english", {})

        arabic_conversation = arabic_data.get("cleaned_conversation", "")
        arabic_knowledge = arabic_data.get("knowledge_base_entry", "")
        english_conversation = english_data.get("cleaned_conversation", "")
        english_knowledge = english_data.get("knowledge_base_entry", "")

        # Define file paths
        transcript_path_english = os.path.join(
            transcripts_dir, "english", base_filename
        )
        knowledge_path_arabic = os.path.join(
            knowledge_base_dir, "arabic", base_filename
        )
        knowledge_path_english = os.path.join(
            knowledge_base_dir, "english", base_filename
        )

        # Write the conversation and knowledge base data using the helper function
        write_content(transcript_path_arabic, arabic_conversation)
        write_content(transcript_path_english, english_conversation)
        write_content(knowledge_path_arabic, arabic_knowledge)
        write_content(knowledge_path_english, english_knowledge)

        print(f"Transcript processed: '{audio_filename}' to '{base_filename}'")
        return True
    except Exception as e:
        print(f"Error processing transcript for {audio_path}: {e}")
        return False


def main():
    # Retrieve the project home dynamically using the marker file
    PROJECT_HOME = find_project_root()
    # Get current file's directory and set up project path
    current_dir = Path(os.getcwd())
    project_root = current_dir

    # Navigate to project root more robustly
    while not (project_root / "fastapi_app").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError(
                "Could not find project root with fastapi_app directory"
            )
        # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # Add project root to sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Added {project_root} to Python path")

    from src.models.gemini_api import GeminiClient

    # Define paths for all stages of the pipeline
    raw_folder = PROJECT_HOME / "data" / "raw"
    cleaned_audio_dir = PROJECT_HOME / "data" / "processed" / "cleaned_audio"
    transcripts_dir = PROJECT_HOME / "data" / "processed" / "transcripts"
    knowledge_base_dir = PROJECT_HOME / "data" / "knowledge_base"

    # Ensure all required directories exist
    cleaned_audio_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["arabic", "english"]:
        os.makedirs(os.path.join(transcripts_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(knowledge_base_dir, subdir), exist_ok=True)

    # Initialize the Gemini client for transcription
    client = GeminiClient()

    # Processing counters
    audio_processed = 0
    audio_skipped = 0
    transcript_processed = 0
    transcript_skipped = 0

    # PHASE 1: Clean the audio files
    print("\n=== PHASE 1: AUDIO CLEANING ===")
    for audio_file in raw_folder.glob("*.wav"):
        output_file = cleaned_audio_dir / audio_file.name

        if process_audio(str(audio_file), str(output_file)):
            audio_processed += 1
        else:
            audio_skipped += 1

    print(
        f"\nAudio cleaning complete: {audio_processed} files processed, {audio_skipped} files skipped"
    )

    # PHASE 2: Process the cleaned audio files to generate transcripts
    print("\n=== PHASE 2: TRANSCRIPT GENERATION ===")
    for audio_file in cleaned_audio_dir.glob("*.wav"):
        if process_transcript(
            str(audio_file), transcripts_dir, knowledge_base_dir, client
        ):
            transcript_processed += 1
        else:
            transcript_skipped += 1

    print(
        f"\nTranscript generation complete: {transcript_processed} files processed, {transcript_skipped} files skipped"
    )
    if transcript_processed > 0:
        from src.rag.chroma_utils import create_or_update_vectorstore

        create_or_update_vectorstore()

    print("\nFull pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
