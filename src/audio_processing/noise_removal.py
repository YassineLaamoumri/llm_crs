from pedalboard.io import AudioFile
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
import soundfile as sf
from pathlib import Path
import numpy as np
import os
import hashlib


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


def generate_file_hash(file_path):
    """
    Generate a hash for a file to use as a processing identifier.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        str: Hash string representing the file content.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_already_processed(input_file, output_file, hash_dir):
    """
    Check if an audio file has already been processed by comparing hash values.

    Parameters:
        input_file (Path): Path to the input audio file.
        output_file (Path): Path to the potential output audio file.
        hash_dir (Path): Directory to store hash files.

    Returns:
        bool: True if already processed, False otherwise.
    """
    # If output file doesn't exist, it hasn't been processed
    if not output_file.exists():
        return False

    # Generate input file hash
    input_hash = generate_file_hash(input_file)
    hash_file = hash_dir / f"{input_file.stem}.hash"

    # If hash file exists, check if hash matches
    if hash_file.exists():
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
        return stored_hash == input_hash

    return False


def save_hash(input_file, hash_dir):
    """
    Save the hash of the input file for future reference.

    Parameters:
        input_file (Path): Path to the input audio file.
        hash_dir (Path): Directory to store hash files.
    """
    input_hash = generate_file_hash(input_file)
    hash_file = hash_dir / f"{input_file.stem}.hash"

    with open(hash_file, "w") as f:
        f.write(input_hash)


def process_audio(input_audio_path, output_audio_path, sr=16000):
    """
    Process an audio file by reducing noise and applying audio effects using Pedalboard.

    Parameters:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str): Path to save the processed audio file.
        sr (int): Target sampling rate.
    """
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
    sf.write(output_audio_path, effected_audio, sr, format="WAV")
    print(f"Processed audio saved to: {output_audio_path}")


if __name__ == "__main__":
    # Retrieve the project home dynamically using the marker file (e.g., README.md)
    PROJECT_HOME = find_project_root()

    # Define input and output directories
    raw_folder = PROJECT_HOME / "data" / "raw"
    output_folder = PROJECT_HOME / "data" / "processed" / "cleaned_audio"
    hash_folder = PROJECT_HOME / "data" / "processed" / "hashes"

    # Create the output and hash folders if they don't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    hash_folder.mkdir(parents=True, exist_ok=True)

    # Loop through all WAV files in the raw folder
    for audio_file in raw_folder.glob("*.wav"):
        output_file = output_folder / audio_file.name

        # Check if the file has already been processed
        if is_already_processed(audio_file, output_file, hash_folder):
            print(f"Skipping {audio_file.name} - already processed")
            continue

        # Process the audio file
        process_audio(str(audio_file), str(output_file))

        # Save the hash of the processed file
        save_hash(audio_file, hash_folder)
