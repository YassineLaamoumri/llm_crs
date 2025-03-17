import os
import json
from src.models.gemini_api import GeminiClient


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


def process_audio_file(audio_path, transcripts_dir, knowledge_base_dir, client):
    """
    Processes an individual audio file:
    - Generates the transcript via the GeminiClient.
    - Extracts and loads the JSON data.
    - Separates Arabic and English data.
    - Writes both the transcript and knowledge base entries to the appropriate file,
      using a modular method that supports various content types.
    """
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

        # Determine the output filename (change extension to .txt)
        audio_filename = os.path.basename(audio_path)
        base_filename = os.path.splitext(audio_filename)[0] + ".txt"

        # Define file paths
        transcript_path_arabic = os.path.join(transcripts_dir, "arabic", base_filename)
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

        print(f"Processed '{audio_filename}' to '{base_filename}'")
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")


def main():
    # Define the paths for cleaned audio, transcripts, and knowledge base
    cleaned_audio_dir = "data/processed/cleaned_audio"
    transcripts_dir = "data/processed/transcripts"
    knowledge_base_dir = "data/knowledge_base"

    # Ensure language-specific subdirectories exist
    for subdir in ["arabic", "english"]:
        os.makedirs(os.path.join(transcripts_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(knowledge_base_dir, subdir), exist_ok=True)

    # Initialize the Gemini client
    client = GeminiClient()

    # Process each audio file in the cleaned audio directory
    for audio_filename in os.listdir(cleaned_audio_dir):
        audio_path = os.path.join(cleaned_audio_dir, audio_filename)
        process_audio_file(audio_path, transcripts_dir, knowledge_base_dir, client)


if __name__ == "__main__":
    main()
