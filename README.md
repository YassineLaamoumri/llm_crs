# Customer Support RAG System

A Retrieval-Augmented Generation (RAG) system for processing customer support calls in an e-commerce environment. This system extracts structured information from audio call transcriptions using Gemini AI to improve customer support knowledge retrieval.

## Overview

This project processes audio recordings from customer support calls through the following pipeline:
1. Audio noise reduction of call center recordings
2. Transcription of the cleaned audio
3. Extraction of structured information using Gemini AI
4. Storage in a vector database for efficient retrieval
5. Retrieval-Augmented Generation for answering customer support queries

## Tech Stack

- **FastAPI**: Backend API services for processing and serving data
- **Streamlit**: User-friendly frontend interface
- **Gemini AI**: For extracting structured information from transcriptions
- **Docker**: Containerization for easy deployment and scaling
- **Docker Compose**: Orchestration of multiple containerized services

## Getting Started

### Prerequisites

- Docker and Docker Compose installed on your system
- Git for cloning the repository
- Create OPENAI and GOOGLE GEMINI API KEY

### Installation

1. Clone the repository:

git clone https://github.com/YassineLaamoumri/llm_crs.git

2. Build and start the project:

sudo docker compose up --build

3. Enter your API KEY for OPENAI and GOOGLE GEMINI into the .env file (you can also add your Langsmith API KEY but it's optional since I kept mine to see how you interact with my RAG)

3. Access the Streamlit application:
Open your browser and navigate to `http://localhost:8501`



## Usage

### Processing New Audio Files

1. Upload audio files through the Streamlit interface
2. The system will automatically:
   - Clean the audio
   - Transcribe the content
   - Extract structured information
   - Index the information for future retrieval

### Querying the System

1. Enter your question in the Streamlit interface
2. The system will:
   - Retrieve relevant information from the knowledge base
   - Generate an accurate response based on the retrieved context
   - Display both the response and supporting information
