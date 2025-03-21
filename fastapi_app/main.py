import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Add the project root to the Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from src.rag.pydantic_models import QueryInput, QueryAnswer
from src.rag.langchain_utils import get_rag_chain
from src.rag.db_utils import insert_application_log, get_chat_history
from src.audio_processing.transcribe_gemini import main as process_audio
import os
import uuid
import logging
import uvicorn
import threading

logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI()


def run_audio_processing():
    """Run the audio processing in a separate thread"""
    try:
        logging.info("Starting audio processing script...")
        process_audio()
        logging.info("Audio processing completed successfully")
    except Exception as e:
        logging.error(f"Error running audio processing script: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Log startup and schedule the processing to run after startup completes"""
    logging.info("Application startup complete")
    # Schedule the audio processing to run after startup
    threading.Thread(target=run_audio_processing).start()


@app.post("/ask", response_model=QueryAnswer)
def ask(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(
        f"Query from session {session_id}: {query_input.question}, Model : {query_input.model.value}"
    )
    if not session_id:
        session_id = str(uuid.uuid4())

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke(
        {"input": query_input.question, "chat_history": chat_history}
    )["answer"]

    insert_application_log(
        session_id, query_input.question, answer, query_input.model.value
    )
    logging.info(f"Response to session {session_id}: {answer}")
    return QueryAnswer(answer=answer, session_id=session_id, model=query_input.model)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
