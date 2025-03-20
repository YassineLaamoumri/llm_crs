import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Add the project root to the Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from src.rag.pydantic_models import QueryInput, QueryAnswer
from src.rag.langchain_utils import get_rag_chain
from src.rag.db_utils import insert_application_log, get_chat_history
import os
import uuid
import logging

logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI()


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
