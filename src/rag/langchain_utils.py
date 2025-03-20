import sys
import os
from pathlib import Path

# Get current file's directory
current_dir = Path(os.getcwd())

# Navigate up to the project root (3 levels up from src/rag)
project_root = current_dir
# Keep going up in the directory hierarchy until we reach the project root
while not (project_root / "fastapi_app").exists():
    project_root = project_root.parent
    # Safety check to avoid infinite loop
    if project_root == project_root.parent:
        raise FileNotFoundError(
            "Could not find project root with fastapi_app directory"
        )

# Add project root to sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")


import langchain
import os
import re
from fastapi_app.config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.rag.chroma_utils import create_or_update_vectorstore

# Set environment variables
os.environ["LANGCHAIN_TRACING"] = LANGSMITH_TRACING
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

vectorstore = create_or_update_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

output_parser = StrOutputParser()


contextualize_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history"
    "formulate a standalone question which can be understood"
    "without the chat history.Do NOT answer the question"
    "just reformulate it if necessary and otherwise return it as it is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a AI assistant from an ecommerce company (Taager).Answer the questions with the help of the provided context. This context represents the history of some relevant calls. Learn from this experience to be the best customer agent. It only help you for customer support.",
        ),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "input"),
    ]
)


def get_rag_chain(model="gemini-2.0-flash-thinking-exp-01-21"):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0.7,
        max_tokens=65000,
        timeout=None,
        max_retries=2,
        top_p=0.9,
        top_k=64,
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain
