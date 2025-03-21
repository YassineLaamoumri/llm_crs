import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

# Add project root to sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Import required libraries
import langchain
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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks import CallbackManager, StdOutCallbackHandler

from src.rag.chroma_utils import create_or_update_vectorstore

# Set environment variables
os.environ["LANGCHAIN_TRACING"] = LANGSMITH_TRACING
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Model configurations
MODEL_CONFIGS = {
    "gemini-2.0-flash-thinking-exp-01-21": {
        "model": "gemini-2.0-flash-thinking-exp-01-21",
        "temperature": 0.7,
        "max_tokens": 65000,
        "timeout": None,
        "max_retries": 2,
        "top_p": 0.9,
        "top_k": 64,
    },
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 16384,
        "timeout": None,
        "max_retries": 2,
    },
    "gpt-4o": {
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 16384,
        "timeout": None,
        "max_retries": 2,
    },
}

# Initialize vectorstore with caching
_vectorstore_cache = {}


def get_vectorstore(force_refresh: bool = False) -> Chroma:
    """Get or create vectorstore with caching option"""
    global _vectorstore_cache
    if force_refresh or not _vectorstore_cache:
        logger.info("Creating or updating vectorstore")
        _vectorstore_cache["vs"] = create_or_update_vectorstore()
    return _vectorstore_cache["vs"]


# Enhanced prompt templates
CONTEXTUALIZE_SYSTEM_PROMPT = """
Given a chat history and the latest user question which might reference context in the chat history,
formulate a standalone question that can be understood without the chat history.

Guidelines:
- Preserve all specific details from the original question
- Resolve pronouns and references to earlier conversation
- Do NOT answer the question
- If the question is already standalone, return it unchanged
- Return only the reformulated question with no additional text
"""

QA_SYSTEM_PROMPT = """
You are an AI assistant for Taager, an ecommerce company. Your goal is to provide accurate, helpful responses to customer inquiries using the provided context.

Guidelines:
- Use ONLY the provided context to answer questions
- The context contains relevant call histories that will help you understand common customer issues
- If the answer isn't in the context, acknowledge this and suggest what information might help
- Be concise but thorough
- Maintain a professional and supportive tone
- Format your responses for clarity when appropriate

Remember that your responses directly impact customer satisfaction and brand reputation.
"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def create_callback_manager(verbose: bool = False) -> Optional[CallbackManager]:
    """Create callback manager for tracing and debugging"""
    if verbose:
        return CallbackManager([StdOutCallbackHandler()])
    return None


def get_llm(model_name: str, verbose: bool = False):
    """Factory function to create LLM instances with proper configuration"""
    if model_name not in MODEL_CONFIGS:
        logger.warning(
            f"Model {model_name} not found in configurations, defaulting to gemini-2.0-flash-thinking-exp-01-21"
        )
        model_name = "gemini-2.0-flash-thinking-exp-01-21"

    config = MODEL_CONFIGS[model_name]
    callback_manager = create_callback_manager(verbose)

    if "gemini" in model_name:
        return ChatGoogleGenerativeAI(**config, callback_manager=callback_manager)
    else:
        return ChatOpenAI(**config, callback_manager=callback_manager)


def get_retriever(k: int = 3, force_refresh: bool = False):
    """Get document retriever with configurable parameters"""
    vectorstore = get_vectorstore(force_refresh)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_rag_chain(
    model: str = "gemini-2.0-flash-thinking-exp-01-21",
    retriever_k: int = 3,
    verbose: bool = False,
    force_refresh_vectorstore: bool = False,
    maintain_dict_output: bool = True,  # Added parameter to control output format
):
    """
    Create an enhanced RAG chain with configurable parameters

    Args:
        model: Model name to use
        retriever_k: Number of documents to retrieve
        verbose: Whether to enable verbose logging
        force_refresh_vectorstore: Whether to force refresh the vectorstore
        maintain_dict_output: If True, returns {"answer": response} to maintain backward compatibility

    Returns:
        A runnable RAG chain that returns either a dict with "answer" key or a string
    """
    # Get LLM
    llm = get_llm(model, verbose)

    # Get retriever
    retriever = get_retriever(retriever_k, force_refresh_vectorstore)

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    # Create QA chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    if maintain_dict_output:
        # Keep the original output format for backward compatibility
        return rag_chain
    else:
        # Add post-processing step for improved formatting and return string directly
        def format_response(response: Dict[str, Any]) -> str:
            """Format the RAG response for better readability"""
            return response["answer"].strip()

        return rag_chain | RunnableLambda(format_response)
