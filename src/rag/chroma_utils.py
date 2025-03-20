# src/rag/chroma_utils.py
import os
import re
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# External imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader


# Setup project path and imports
def setup_project_path():
    """Set up the project path and add it to sys.path."""
    # Get current file's directory
    current_dir = Path(os.getcwd())

    # Navigate up to the project root
    project_root = current_dir
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
        logger.info(f"Added {project_root} to Python path")

    return project_root


# Initialize environment and configuration
project_root = setup_project_path()

try:
    from fastapi_app.config import (
        GEMINI_API_KEY,
        OPENAI_API_KEY,
        LANGSMITH_API_KEY,
        LANGSMITH_ENDPOINT,
        LANGSMITH_PROJECT,
        LANGSMITH_TRACING,
    )

    # Set environment variables
    os.environ["LANGCHAIN_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

except ImportError as e:
    logger.error(f"Failed to import configuration: {e}")
    raise

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# Create persist directory path
persist_directory = f"{str(project_root)}/chroma_db"


def attach_document_id(doc):
    """Attach a unique document ID to a document based on its filename."""
    filename = os.path.basename(doc.metadata["source"])
    match = re.match(r"(\d+)", filename)
    doc_id = match.group(1) if match else filename  # Use filename as fallback
    doc.metadata["doc_id"] = doc_id
    return doc


def load_documents():
    """Load text documents from the knowledge base directory."""
    try:
        # Load documents from directory
        loader = DirectoryLoader(
            project_root / "data/knowledge_base/english",
            glob="*.txt",
            loader_cls=TextLoader,
            use_multithreading=True,
        )

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from knowledge base")

        # Attach a unique document ID using regex from filename
        documents = [attach_document_id(doc) for doc in documents]

        return documents

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise


def get_existing_vectorstore():
    """Try to get an existing vector store."""
    try:
        vectorstore = Chroma(
            collection_name="calls",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        # Check if collection exists and get document count
        collection_count = vectorstore._collection.count()
        logger.info(f"Found existing collection with {collection_count} documents")
        return vectorstore, True

    except Exception as e:
        logger.info(f"No existing vector store found or error loading it: {e}")
        return None, False


def create_or_update_vectorstore():
    """Create or update the vector store with documents."""
    # Load documents
    documents = load_documents()
    document_ids = [doc.metadata["doc_id"] for doc in documents]

    # Check if vector store exists
    vectorstore, exists = get_existing_vectorstore()

    if not exists or vectorstore is None:
        # Create new vector store
        logger.info("Creating new Chroma collection with all documents")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="calls",
            persist_directory=persist_directory,
            ids=document_ids,
        )
        logger.info(f"Created new vector store with {len(documents)} documents")
    else:
        try:
            # Get existing IDs
            existing_ids = set(vectorstore._collection.get()["ids"])

            # Filter out documents that are already in the store
            new_docs = [
                doc for doc in documents if doc.metadata["doc_id"] not in existing_ids
            ]
            new_ids = [doc.metadata["doc_id"] for doc in new_docs]

            if new_docs:
                logger.info(
                    f"Adding {len(new_docs)} new documents to existing vector store"
                )
                vectorstore.add_documents(documents=new_docs, ids=new_ids)
                logger.info(f"Successfully added {len(new_docs)} new documents")
            else:
                logger.info("No new documents to add")
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
            # If updating fails, recreate the collection
            logger.warning(
                "Recreating the collection with all documents due to update failure"
            )
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name="calls",
                persist_directory=persist_directory,
                ids=document_ids,
            )

    return vectorstore


# Initialize the vector store at module level
try:
    # First try to get existing store
    vectorstore, exists = get_existing_vectorstore()

    # If it doesn't exist, create it
    if not exists or vectorstore is None:
        vectorstore = create_or_update_vectorstore()

    logger.info("Vector store initialized and ready to use")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    # Provide a fallback or raise exception based on your requirements
    raise

# If this script is run directly, execute the create/update function
if __name__ == "__main__":
    vectorstore = create_or_update_vectorstore()
