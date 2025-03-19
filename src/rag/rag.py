# %%

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# %%
# Set environment variables for LangSmith (if using)
os.environ["LANGCHAIN_TRACING"] = LANGSMITH_TRACING
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
# %%
# Load documents from directory
loader = DirectoryLoader(
    "data/knowledge_base/english",
    glob="*.txt",
    loader_cls=TextLoader,
    use_multithreading=True,
)

documents = loader.load()


# %%
# Attach a unique document ID using regex from filename
def attach_id(doc):
    filename = os.path.basename(doc.metadata["source"])
    match = re.match(r"(\d+)", filename)
    doc_id = (
        match.group(1) if match else os.path.basename(doc.metadata["source"])
    )  # Use filename as fallback
    doc.metadata["doc_id"] = doc_id
    return doc


documents = [attach_id(doc) for doc in documents]

# Initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
    # If needed, uncomment to specify dimensions
    # dimensions=1024
)

# Path for persistent storage
persist_directory = "chroma_db"

# Check if the collection already exists to avoid duplicates
try:
    # Initialize Chroma with persistence
    vectorstore = Chroma(
        collection_name="calls",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Try to get existing IDs (this approach is more reliable)
    # We'll query with a dummy vector to get collection info
    try:
        collection_data = vectorstore._collection.count()
        # If we get here, the collection exists but might be empty
        print(f"Found existing collection with {collection_data} documents.")
        collection_exists = True
    except Exception as e:
        print(f"Error checking collection: {e}")
        collection_exists = False

except Exception as e:
    print(f"Error loading vector store: {e}")
    collection_exists = False
    vectorstore = None

# %%
# Get document IDs from our documents
document_ids = [doc.metadata["doc_id"] for doc in documents]
# %%
# If the collection doesn't exist or is empty, create it with all documents
if not collection_exists or vectorstore is None:
    print("Creating new Chroma collection with all documents...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="calls",
        persist_directory=persist_directory,
        ids=document_ids,
    )
    print(f"Added {len(documents)} documents to new vector store.")
else:
    # For an existing collection, we want to avoid duplicates
    # Get existing IDs by querying the collection
    try:
        # Get all existing IDs
        # This is a dummy query that returns no results but gives us collection metadata
        existing_ids = set(vectorstore._collection.get()["ids"])

        # Filter out documents that are already in the store
        new_docs = [
            doc for doc in documents if doc.metadata["doc_id"] not in existing_ids
        ]
        new_ids = [doc.metadata["doc_id"] for doc in new_docs]

        if new_docs:
            print(
                f"Adding {len(new_docs)} new documents to the existing vector store..."
            )
            vectorstore.add_documents(documents=new_docs, ids=new_ids)
            print(f"Successfully added {len(new_docs)} new documents.")
        else:
            print("No new documents to add.")
    except Exception as e:
        print(f"Error adding documents: {e}")
        # If we encounter problems with incremental updates, consider recreating the collection
        print("Recreating the collection with all documents...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="calls",
            persist_directory=persist_directory,
            ids=document_ids,
        )


# %%

query = "I sant a special offer?"
docs = vectorstore.similarity_search(query, k=3)
for doc in docs:
    print(f"Document ID: {doc.metadata['doc_id']}")
    print(f"Content: {doc.page_content}")
    print("-" * 100)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever.invoke(query)
# %%
from langchain_core.prompts import ChatPromptTemplate

template = """Answer the questions with the help of the provided context. This context represents the history of some relevants calls.
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)  # Create a prompt template

from langchain.schema.runnable import RunnablePassthrough


def docs2str(docs):
    return "\n\n".join(
        [
            f"Document ID: {doc.metadata['doc_id']}\nContent: {doc.page_content}"
            for doc in docs
        ]
    )


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    temperature=0.7,
    max_tokens=65000,
    timeout=None,
    max_retries=2,
    top_p=0.9,
    top_k=64,
)


rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(query)


chat_history = []

chat_history.extend([HumanMessage(content=query), AIMessage(content=answer)])

from langchain_core.prompts import MessagesPlaceholder

contextualize_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history"
    "formulate a standalone question which can be understood"
    "without the chat history.Do NOT answer the question"
    "just reformulate it if necessary and otherwise return it as it is."
)


contextualize_q_prompt = ChatPromptTemplate.from_message(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history")("human", "{input}"),
    ]
)


from langchain.chains import create_history_awara_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

history_aware_retriever = create_history_awara_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a AI assistant from an ecommerce company (Taager).Answer the questions with the help of the provided context. This context represents the history of some relevant calls.",
        ),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "input"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db"


def get_db_connexion():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_application_logs():
    conn = get_db_connexion()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS application_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_query TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def insert_application_log(session_id, user_query, ai_response, model):
    conn = get_db_connexion()
    conn.execute(
        """
        INSERT INTO application_logs(session_id, user_query, ai_response, model)
        VALUES(?,?,?,?)
        """,
        (session_id, user_query, ai_response, model),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    conn = get_db_connexion()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_query, ai_response FROM application_logs WHERE session_id = ? ORDER BY created_at DESC
        """,
    )
    messages = []
    for row in cursor.fetchall():
        messages.extend(
            [
                HumanMessage(content=row["user_query"]),
                AIMessage(content=row["ai_response"]),
            ]
        )
    conn.close()
    return messages


create_application_logs()
