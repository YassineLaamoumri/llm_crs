import sqlite3
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage


# Use environment variable with fallback
data_dir = os.environ.get("DATA_DIR", ".")
DB_NAME = os.path.join(data_dir, "rag_app.db")


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
        "SELECT user_query, ai_response FROM application_logs WHERE session_id = ? ORDER BY created_at DESC",
        (session_id,),
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
