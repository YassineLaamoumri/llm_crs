import os
import requests
import streamlit as st


def get_api_response(question, session_id, model):
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    data = {
        "question": question,
        "model": model,
    }
    if session_id:
        data["session_id"] = session_id

    # Get the FastAPI URL from environment variable or use default
    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")

    try:
        response = requests.post(f"{fastapi_url}/ask", headers=headers, json=data)
        if response.status_code == 200:
            print(response.json())
            return response.json()
        else:
            st.error(f"API call failed with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed with error: {e}")
        return None
