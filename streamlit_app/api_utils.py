import requests
import streamlit as st


def get_api_response(question, session_id, model):
    headers = {
        "Content-Type": "application",
        "Accept": "application/json",
    }

    data = {
        "question": question,
        "model": model,
    }
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(
            "http://localhost:8000/ask", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API call failed with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed with error: {e}")
        return None
