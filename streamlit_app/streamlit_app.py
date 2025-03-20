import streamlit as st

from chat_interface import display_chat_interface
from sidebar import display_sidebar

st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")

st.title("Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "content": "Hello, I'm a chatbot assistant from Taager, How can I help you today?",
            "role": "assistant",
        }
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = None


display_chat_interface()


display_sidebar()
