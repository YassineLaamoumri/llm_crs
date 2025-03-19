import streamlit as st

st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")


if "message_history" not in st.session_state:
    st.session_state.message_history = [
        {
            "content": "Hello, I'm a chatbot assistant from Taager, How can I help you today?",
            "type": "assistant",
        }
    ]


left_column, main_column, right_column = st.columns([1, 2, 1])

with left_column:
    if st.button("Clear Chat"):
        st.session_state.message_history = [""]


with main_column:
    user_input = st.chat_input("Type here ...", key="user_input")
    if user_input:
        st.session_state.message_history.append({"content": user_input, "type": "user"})

    for i in range(1, len(st.session_state.message_history) + 1):
        this_message = st.session_state.message_history[-i]
        message_box = st.chat_message(this_message["type"])
        message_box.markdown(this_message["content"])

with right_column:
    st.text(st.session_state.message_history)
