import streamlit as st


def display_sidebar():
    with st.sidebar:
        st.subheader("Settings")

        # Model selection
        model_options = ["gpt-4o-mini", "gemini-2.0-flash-thinking-exp-01-21"]
        selected_model = st.selectbox(
            "Select Model", options=model_options, index=1, key="model"
        )

        # Store selected model in session state
        if "model" not in st.session_state:
            st.session_state.model = selected_model

        # Initialize messages list if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.success("Chat history cleared.")

        # Display messages count instead of the actual messages
        if st.session_state.messages:
            st.text(f"Messages: {len(st.session_state.messages)}")
