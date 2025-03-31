import streamlit as st
import random
import time
from app.model_manager import ModelManager

st.title("NU OGS Chatbot")

# Cache the model manager so that the model is loaded only once.
@st.cache_resource
def get_model_manager():
    return ModelManager()

mm = get_model_manager()

# Streamed response emulator
def response_generator(query):
    response = mm.query_handler(query)
    if isinstance(response, str):
        response = response
    else:
        response = response['output']
    for word in response.split():
        yield word + " "
        time.sleep(0.02)
    if isinstance(response, dict):
        yield "See the following links:\n" + "\n".join(response['urls'])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask anything OGS-Related"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})