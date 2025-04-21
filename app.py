import streamlit as st
import time
import logging
import bleach
from app.model_manager import ModelManager

# logger
logger = logging.getLogger("ogs_chatbot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)

# guard rails
MAX_PROMPT_LENGTH = 1000
MAX_QUERIES_PER_SESSION = 50
ALLOWED_URL_SCHEMES = {"http", "https"}
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

st.title("NU OGS Chatbot")

# Cache the model manager so that the model is loaded only once.
@st.cache_resource
def get_model_manager():
    try:
        return ModelManager()  # prep for bad credentials/network
    except Exception as e:
        logger.exception("Failed to initialize ModelManager")
        st.error("Internal error loading the chatbot. Please try again later.")
        st.stop()

mm = get_model_manager()

# helper functions
def sanitize_output(text: str) -> str:
    # allow only simple tags; strip anything else
    return bleach.clean(text, tags=["b","i","strong","em","br"], strip=True)

# Streamed response emulator
def response_generator(query: str):
    try:
        resp = mm.query_handler(query)
    except Exception as e:
        logger.exception("Error in query_handler")
        yield "\n\n**Error:** Could not get a response from the model."
        return

    output = sanitize_output(resp.get("output", ""))
    # type‑writer effect
    for word in output.split():
        yield word + " "
        time.sleep(0.02)

    urls = resp.get("urls", [])
    # After output, yield the URLs with HTML line breaks
    if urls:
        links_html = "<br>".join(
            f'<a href="{u}" target="_blank">{u}</a>' for u in urls
        )
        yield "<br><br>See the following links for more:<br>" + links_html


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask anything OGS‑Related"):
    # Rate limit
    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error("You’ve reached the maximum number of queries for this session.")
        st.stop()

    # Input validation
    if len(prompt) > MAX_PROMPT_LENGTH:
        st.error(f"Prompt too long (max {MAX_PROMPT_LENGTH} characters).")
        st.stop()

    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(bleach.clean(prompt), unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated = ""
        for token in response_generator(prompt):
            accumulated += token
            placeholder.markdown(f"<div>{accumulated}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": accumulated})