
import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Research Assistant")
st.caption("Ask questions about AI, LLMs, Transformers, and RAG research papers.")

# Store conversation messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question
if question := st.chat_input("Ask a question about your research papers..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching research papers..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )
                response.raise_for_status()

                result = response.json()
                answer = result.get("answer", "No answer available.")
                sources = result.get("sources", [])

                content = answer

                if sources:
                    content += "\n\n**Sources:**\n"
                    for source in sources:
                        content += f"\n- {source}"

                st.markdown(content)

            except requests.RequestException as error:
                content = (
                    "Could not connect to the RAG API. "
                    "Make sure FastAPI is running."
                )
                st.error(content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": content
    })
