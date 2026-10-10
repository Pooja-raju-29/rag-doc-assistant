import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)
def generate_answer(query, retrieved_chunks):

    # Combine retrieved chunks into one context
    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    # Create prompt for Gemini
    prompt = f"""
    Answer the question using only the context provided below.
    Context:
    {context}
    Question:
    {query}
    Answer:
    """
    # Send prompt to Gemini
    response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=prompt
)
    return response.text

def get_sources(retrieved_chunks):

    sources = []

    for chunk in retrieved_chunks:
        source = f'{chunk["source"]} - Page {chunk["page"]}'

        if source not in sources:
            sources.append(source)

    return sources