#####PDF-reading functionality
import pymupdf
from pathlib import Path
from chunker import create_chunks
from embedder import create_embeddings
##from vector_store import create_vector_store
from vector_store import create_vector_store, save_vector_store, save_chunks
from retriever import retrieve
from embedder import model
from llm import generate_answer, get_sources
# Path to the folder containing our research papers
data_folder = Path("data")
# This list will store all extracted pages
documents = []
# Find every PDF inside the data folder
for pdf_path in data_folder.glob("*.pdf"):
    print(f"Loading: {pdf_path.name}")
    # Open the PDF
    document = pymupdf.open(pdf_path)
    # Go through every page
    for page_number, page in enumerate(document):
        # Extract text from the page
        text = page.get_text()
        # Skip pages that contain no useful text
        if not text.strip():
            continue
        # Store text + metadata
        page_data = {
            "text": text,
            "source": pdf_path.name,
            "page": page_number + 1
        }
        documents.append(page_data)
    # Close the PDF
    document.close()
print("\n-----------------------------")
print("DOCUMENT LOADING COMPLETE")
print("-----------------------------")
print("Total pages loaded:", len(documents))
# Show one example
if documents:
    print("\nExample document:")
    print("Source:", documents[0]["source"])
    print("Page:", documents[0]["page"])
    print("Text:")
    print(documents[0]["text"][:500])
####chunks function calling 
chunks = create_chunks(documents)
print("\nTotal chunks created:", len(chunks))
print("\nExample chunk:")
print("Source:", chunks[0]["source"])
print("Page:", chunks[0]["page"])
print("Text:")
print(chunks[0]["text"])
###embedd
embeddings = create_embeddings(chunks)
print("\nEMBEDDINGS CREATED")
print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])
####vector
embeddings = create_embeddings(chunks)
index = create_vector_store(embeddings)
# Save FAISS index to disk
save_vector_store(index)
# Save original chunks and metadata
save_chunks(chunks)
print("FAISS index and chunks saved successfully!")
print("\nVECTOR STORE CREATED")
print("Number of vectors stored:", index.ntotal)
####retriever###
####query = "What is self-attention?"
query = "What is retrieval-augmented generation?"

results = retrieve(
    query,
    model,
    index,
    chunks,
    top_k=5
)

print("\nQUESTION:")
print(query)

print("\nRETRIEVED CHUNKS:")

for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Text:", result["text"])
####llm
###query = "What is self-attention?"
query = "What is retrieval-augmented generation?"
####Generate final answer using Gemini
answer = generate_answer(query, results)
print("\nFINAL RAG ANSWER:")
print(answer)
#####Get sources from retrieved chunks
sources = get_sources(results)
print("\nSOURCES:")
for source in sources:
    print(source)