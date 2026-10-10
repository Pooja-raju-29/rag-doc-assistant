from pathlib import Path
import pickle
import faiss

from doc_loader import load_documents
from chunker import create_chunks
from embedder import create_embeddings
from vector_store import create_vector_store


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    storage_dir = project_root / "storage"

    storage_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PDF documents
    documents = load_documents()
    print(f"Loaded {len(documents)} document pages")

    # Step 2: Split documents into chunks
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError("No text chunks found. Add PDF files to data/.")

    # Step 3: Convert chunks into embeddings
    embeddings = create_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    # Step 4: Build FAISS vector index
    index = create_vector_store(embeddings)

    # Step 5: Save the index and chunk metadata
    faiss.write_index(index, str(storage_dir / "faiss_index.bin"))

    with open(storage_dir / "chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

    print("FAISS index and chunks saved successfully!")


if __name__ == "__main__":
    main()