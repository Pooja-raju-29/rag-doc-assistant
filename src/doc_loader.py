import pymupdf
from pathlib import Path


def load_documents():

    data_folder = Path("data")
    documents = []

    for pdf_path in data_folder.glob("*.pdf"):

        print(f"Loading: {pdf_path.name}")

        document = pymupdf.open(pdf_path)

        for page_number, page in enumerate(document):

            text = page.get_text()

            if not text.strip():
                continue

            page_data = {
                "text": text,
                "source": pdf_path.name,
                "page": page_number + 1
            }

            documents.append(page_data)

        document.close()

    print("DOCUMENT LOADING COMPLETE")
    print("Total pages loaded:", len(documents))

    return documents