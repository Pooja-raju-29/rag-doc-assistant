def load_tokenizer():
    # Imported here so the chunker can be tested without loading the model
    from embedder import model

    return model.tokenizer


def count_tokens(text, tokenizer):
    # Includes the special tokens the embedding model adds to every text
    return len(tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"])


def create_chunks(documents, chunk_size=224, overlap=32, tokenizer=None):
    if not documents:
        return []

    if tokenizer is None:
        tokenizer = load_tokenizer()

    # chunk_size includes the special tokens, so leave room for them
    content_size = chunk_size - tokenizer.num_special_tokens_to_add(pair=False)

    if not 0 <= overlap < content_size:
        raise ValueError("overlap must be smaller than the tokens available in a chunk")

    chunks = []
    for document in documents:
        text = " ".join(document["text"].split())

        # Offsets give each token's position, so chunks keep the original text.
        # Decoding the tokens would lowercase it and change the spacing.
        offsets = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False
        )["offset_mapping"]

        start = 0
        while start < len(offsets):
            end = min(start + content_size, len(offsets))
            chunk_text = text[offsets[start][0]:offsets[end - 1][1]]

            # A chunk that starts mid-word can gain tokens when tokenized again
            while count_tokens(chunk_text, tokenizer) > chunk_size and end - start > overlap + 1:
                end -= 1
                chunk_text = text[offsets[start][0]:offsets[end - 1][1]]

            chunk_data = {
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"]
            }
            chunks.append(chunk_data)

            if end == len(offsets):
                break

            start = end - overlap
    return chunks
