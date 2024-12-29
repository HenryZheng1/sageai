import os
import uuid
from dotenv import load_dotenv

import fitz  # PyMuPDF
import tiktoken

# Concurrency imports
from concurrent.futures import ThreadPoolExecutor, as_completed

# Official OpenAI library (used inside MyOpenAIClient)
import openai

# Your custom Pinecone client
# Make sure `client.py` defines PineconeClient with the required methods.
# For example: PineconeClient(api_key, index_name) => .index for upserts
from client import PineconeClient


# -------------------------------------------------------------------------------
# 1. Custom OpenAI Client Wrapper
# -------------------------------------------------------------------------------
class MyOpenAIClient:
    """
    Simple wrapper to replicate the idea of "client = OpenAI()"
    but using the official openai library under the hood.
    """
    def __init__(self, api_key: str):
        openai.api_key = api_key  # sets global var in openai library
        self.api_key = api_key

    def create_embedding(self, text_input: str, model_name: str) -> list[float]:
        """
        Create an embedding using the official openai.Embedding.create method.
        """
        response = openai.Embedding.create(
            model=model_name,
            input=text_input
        )
        return response["data"][0]["embedding"]


# -------------------------------------------------------------------------------
# 2. Environment & Config
# -------------------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = "baserag"

# Example of an OpenAI embedding model + dimension
MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072  # Example dimension for illustration

# Initialize custom OpenAI client
openai_client = MyOpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone client
pc = PineconeClient(
    api_key=PINECONE_API_KEY,
    index_name=PINECONE_INDEX_HOST
)
index = pc.index  # Pinecone index object


# -------------------------------------------------------------------------------
# 3. PDF Reading and Tokenization
# -------------------------------------------------------------------------------
def read_pdf_text(pdf_path: str) -> str:
    """
    Reads all text from a PDF file using PyMuPDF (fitz)
    and returns the concatenated text.
    """
    doc = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        all_text.append(text)
    doc.close()
    return "\n".join(all_text)


def tokenize_text(text: str, encoding_name: str) -> list[int]:
    """
    Tokenizes text using a tiktoken encoding and returns a list of token IDs.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)


def detokenize_tokens(tokens: list[int], encoding_name: str) -> str:
    """
    Detokenizes a list of token IDs back into a string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.decode(tokens)


def chunk_tokens(token_ids: list[int], chunk_size: int = 1000, overlap: int = 250) -> list[list[int]]:
    """
    Chunks the list of token IDs into segments of chunk_size with overlap tokens.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than the chunk size.")

    chunks = []
    start_index = 0
    while start_index < len(token_ids):
        end_index = start_index + chunk_size
        chunk = token_ids[start_index:end_index]
        chunks.append(chunk)
        # Move start index by (chunk_size - overlap) to maintain overlap
        start_index += (chunk_size - overlap)

        if start_index >= len(token_ids):
            break

    return chunks


# -------------------------------------------------------------------------------
# 4. Embedding & Upserting to Pinecone
# -------------------------------------------------------------------------------
def embed_and_upsert(chunks: list[str], model_name: str = MODEL_NAME):
    """
    Takes a list of text chunks, gets embeddings from OpenAI (via custom client),
    and upserts them to the Pinecone index. Uses ThreadPoolExecutor for concurrency.
    """

    def process_chunk(chunk_text: str):
        # 1) Get embedding from custom openai client
        embedding = openai_client.create_embedding(text_input=chunk_text, model_name=model_name)

        # 2) Create metadata
        metadata = {"text": chunk_text}

        # 3) Generate a unique ID for Pinecone
        vector_id = str(uuid.uuid4())

        # 4) Upsert into Pinecone
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        return vector_id

    # Adjust max_workers as needed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for text_chunk in chunks:
            future = executor.submit(process_chunk, text_chunk)
            futures[future] = text_chunk

        for future in as_completed(futures):
            chunk_text = futures[future]
            try:
                upserted_id = future.result()
                print(f"[INFO] Upserted chunk (ID: {upserted_id}): {chunk_text[:70]}...")
            except Exception as e:
                print(f"[ERROR] Failed to upsert chunk: {chunk_text[:70]}..., Error: {e}")


# -------------------------------------------------------------------------------
# 5. Main Function
# -------------------------------------------------------------------------------
def main():
    """
    This main function uses hard-coded parameters for:
      - pdf_path
      - chunk_size
      - overlap
      - encoding_name
    """
    # Hardcode your variables here
    pdf_path = "./documents/calc.pdf"
    chunk_size = 1000
    overlap = 250
    encoding_name = "cl100k_base"

    # 1) Read and concatenate PDF text
    print(f"[INFO] Reading PDF: {pdf_path}")
    text = read_pdf_text(pdf_path)

    # 2) Tokenize
    print("[INFO] Tokenizing text...")
    token_ids = tokenize_text(text, encoding_name=encoding_name)
    print(f"[INFO] Total tokens: {len(token_ids)}")

    # 3) Chunk
    print(f"[INFO] Chunking tokens into segments of {chunk_size} with {overlap} overlap...")
    token_chunks = chunk_tokens(token_ids, chunk_size=chunk_size, overlap=overlap)
    print(f"[INFO] Number of token chunks: {len(token_chunks)}")

    # 4) Convert token chunks back to strings
    text_chunks = [detokenize_tokens(chunk, encoding_name=encoding_name) for chunk in token_chunks]

    # 5) Embed & upsert chunks into Pinecone
    print("[INFO] Embedding and upserting chunks to Pinecone...")
    embed_and_upsert(text_chunks, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
