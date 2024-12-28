import os
import json
import uuid
from dotenv import load_dotenv
from client import AzureClient, PineconeClient

# Concurrency imports
from concurrent.futures import ThreadPoolExecutor, as_completed

# NEW Pinecone classes (2.x+)
from pinecone import Pinecone
# Azure OpenAI import
load_dotenv()

# ------------------------------------------------------------------------------
# 1. Environment & Config
# ------------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = "2024-02-01"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
BASE_DIR = './datasets'
JSONL_FILE = "./qa_pairs_formatted.jsonl"
JSONL_FILE = os.path.join(BASE_DIR, JSONL_FILE)
client = AzureClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION
)

pc = PineconeClient(
    api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_HOST
)
index = pc.index

def process_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    # Parse JSON
    record = json.loads(line)
    messages = record.get("messages", [])

    question = ""
    answer = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            question = content
        elif role == "assistant":
            answer = content

    # Skip if question/answer is missing
    if not question or not answer:
        return ""

    # Create embedding via Azure OpenAI
    embedding_response = client.embeddings.create(
        model=MODEL_NAME,
        input=[question]
        # dimensions=EMBEDDING_DIMENSION  # If your model supports specifying dimension
    )

    # Extract the first embedding from the response
    vector = embedding_response.data[0].embedding

    # Prepare metadata
    metadata = {
        "question": question,
        "answer": answer,
    }

    # Generate doc_id and upsert to Pinecone
    doc_id = str(uuid.uuid4())
    index.upsert(
        vectors=[
            {
                "id": doc_id,
                "values": vector,
                "metadata": metadata
            }
        ]
    )

    # Return the (abbreviated) question for logging
    return question
def test():
    print(pc.list_indexes())
# ------------------------------------------------------------------------------
# 5. Read JSONL, run with 20 threads, and handle results
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read lines from file
    with open(JSONL_FILE, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_line_num = {
            executor.submit(process_line, line): idx
            for idx, line in enumerate(lines, start=1)
        }

        for future in as_completed(future_to_line_num):
            line_num = future_to_line_num[future]
            try:
                result = future.result()  # This is the question or ""
                if result:
                    print(f"Upserted line #{line_num}: question=\"{result[:40]}...\"")
            except Exception as e:
                print(f"Error processing line #{line_num}: {e}")

    print("All done!")
    test()
