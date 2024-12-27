import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# 1. Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Connect to your existing index
index = pc.Index(host=PINECONE_INDEX_HOST)

# 3. Delete all vectors in the index (retaining the index itself)
response = index.delete(delete_all=True)
print("Delete response:", response)
print("All vectors have been removed from the index.")