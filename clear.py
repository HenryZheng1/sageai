import os
from dotenv import load_dotenv
from pinecone import Pinecone
def clear(PINECONE_API_KEY, PINECONE_INDEX_HOST):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    response = index.delete(delete_all=True)
    print("Delete response:", response)
    print("All vectors have been removed from the index.")
if __name__ == '__main__':
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")