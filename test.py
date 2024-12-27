import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")
print(pinecone.list_indexes())