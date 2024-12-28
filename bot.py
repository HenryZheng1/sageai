import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# ------------------------------------------------------------------------------
# 1. Environment & Config
# ------------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = "2024-02-01"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CHAT_MODEL_NAME = "gpt-4o"                   # The name of your Azure OpenAI chat deployment
EMBED_MODEL_NAME = "text-embedding-3-large"  # The name of your Azure OpenAI embedding model

# ------------------------------------------------------------------------------
# 2. Initialize Azure OpenAI
# ------------------------------------------------------------------------------
def initialize_azure_client():
    """
    Create and return an AzureOpenAI client object.
    """
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION
    )

# ------------------------------------------------------------------------------
# 3. Initialize Pinecone
# ------------------------------------------------------------------------------
def initialize_pinecone():
    """
    Create and return a Pinecone index instance.
    """
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    return index

# ------------------------------------------------------------------------------
# 4. Process Input
# ------------------------------------------------------------------------------
def process_input(user_input):
    """
    Processes a single input using Pinecone and GPT-4o.
    
    Args:
        user_input (str): The input question or query.

    Returns:
        str: The model's response.
    """
    # Initialize clients
    client = initialize_azure_client()
    index = initialize_pinecone()

    # 4.1 Generate embedding (vector) for the user query
    try:
        embedding_response = client.embeddings.create(
            model=EMBED_MODEL_NAME,
            input=[user_input]
        )
        user_vector = embedding_response.data[0].embedding
    except Exception as e:
        return f"Error generating embedding: {e}"

    # 4.2 Query Pinecone
    try:
        pinecone_results = index.query(
            vector=user_vector,
            text=user_input,
            alpha=0.5,  # Hybrid search: balance between lexical and semantic
            top_k=5,
            include_metadata=True
        )
    except Exception as e:
        return f"Error querying Pinecone: {e}"

    # 4.3 Gather top-5 matches
    context_snippets = []
    for match in pinecone_results.matches:
        q = match.metadata.get("question", "")
        a = match.metadata.get("answer", "")
        context_snippets.append(f"Q: {q}\nA: {a}\n")

    reference_text = "\n".join(context_snippets)

    # 4.4 Create messages for GPT-4o
    messages = [
        {
            "role": "system",
            "content": "You are a helpful tutor. Please explain thoroughly."
        },
        {
            "role": "user",
            "content": (
                f"User's question: {user_input}\n\n"
                "Use the following reference Q&A pairs to guide your answer:\n"
                f"{reference_text}\n\n"
                "Now, provide the best possible answer to the user's question."
            )
        }
    ]

    # 4.5 Call GPT-4o to get the final answer
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with GPT-4o completion: {e}"

# ------------------------------------------------------------------------------
# 5. Main Entry for Interactive Console Chat (Optional)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Welcome to the GPT-4o + Pinecone chatbot! Type 'exit' or 'quit' to leave.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = process_input(user_input)
        print(f"\nAssistant: {response}\n")