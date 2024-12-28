from client import AzureClient
import os
import json
import concurrent.futures
from threading import Lock

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

def initialize_client():
    """
    Initializes the Azure OpenAI client using environment variables.
    """
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

def call_gpt4o(client, question: str) -> str:
    """
    Calls Azure GPT-4o (ChatCompletion) to get a response to `question`.
    """
    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant. Answer all questions completely and precisely."},
        {"role": "user",   "content": question},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Adjust to your actual deployment name if needed
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from GPT-4o: {e}"

def worker(item, idx, total, file_obj, file_lock, client):
    """
    Worker function that processes a single dataset item.
    Writes the result (in JSONL format) to file immediately.
    Prints progress as "idx/total".
    """
    question = item.get("question", "")
    gold_answer = item.get("answer", "")

    # Call GPT-4o to get the model's answer
    model_answer = call_gpt4o(client, question)

    # Build a result record
    result_record = {
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer
    }

    # Print minimal progress info to console
    print(f"{idx}/{total}", flush=True)

    # Write to JSONL file (thread-safe using a lock)
    with file_lock:
        line = json.dumps(result_record, ensure_ascii=False)
        file_obj.write(line + "\n")
        file_obj.flush()


def main():
    input_file = "./mathematics_dataset_json/math_data/train-medium/calculus__differentiate.json"
    output_file = "./datasets/validation_results_base.jsonl"

    client = AzureClient(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    print("Loading dataset...")
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Dataset loaded. Number of items: {len(dataset)}")

    file_lock = Lock()

    print(f"Processing and writing immediately to {output_file}...")

    # Open the output file once
    with open(output_file, "w", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            total = len(dataset)

            # Submit each item to the executor
            for idx, item in enumerate(dataset, start=1):
                future = executor.submit(
                    worker, item, idx, total, f_out, file_lock, client
                )
                futures.append(future)

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    print(f"All results saved to {output_file}")

if __name__ == "__main__":
    main()
