import bot
from client import AzureClient
import os
import json
import concurrent.futures
from threading import Lock

from dotenv import load_dotenv
from client import AzureClient, HuggingFaceClient, PerplexityClient

import argparse


def call_gpt4o(client, question: str) -> str:
    """
    Calls Azure GPT-4o (ChatCompletion) to get a response to `question`.
    """
    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant. Answer all questions completely and precisely."},
        {"role": "user",   "content": question},
    ]
    if client.__class__.__name__ == "AzureClient":
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Adjust to your actual deployment name if needed
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from GPT-4o: {e}"
    elif client.__class__.__name__ == "HuggingFaceClient":
        try:
            response = client.generate_text(question)
            return response.strip()
        except Exception as e:
            return f"Error from GPT-4o: {e}"


def process_question_with_bot(question, client=None):
    """
    Calls the bot's `process_input` function to process a single question.
    """
    try:
        return bot.process_input(question, client=client)
    except Exception as e:
        return f"Error: {e}"


def rag_worker_perplexity(item, idx, total, file_obj, file_lock, client: PerplexityClient):
    """
    Worker function that processes a single dataset item.
    Writes the result (in JSONL format) to file immediately.
    Prints progress as "idx/total".
    """
    def answer_perplexity(user_input):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful tutor. Please explain thoroughly, make clear the actual final answer."
            },
            {
                "role": "user",
                "content": (
                    f"User's question: {user_input}\n\n"
                    "Use the following reference Q&A pairs to guide your answer:\n"
                )
            }
        ]
        try:
            response = client.generate_response(messages)
            return response
        except Exception as e:
            return f"Error with Perplexity completion: {e}"

    question = item.get("question", "")
    gold_answer = item.get("answer", "")
    # Call the bot to process the question
    model_answer = answer_perplexity(question)

    # Build a result record
    result_record = {
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer
    }

    # Print minimal progress info to console
    print(f"{idx}/{total}", flush=True)

    # Write to JSONL file
    # We use a lock so multiple threads don't overwrite each other's data
    with file_lock:
        line = json.dumps(result_record, ensure_ascii=False)
        file_obj.write(line + "\n")
        file_obj.flush()


def rag_worker(item, idx, total, file_obj, file_lock, client):
    """
    Worker function that processes a single dataset item.
    Writes the result (in JSONL format) to file immediately.
    Prints progress as "idx/total".
    """
    question = item.get("question", "")
    gold_answer = item.get("answer", "")

    # Call the bot to process the question
    model_answer = process_question_with_bot(question, client=client)

    # Build a result record
    result_record = {
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer
    }

    # Print minimal progress info to console
    print(f"{idx}/{total}", flush=True)

    # Write to JSONL file
    # We use a lock so multiple threads don't overwrite each other's data
    with file_lock:
        line = json.dumps(result_record, ensure_ascii=False)
        file_obj.write(line + "\n")
        file_obj.flush()

def base_worker(item, idx, total, file_obj, file_lock, client):
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


def finetune_worker(item, idx, total, file_obj, file_lock, client):
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
    parser = argparse.ArgumentParser(description="Run testcases")
    parser.add_argument("--input_file", type=str, help="Path to the input file",
                        default="./mathematics_dataset_json/math_data/train-medium/calculus__differentiate.json")
    parser.add_argument("--output_file", type=str, help="Path to the output file",
                        default="./datasets/validation_results_base.jsonl")
    parser.add_argument('--client', type=str, help="Client to use",
                        choices=["azure", "local", 'perplexity'], default="local")
    parser.add_argument('--model_type', type=str, help="Model type to use",
                        choices=['rag', 'base', 'finetune'], default="base")
    args = parser.parse_args()
    client_name = args.client
    base_model = args.model_type
    print(f"Client: {client_name}, Model: {base_model}")
    input_file = "./mathematics_dataset_json/math_data/train-medium/calculus__differentiate.json"
    output_file = f"./datasets/validation_results_{client_name}_{base_model}.jsonl"
    if client_name == "azure":
        load_dotenv()
        client = AzureClient(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01"
        )
    elif client_name == "local":
        client = HuggingFaceClient(model_name="gpt2")
    elif client_name == "perplexity":
        client = PerplexityClient(api_key=os.getenv("PERPLEXITY_API_KEY"))
    else:
        raise ValueError(f"Unknown client: {client_name}")
    if base_model == "rag" and client_name == "azure":
        worker = rag_worker
    elif base_model == "base":
        worker = base_worker
    elif base_model == "finetune":
        worker = finetune_worker
    elif base_model == "rag" and client_name == "perplexity":
        worker = rag_worker_perplexity
    else:
        raise ValueError(f"Unknown model type: {base_model}")
    print(f"Using {base_model} model with {client_name} client")
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
