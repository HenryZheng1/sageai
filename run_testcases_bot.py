import json
import bot  # Import the bot.py module
import concurrent.futures
from threading import Lock


def process_question_with_bot(question):
    """
    Calls the bot's `process_input` function to process a single question.
    """
    try:
        return bot.process_input(question)
    except Exception as e:
        return f"Error: {e}"


def worker(item, idx, total, file_obj, file_lock):
    """
    Worker function that processes a single dataset item.
    Writes the result (in JSONL format) to file immediately.
    Prints progress as "idx/total".
    """
    question = item.get("question", "")
    gold_answer = item.get("answer", "")

    # Call the bot to process the question
    model_answer = process_question_with_bot(question)

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


def main():
    input_file = "./mathematics_dataset_json/math_data/train-medium/calculus__differentiate.json"
    output_file = "./datasets/validation_results.jsonl"

    print("Loading dataset...")
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Dataset loaded. Number of items: {len(dataset)}")

    # We'll use a lock to prevent threads from writing at the same time
    file_lock = Lock()

    print(f"Processing and writing immediately to {output_file}...")

    # Open the output file once
    with open(output_file, "w", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            total = len(dataset)
            # Submit each item to the executor
            for idx, item in enumerate(dataset, start=1):
                future = executor.submit(worker, item, idx, total, f_out, file_lock)
                futures.append(future)

            # Wait for all tasks to complete before exiting
            concurrent.futures.wait(futures)

    print(f"All results saved to {output_file}")


if __name__ == "__main__":
    main()