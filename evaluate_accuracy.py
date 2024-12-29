
import os
import json
from dotenv import load_dotenv
from client import AzureClient

load_dotenv()

def chunker(seq, size):
    """
    Yields successive chunks of size `size` from list `seq`.
    """
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]

def build_batch_messages(batch_data):
    """
    Given a list of items, each item is a dict with:
      - "question"
      - "gold_answer"
      - "model_answer"

    Returns the system and user messages for Azure GPT.

    The GPT is instructed to:
      - Output exactly one word ('correct' or 'incorrect') per item,
      - In the same order as the items,
      - One result per line, no extra explanations.
    """
    # System role message
    system_message = {
        "role": "system",
        "content": (
            "You are a math expert. You will receive multiple items in a single request. "
            "Each item has a question, a gold (reference) answer, and a student's answer. "
            "For each item, determine if the student's final answer is correct (matches the gold answer) "
            "or incorrect. Output exactly one word per item: 'correct' or 'incorrect', in the same order, "
            "one result per line, with no extra explanation."
        ),
    }

    # Build the user content string
    user_content_lines = []
    for idx, item in enumerate(batch_data, start=1):
        user_content_lines.append(f"Item #{idx}:")
        user_content_lines.append(f"Question: {item['question']}")
        user_content_lines.append(f"Gold Answer: {item['gold_answer']}")
        user_content_lines.append(f"Student Answer: {item['model_answer']}")
        user_content_lines.append("")  # blank line to separate items

    user_message = {
        "role": "user",
        "content": "\n".join(user_content_lines)
    }

    return [system_message, user_message]

def azure_compare_batch(client, batch_data, model_name):
    """
    Calls Azure GPT for a batch of items. Returns a list of 'correct', 'incorrect', or 'unknown' 
    evaluations in the same order as `batch_data`.

    1. Build one prompt with all items.
    2. Expect one line per item in the response.
    3. Parse each line into 'correct' or 'incorrect' (or 'unknown' if parsing fails).
    """
    messages = build_batch_messages(batch_data)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        full_response = response.choices[0].message.content.strip()

        # The GPT should return one line per item, each "correct" or "incorrect"
        lines = full_response.splitlines()
        lines = [line.strip().lower() for line in lines if line.strip()]

        # Map lines back to items
        results = []
        for i in range(len(batch_data)):
            if i < len(lines) and lines[i] in ["correct", "incorrect"]:
                results.append(lines[i])
            else:
                results.append("unknown")

        return results

    except Exception as e:
        print(f"[ERROR] Azure GPT batch call failed: {e}", flush=True)
        # If there's an error, just return "unknown" for each item in the batch
        return ["unknown"] * len(batch_data)

def main():
    """
    Reads from 'validation_results.jsonl', processes each record in batches, 
    writes to 'final_results.jsonl', and prints summary statistics. 
    (Single-threaded, but batched approach)
    """
    # Configuration
    input_file = "./datasets/validation_results_base.jsonl"
    output_file = "./datasets/final_results_base.jsonl"
    model_name = "gpt-4o-mini"
    
    # How many lines per batch
    BATCH_SIZE = 1

    # Initialize Azure GPT client
    client = AzureClient(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    # Read all lines (filter out any blank lines)
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = [line for line in infile if line.strip()]

    total_count = len(lines)
    print(f"Found {total_count} lines to process.\n", flush=True)

    correct_count = 0
    incorrect_count = 0
    unknown_count = 0

    # Open output file once
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Break lines into chunks of BATCH_SIZE
        line_chunks = list(chunker(lines, BATCH_SIZE))

        processed_so_far = 0
        for chunk_idx, chunk in enumerate(line_chunks, start=1):
            # Parse JSON from each line in this chunk
            batch_data = []
            for raw_line in chunk:
                try:
                    record = json.loads(raw_line.strip())
                    batch_data.append({
                        "question": record.get("question", ""),
                        "gold_answer": record.get("gold_answer", ""),
                        "model_answer": record.get("model_answer", "")
                    })
                except json.JSONDecodeError:
                    # In case of malformed lines
                    batch_data.append({
                        "question": "",
                        "gold_answer": "",
                        "model_answer": ""
                    })

            # Call GPT once for the entire batch
            evaluations = azure_compare_batch(client, batch_data, model_name)

            # Write each item's result
            for i, raw_line in enumerate(chunk):
                processed_so_far += 1
                eval_result = evaluations[i] if i < len(evaluations) else "unknown"

                try:
                    record = json.loads(raw_line.strip())
                except json.JSONDecodeError:
                    record = {
                        "error": "Invalid JSON line",
                        "original_line": raw_line
                    }
                record["evaluation"] = eval_result

                # Update counters
                if eval_result == "correct":
                    correct_count += 1
                elif eval_result == "incorrect":
                    incorrect_count += 1
                else:
                    unknown_count += 1

                # Write to file
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

            outfile.flush()

            # Print progress for the batch
            print(
                f"Processed chunk {chunk_idx}/{len(line_chunks)}. "
                f"({processed_so_far}/{total_count} lines done)",
                flush=True
            )

    # Final summary
    print("\n=== EVALUATION SUMMARY ===", flush=True)
    print(f"Correct:   {correct_count}/{total_count} "
          f"({correct_count / total_count * 100:.2f}%)", flush=True)
    print(f"Incorrect: {incorrect_count}", flush=True)
    print(f"Unknown:   {unknown_count}", flush=True)

if __name__ == "__main__":
    main()
