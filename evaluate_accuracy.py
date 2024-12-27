import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# For loading environment variables
from dotenv import load_dotenv

# AzureOpenAI is part of the openai package when configured for Azure usage
# pip install openai
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

def azure_compare(client, question: str, gold_answer: str, model_answer: str, model_name: str) -> str:
    """
    Calls Azure GPT to decide whether the model_answer is "correct" or "incorrect",
    comparing it to gold_answer. Returns one of: "correct", "incorrect", or "unknown".
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math expert. The user will give you a question, a gold (reference) answer, "
                "and a student's answer. You must determine if the student's final answer matches the gold answer. "
                "If they are equivalent or correct, respond with the single word: correct. Otherwise respond with the single word: incorrect."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Gold Answer: {gold_answer}\n"
                f"Student Answer: {model_answer}\n\n"
                "Respond only with either 'correct' or 'incorrect'. No additional explanation."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        evaluation = response.choices[0].message.content.strip().lower()

        # Enforce strict parsing: if GPT does not return exactly "correct" or "incorrect", set "unknown"
        if evaluation not in ["correct", "incorrect"]:
            evaluation = "unknown"

        return evaluation
    except Exception as e:
        print(f"[ERROR] Azure GPT call failed: {e}")
        return "unknown"

def process_line(line: str, client, model_name: str) -> dict:
    """
    Process a single JSONL line:
    1. Parse JSON.
    2. Call Azure GPT to compare gold vs. model answer.
    3. Return updated record with "evaluation" field.
    """
    try:
        data = json.loads(line.strip())
        question = data.get("question", "")
        gold_answer = data.get("gold_answer", "")
        model_answer = data.get("model_answer", "")

        evaluation = azure_compare(client, question, gold_answer, model_answer, model_name)
        data["evaluation"] = evaluation
        return data
    except json.JSONDecodeError:
        # If we cannot parse, return an error record
        return {"error": "Invalid JSON line", "original_line": line}

def main():
    """
    Sets the required variables and processes input JSONL to evaluate answers using Azure GPT.
    """
    # Set paths, model name, and worker count directly
    input_file = "./validation_results.jsonl"
    output_file = "./final_results.jsonl"
    model_name = "gpt-4o"
    max_workers = 3

    # Initialize Azure GPT client
    client = initialize_client()

    # Read all lines so we know how many there are and to filter out any blank lines
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = [line for line in infile if line.strip()]

    total_count = len(lines)
    print(f"Found {total_count} lines to process.\n")

    with open(output_file, "w", encoding="utf-8") as outfile, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Submit all lines to the thread pool
        futures = [executor.submit(process_line, line, client, model_name) for line in lines]

        # We'll store all completed results so we can compute statistics at the end
        all_results = []

        # Collect results as they complete
        for i, future in enumerate(as_completed(futures)):
            processed_record = future.result()
            all_results.append(processed_record)

            # Write to output file immediately
            outfile.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
            outfile.flush()

            # Print progress: i+1 because i is 0-based
            print(f"{i+1}/{total_count}")

    # Now compute correctness statistics
    correct_count = sum(1 for r in all_results if r.get("evaluation") == "correct")
    incorrect_count = sum(1 for r in all_results if r.get("evaluation") == "incorrect")
    unknown_count = sum(1 for r in all_results if r.get("evaluation") == "unknown")

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Correct:   {correct_count}/{total_count} "
          f"({correct_count / total_count * 100:.2f}%)")
    print(f"Incorrect: {incorrect_count}")
    print(f"Unknown:   {unknown_count}")

if __name__ == "__main__":
    main()
