import os
import json
import pymupdf
from dotenv import load_dotenv
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

def initialize_client():
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )

def extract_pdf_to_jsonl(pdf_path, output_jsonl_path):
    """
    Extract text from a PDF and save it into a JSONL file,
    where each line corresponds to a *pair of consecutive pages*.
    E.g., page1-page2, page2-page3, etc.
    """
    try:
        doc = pymupdf.open(pdf_path)

        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            # Only go up to len(doc) - 1 so we can pair each page with the next one.
            for page_num in range(len(doc) - 1):
                # Current page
                page1 = doc[page_num]
                text1 = page1.get_text()

                # Next consecutive page
                page2 = doc[page_num + 1]
                text2 = page2.get_text()

                # Combine texts
                combined_text = text1 + "\n" + text2

                # Create data record for the pair
                data = {
                    "page_number": f"{page_num + 1}-{page_num + 2}",
                    "content": combined_text.strip(),
                }
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"PDF content (in page pairs) saved to {output_jsonl_path}")

    except Exception as e:
        print(f"Error during PDF extraction: {e}")

def _request_qa_pair(client, content, model_name, call_index):
    """
    Helper function to send one request to the model to generate Q&A pairs.
    We use call_index to vary the user prompt slightly, which can help the model produce
    more diverse outputs if needed.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful educational assistant that generates high-quality "
                "calculus-based Q&A datasets for training an AI tutor."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate 10 unique and involved calculus-based question-answer pairs (variant #{call_index+1}) "
                "based on concepts and topics covered in the following text:\n\n"
                f"{content}\n\n"
                "Each question-answer pair must be complete and self-contained, without referencing "
                "the textbook or requiring additional context outside of the pair itself. "
                "Thoroughly explain each answer, the steps to get there, and the concepts involved as if you were a tutor helping a confused student. "
                "Output only a single line with a comma-separated list of JSON objects of the form "
                '{"question":"Your question here.", "answer":"Your answer here."}, '
                "and no additional text or line breaks."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    return response

def generate_qa_from_content(client, input_jsonl, output_jsonl, model_name):
    """
    Reads each line from the JSONL file (where each line is a pair of pages),
    and for each line, makes 10 parallel requests to the model.
    
    Each request returns 10 Q&A pairs, so in total we'll get 100 Q&A pairs 
    (in 10 separate responses) for every pair of pages.

    The final output is a JSONL in which each line corresponds to one
    question-answer pair, including page range and token usage.
    """
    try:
        with open(input_jsonl, "r", encoding="utf-8") as infile, open(
            output_jsonl, "w", encoding="utf-8"
        ) as outfile:

            for line in infile:
                page_data = json.loads(line)
                content = page_data.get("content", "")
                page_number = page_data.get("page_number", "Unknown")

                # Skip if no content
                if not content:
                    continue

                # We will make 10 parallel calls. Each call returns 10 Q&A pairs.
                # That means we get a total of 100 question-answer pairs for this pair of pages.
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(
                            _request_qa_pair,
                            client,
                            content,
                            model_name,
                            call_index
                        )
                        for call_index in range(10)
                    ]

                    for future in as_completed(futures):
                        try:
                            response = future.result()

                            # Extract the raw text from the first choice
                            qa_pairs_raw = response.choices[0].message.content.strip()
                            usage_data = {
                                "completion_tokens": response.usage.completion_tokens,
                                "prompt_tokens": response.usage.prompt_tokens,
                                "total_tokens": response.usage.total_tokens,
                            }

                            # The model returns a single line containing a JSON array
                            # of 10 objects: [{"question":"...","answer":"..."}, ...]
                            # Parse that string to get the individual Q&A pairs.
                            try:
                                qa_pairs_list = json.loads(qa_pairs_raw)
                            except json.JSONDecodeError:
                                # If parsing fails, skip
                                continue

                            # Write each question-answer pair on its own line
                            for qa_item in qa_pairs_list:
                                out_record = {
                                    "page_number": page_number,
                                    "question": qa_item.get("question", ""),
                                    "answer": qa_item.get("answer", ""),
                                    "prompt_tokens": usage_data["prompt_tokens"],
                                    "completion_tokens": usage_data["completion_tokens"],
                                    "total_tokens": usage_data["total_tokens"],
                                    "model": response.model,
                                    "response_id": response.id,
                                }
                                outfile.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                        except Exception as e:
                            # If a call fails, we can log an error line if needed
                            error_record = {
                                "page_number": page_number,
                                "error": str(e),
                            }
                            outfile.write(json.dumps(error_record, ensure_ascii=False) + "\n")

        print(f"QA pairs saved to {output_jsonl}")
    except Exception as e:
        print(f"Error during QA generation: {e}")

if __name__ == "__main__":
    # Initialize client
    client = initialize_client()

    # Paths for PDF and output files
    pdf_path = "./documents/calc.pdf"
    output_jsonl_path = "./datasets/output.jsonl"       # Page-paired text
    qa_output_path = "./datasets/qa_pairs.jsonl"        # Individual Q&A pairs
    model_name = "gpt-4o"  # Or your Azure OpenAI model name

    # 1) Extract PDF content in pairs
    extract_pdf_to_jsonl(pdf_path, output_jsonl_path)

    # 2) Generate QA pairs from each page-pair (10 parallel calls per page-pair)
    generate_qa_from_content(client, output_jsonl_path, qa_output_path, model_name)
