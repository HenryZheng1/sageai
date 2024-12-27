import json

def reformat_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_number, line in enumerate(infile, start=1):
            try:
                # Parse the JSON line
                record = json.loads(line)

                # Skip if the record contains an 'error' key
                if "error" in record:
                    print(f"Skipping line {line_number}: Contains error key")
                    continue
                
                # Ensure required keys are present
                if "question" not in record or "answer" not in record:
                    print(f"Skipping line {line_number}: Missing 'question' or 'answer'")
                    continue
                
                # Construct the new format
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": record["question"]},
                    {"role": "assistant", "content": record["answer"]}
                ]

                # Write the reformatted JSON to the output file
                json.dump({"messages": messages}, outfile)
                outfile.write('\n')
            
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_number}: Invalid JSON - {e}")
            except Exception as e:
                print(f"Error on line {line_number}: {e}")

# Example usage
input_file = "./qa_pairs.jsonl"  # Replace with your input file path
output_file = "./qa_pairs_formatted.jsonl"  # Replace with your desired output file path
reformat_jsonl(input_file, output_file)
