#!/usr/bin/env python3
"""
Script to parse JSONL files containing generation data with nested JSON strings.
Loads the file, extracts the 'generation' field, and parses it into structured data.
"""

import orjson
import json
import sys
from typing import Dict, List, Any
from pathlib import Path
import random



def parse_generation_content(generation_str: str) -> Dict[str, Any]:
   
    EXPECTED_KEYS = {"question_text", "initial_options", "additional_options", "correct_option_index"}
    
    generation_str = generation_str.split("<|start|>assistant<|channel|>final<|message|>")[-1].strip()
    
    # Find the JSON part (starts with '{' and ends with '}')
    start_idx = generation_str.find('{')
    end_idx = generation_str.rfind('}') + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No valid JSON found in generation string")
    
    json_str = generation_str[start_idx:end_idx]
    
    try:
        # Parse the JSON content
        parsed_data = orjson.loads(json_str)
    except orjson.JSONDecodeError as e:
        # Fallback to standard json if orjson fails
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON content: {e}")
    
    # Validate that only expected keys are present
    actual_keys = set(parsed_data.keys())
    unexpected_keys = actual_keys - EXPECTED_KEYS
    missing_keys = EXPECTED_KEYS - actual_keys
    
    if unexpected_keys:
        raise ValueError(f"Unexpected keys found: {unexpected_keys}. Only {EXPECTED_KEYS} are allowed.")
    
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}. All keys {EXPECTED_KEYS} are required.")
    
    return parsed_data

def shuffle_options(parsed_data: Dict[str, Any]) -> (str, str):
    """
    Combine initial and additional options, shuffle them, and return the modified problem text
    along with the correct answer.
    """

    question_text = parsed_data["question_text"]
    initial_options = parsed_data["initial_options"]
    additional_options = parsed_data["additional_options"]
    correct_index = parsed_data["correct_option_index"]

    all_options = initial_options + additional_options
    correct_answer = all_options[correct_index]

    # Shuffle options
    random.shuffle(all_options)

    correct_answer_index = all_options.index(correct_answer)

    # Create new problem text with shuffled options
    options_text = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(all_options)])
    modified_problem = f"{question_text}\n{options_text}"

    correct_answer = [f"{chr(65 + i)}" for i in range(len(all_options))][correct_answer_index]

    return modified_problem, correct_answer, all_options


def load_and_parse_jsonl(file_path: str, output_file_path: str = None) -> List[Dict[str, Any]]:

    extracted_data = []
    
    with open(file_path, 'rb') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the JSONL line
                data = orjson.loads(line)
                
                if 'generation' not in data:
                    print(f"Warning: No 'generation' field found in line {line_num}")
                    continue
                
                # Parse the generation content
                generation_content = parse_generation_content(data['generation'])
                
                # Create result entry with original data + parsed generation
                data.update(generation_content)
                data["raw_problem"] = data["problem"]
                problem, correct_answer, options = shuffle_options(generation_content)
                data["problem"] = problem
                data["expected_answer"] = correct_answer
                data["options"] = options
                keep_only = [
                    "id",
                    "problem",
                    "topic",
                    "subtoptic",
                    "difficulty_model",
                    "difficulty_model_pass_at_n",
                    "difficulty_model_pass_rate",
                    "correct_answer",
                    "options",
                    "question_text",
                ]
                data = {k: data[k] for k in keep_only if k in data}
                
                extracted_data.append(data)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Save extracted data to output file if specified
    if output_file_path and extracted_data:
        save_extracted_jsonl(extracted_data, output_file_path)
        print(f"Saved {len(extracted_data)} extracted entries to {output_file_path}")
    
    return extracted_data


def save_extracted_jsonl(extracted_data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the extracted generation data to a JSONL file.
    
    Args:
        extracted_data: List of dictionaries containing only the extracted keys
        output_path: Path where to save the JSONL file
    """
    with open(output_path, 'wb') as f:
        for entry in extracted_data:
            line = orjson.dumps(entry) + b'\n'
            f.write(line)


def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage: python parse_new_options.py <input_jsonl_file> [output_jsonl_file]")
        print("If output_jsonl_file is not provided, it will be auto-generated as <input>_extracted.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_extracted.jsonl"
    
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    try:
        results = load_and_parse_jsonl(input_file, str(output_file))
        
        print(f"Successfully processed {len(results)} entries")
        
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
