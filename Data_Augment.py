import json
import time
import random
import sys
import os
import argparse 
from typing import List, Dict, Any, Union
from tqdm import tqdm
from together import Together

# Environment variable to set your Together AI API key.
# export TOGETHER_API_KEY='your_api_key_here' (Linux/macOS)
TOGETHER_API_KEY_ENV_VAR = "TOGETHER_API_KEY" 

AI_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" 
MIN_STEPS = 7
MAX_STEPS = 9 
BASE_DELAY = 2  
MAX_DELAY = 60  
JITTER_PERCENTAGE = 0.10 

def get_exponential_backoff_delay(attempt: int, base_delay: int = BASE_DELAY, max_delay: int = MAX_DELAY) -> float:
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(-JITTER_PERCENTAGE * delay, JITTER_PERCENTAGE * delay)
    return delay + jitter

def get_step_wise_answer(
    client: Together,
    question: str,
    answer: str,
    max_retries: int = 12
) -> str:
    for attempt in range(max_retries):
        try:
            user_prompt = f"""You have the following question and its corresponding answer. Your task is to convert the answer only into {MIN_STEPS} - {MAX_STEPS} logical steps.

## Question: {question}
## Answer: {answer}

Give the response in the following format:
## Step wise format: [your response]
"""
            response = client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=60
            )
            content = response.choices[0].message.content
            if "step" in content.lower():
                return content
            delay = get_exponential_backoff_delay(attempt)
            print(f"Attempt {attempt + 1}/{max_retries}: 'step' not found. Retrying in {delay:.2f}s...")
            time.sleep(delay)
        except Exception as e:
            delay = get_exponential_backoff_delay(attempt)
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    print(f"Max retries ({max_retries}) reached for: '{question[:100]}...' Exiting.")
    sys.exit(1)

def process_data(input_file_path: str, output_file_path: str) -> None:
    api_key = os.getenv(TOGETHER_API_KEY_ENV_VAR)
    if not api_key:
        print(f"Error: {TOGETHER_API_KEY_ENV_VAR} environment variable not set.")
        sys.exit(1)

    try:
        client = Together(api_key=api_key)
        with open(input_file_path, 'r', encoding='utf-8') as in_f:
            data = json.load(in_f)

        with open(output_file_path, 'a', encoding='utf-8') as out_f:
            for i, item in enumerate(tqdm(data, desc="Processing questions")):
                try:
                    question = item.get('problem')
                    answer = item.get('solution')
                    if not question or not answer:
                        continue
                    step_wise_answer = get_step_wise_answer(client, question, answer)
                    result = {
                        'question': question,
                        'answer': answer,
                        'step_wise_answer': step_wise_answer
                    }
                    json.dump(result, out_f, ensure_ascii=False)
                    out_f.write('\n')
                    out_f.flush()
                    time.sleep(BASE_DELAY)
                except Exception as e:
                    print(f"\nError processing index {i}: {str(e)}. Continuing...")
                    time.sleep(BASE_DELAY * 2)
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"A fatal error occurred: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process a JSON file to generate step-wise answers using Together AI.")
    parser.add_argument('input_file', type=str, help="Path to the input JSON file")
    parser.add_argument('output_file', type=str, help="Path to the output JSONL file")
    
    args = parser.parse_args()
    process_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()