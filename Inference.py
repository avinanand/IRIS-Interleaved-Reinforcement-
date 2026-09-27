import os
import json
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

CKPT_DIR = Path("/output/checkpoint")
DATA_DIR = Path("/dataset")
OUTPUT_PATH = Path("/output/evaluation_results.json")
BATCH_SIZE = 8

def load_test_data(data_dir: Path) -> list[dict]:
    test_items = []
    skipped_count = 0
    json_files = sorted(glob(str(data_dir / "*.json")))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in {data_dir}")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Warning: Skipping {file_path}, top-level element is not a list.")
                continue

            for item in data:
                if isinstance(item, dict) and "question" in item and "gold_answer" in item:
                    test_items.append(item)
                else:
                    skipped_count += 1
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {file_path}. Error: {e}")
    return test_items

def main():
    config = PeftConfig.from_pretrained(CKPT_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True,
        padding_side='right'
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, CKPT_DIR)
    model.eval()

    test_items = load_test_data(DATA_DIR)
    if not test_items:
        print("No valid test data found. Exiting.")
        return

    results = []
    print(f"Starting evaluation on {len(test_items)} items with batch size {BATCH_SIZE}...")

    for i in tqdm(range(0, len(test_items), BATCH_SIZE), desc="Evaluating Batches"):
        batch_items = test_items[i:i + BATCH_SIZE]
        prompts = [
            f"### Question:\n{item['question']}\n\n### Answer:\n"
            for item in batch_items
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024 
        ).to(model.device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_lengths = inputs["input_ids"].shape[1]
        decoded_outputs = tokenizer.batch_decode(
            output_tokens[:, prompt_lengths:], 
            skip_special_tokens=True
        )
        for item, generated_text in zip(batch_items, decoded_outputs):
            results.append({
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "full_model_response": generated_text.strip(),
            })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
