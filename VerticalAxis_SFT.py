import os
import json
import torch
import tensorboard
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

DATASET_PATH = "/dataset"
OUTPUT_DIR = "/output"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_data(examples):
    if isinstance(examples['question'], list):
        formatted_texts = []
        for i in range(len(examples['question'])):
            messages = [
                {"role": "user", "content": examples['question'][i]},
                {"role": "assistant", "content": examples['step_wise_answer'][i]}
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        return formatted_texts
    else:
        messages = [
            {"role": "user", "content": examples['question']},
            {"role": "assistant", "content": examples['step_wise_answer']}
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return [formatted_text]

def load_and_prepare_dataset(dataset_path, split_ratio=0.9, seed=42):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_data, val_data = train_test_split(
        raw_data, test_size=1 - split_ratio, random_state=seed
    )

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(val_data)

    return DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })

dataset = load_and_prepare_dataset(DATASET_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available(),
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    save_strategy="steps",
    eval_strategy="steps",
    report_to="none",
    load_best_model_at_end=True,
    save_total_limit=2,
    eval_accumulation_steps=4
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    formatting_func=format_data,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
    max_seq_length=1500,
    packing=False,
)

trainer.train()
#trainer.train(resume_from_checkpoint="Checkpoint Path") #Uncomment to resume training from a checkpoint

final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)