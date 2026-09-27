from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import torch, math, random, itertools
from torch.utils.data import DataLoader, WeightedRandomSampler
import re
import glob
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, util
from typing import Optional
import os
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from collections import defaultdict

# --- Configuration ---
max_seq_length = 2048
lora_rank = 64
DATASET_DIR = "/dataset"
SFT_CHECKPOINT_PATH = "/output/checkpoint"
OUTPUT_DIR = "/output"
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "adapters")
OTHER_MAX = 2.5

SYSTEM_PROMPT = """
You are a maths question solving model, currently you are learning to be better. Following the instruction carefully:
If the user message contains the token <CONTINUE>, that token marks the
point where your reasoning must start. Continue from there, then answer.
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#embedder = SentenceTransformer("sentence-transformers/LaBSE") #Uncomment for training on Hindi/Marathi dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_CHECKPOINT_PATH,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,
)
tokenizer.pad_token = tokenizer.eos_token


def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    boxed_match = re.search(r'\\boxed\s*\{', text)
    if boxed_match:
        start_pos = boxed_match.end()
        brace_count = 1
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content = text[start_pos:i].strip()
                    nums_in_box = re.findall(r"-?\d+(?:\.\d+)?", content)
                    if nums_in_box:
                        return nums_in_box[-1]
                    break 
    all_nums = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    if all_nums:
        return all_nums[-1].replace(',', '')
        
    return None

def is_answer_present(extracted_answer: str, full_response: str) -> bool:
    if not extracted_answer:
        return False
    search_area = full_response[-200:].lower()
    answer = extracted_answer.strip().lower()
    if re.match(r'^-?\d+(\.\d+)?$', answer):
        return bool(re.search(rf'\b{re.escape(answer)}\b', search_area))
    else:
        return answer in search_area

def is_single_token_expression(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[-+]?[\w\d\s\^*/()+-]+", stripped))

def _as_list_reward(x, n):
    return x if isinstance(x, (list, tuple)) else [x] * n

def correctness_reward_noxml(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response, correct_answer in zip(responses, answer):
        if not correct_answer:
            rewards.append(0.0)
        elif is_answer_present(extracted_answer=correct_answer, full_response=response):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

def enforce_continuation_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for prompt, response, _ in zip(prompts, responses, answer):
        prompt_text = prompt[-1]['content']
        stepwise_match = re.search(r"Step wise format:\s*(.*?)<CONTINUE>", prompt_text, re.DOTALL)
        #stepwise_match = re.search(r"चरणनिहाय स्वरूपः\s*(.*?)<CONTINUE>", prompt_text, re.DOTALL) #Uncomment for Marathi
        #stepwise_match = re.search(r"चरणबद्ध प्रारूपः\s*(.*?)<CONTINUE>", prompt_text, re.DOTALL)  #Uncomment for Hindi

        if stepwise_match:
            steps = re.findall(r"^\s*(\d+)\.", stepwise_match.group(1), re.MULTILINE)
            expected_start = int(steps[-1]) + 1 if steps else 1
        else:
            expected_start = 1
        
        model_match = re.match(r"^\s*(\d+)\.", response.strip())
        if not model_match:
            rewards.append(0.0)
            continue
            
        model_start = int(model_match.group(1))
        
        if model_start == expected_start:
            rewards.append(1.0)
        elif model_start == 1 and expected_start > 1:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards

def cosine_reward_func(prompts, completions, reference_continuation, stage, **kwargs):
    B = len(completions)
    reference_continuation = _as_list_reward(reference_continuation, B)
    stage = _as_list_reward(stage, B)
    gen_responses = [c[0]["content"] for c in completions]
    cos_vals = []
    for gen, ref in zip(gen_responses, reference_continuation):
        emb_g = embedder.encode(gen, convert_to_tensor=True, normalize_embeddings=True)
        emb_r = embedder.encode("" if ref is None else ref, convert_to_tensor=True, normalize_embeddings=True)
        cs = util.cos_sim(emb_g, emb_r).item()
        cos_vals.append((cs + 1) / 2)
    
    S_max = max(stage) or 1
    w = [OTHER_MAX * (1 - s / S_max) for s in stage]
    return [float(w_i * cs_i) for w_i, cs_i in zip(w, cos_vals)]

def strict_numeric_or_symbolic_reward(completions, **_):
    rewards = []
    for completion in completions:
        out = completion[0]["content"]
        boxed = extract_final_answer(out)
        if boxed and is_single_token_expression(boxed):
            rewards.append(0.5)
            continue

        tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", out, re.S)
        if tag_match:
            answer = tag_match.group(1)
            if is_single_token_expression(answer):
                rewards.append(0.5)
                continue

        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            if is_single_token_expression(last_line):
                rewards.append(0.5)
                continue
        rewards.append(0.0)
    return rewards



def _as_list_data(steps):
    return steps if isinstance(steps, list) else [s for s in steps.split("\n") if s.strip()]

def load_local_dataset():
    files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {DATASET_DIR}")
    ds = load_dataset("json", data_files=files, split="train")
    return ds.map(lambda x: {"prompt": x["question"], "gold": x["answer"]})

def make_shifted_staged_sets(split="train"):
    raw = load_local_dataset()
    stagebuf = defaultdict(list)
    for ex in raw:
        steps = _as_list_data(ex["step_wise_answer"])
        n = len(steps)
        stage = 0
        while True:
            removal = stage + 2
            keep = n - removal
            if keep <= 0:
                break

            partial_with_tag = "\n".join(steps[:keep]) + "\n<CONTINUE>\n"
            user_content = ex["question"] + "\n" + partial_with_tag
            full_ref_continuation = "\n".join(steps[keep:]).strip()

            stagebuf[stage].append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (user_content).strip()}
                ],
                "answer": extract_final_answer(ex["gold"]),
                "stage": stage,
                "reference_continuation": full_ref_continuation
            })
            stage += 1
    return [Dataset.from_list(stagebuf[s]) for s in sorted(stagebuf)]

def build_mixed_dataset(staged_sets, alpha_start=0.7):
    full = Dataset.from_list(list(itertools.chain.from_iterable(staged_sets)))
    weights = torch.tensor([alpha_start ** ex["stage"] for ex in full])
    sampler = WeightedRandomSampler(weights, num_samples=len(full), replacement=True)
    return full, sampler

# --- Custom Trainer and Callback ---

class CurriculumTrainer(GRPOTrainer):
    def __init__(self, sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._curriculum_sampler = sampler

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self._curriculum_sampler,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            drop_last=True
        )

class CurriculumSamplerCallback(TrainerCallback):
    def __init__(self, sampler, dataset, warmup_steps, alpha_start=0.7, alpha_end=1.0):
        self.sampler = sampler
        self.dataset = dataset
        self.a0 = alpha_start
        self.a1 = alpha_end
        self.T = warmup_steps

    def on_step_end(self, args, state, control, **kwargs):
        frac = min(state.global_step / self.T, 1.0)
        alpha = self.a0 + frac * (self.a1 - self.a0)
        new_w = torch.tensor([alpha ** ex["stage"] for ex in self.dataset])
        self.sampler.weights.copy_(new_w)

# --- Training Execution ---

M = 4
alpha_start = 0.7
alpha_end = 1.0
warmup_steps = 100

staged_sets = make_shifted_staged_sets("train")
full_mixed_set, sampler = build_mixed_dataset(staged_sets, alpha_start)
curriculum_cb = CurriculumSamplerCallback(sampler, full_mixed_set, warmup_steps, alpha_start, alpha_end)

reward_funcs = [
    strict_numeric_or_symbolic_reward,
    correctness_reward_noxml,
    enforce_continuation_reward,
    cosine_reward_func
]

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=1200,
    max_completion_length=1024,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=3,
    max_grad_norm=0.1,
    report_to="tensorboard",
    output_dir="/output",
    dataloader_drop_last=False,
    gradient_checkpointing=True,
    ignore_data_skip=True,
    generation_kwargs={
        "max_tokens": 1200,
        "temperature": 0.6
    }
)

trainer = CurriculumTrainer(
    sampler=sampler,
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=full_mixed_set,
    callbacks=[curriculum_cb]
)

trainer.train()

model.save_lora(OUTPUT_DIR)