# Training and setup reference

[Back to IRIS](../README.md)

Detailed setup instructions for the supplied experiment scripts.

## Installation

Use Python 3.10 or 3.11 as a starting point for the older training APIs. For GPU workflows, use a Linux machine with an NVIDIA GPU and a matching CUDA-enabled PyTorch installation. GRPO uses Unsloth, vLLM, and 4-bit loading; translation also requests 4-bit loading. The supplied settings are not a CPU or macOS training configuration. GPU memory requirements have not been benchmarked for this release.

Create an environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install only the profile needed for the current stage:

| Stage | Installation command |
| --- | --- |
| API-based augmentation | `python -m pip install -r requirements.txt` |
| Translation | `python -m pip install -r requirements/translation.txt` |
| SFT | `python -m pip install -r requirements/sft.txt` |
| GRPO | `python -m pip install -r requirements/grpo.txt` |
| Inference | `python -m pip install -r requirements/inference.txt` |

**Use separate environments for SFT and GRPO.** The SFT script uses an older TRL interface. The GRPO dependency file lists the required packages but does not establish a compatible, reproducible version combination. See [Compatibility and reproducibility](#compatibility-and-reproducibility) before training.

Install PyTorch using the command appropriate to your GPU from the [official installation guide](https://pytorch.org/get-started/locally/). Follow the [Unsloth installation guide](https://unsloth.ai/docs/get-started/install/pip-install) for the GRPO environment. After installation, check package consistency:

```bash
python -m pip check
```

Translation also requires NLTK sentence-tokenization resources:

```bash
python -m nltk.downloader punkt punkt_tab
```

Model downloads require access to Hugging Face. Authenticate and obtain access to any gated model before running its script.

## Data formats

All examples below are illustrative, not paper benchmark data. Use UTF-8 files and keep training and test data in separate directories.

### Augmentation input

`Data_Augment.py` reads a JSON array with `problem` and `solution` fields:

```json
[
  {
    "problem": "A box contains 3 red balls and 2 blue balls. How many balls are there?",
    "solution": "There are 3 + 2 = 5 balls."
  }
]
```

It writes one JSON object per line (JSONL), containing `question`, `answer`, and `step_wise_answer`.

### Translation and training input

Translation and SFT expect JSON arrays. Store the solution as a string with one numbered step per line:

```json
[
  {
    "question": "A box contains 3 red balls and 2 blue balls. How many balls are there?",
    "answer": "5",
    "step_wise_answer": "## Step wise format:\n1. There are 3 red balls.\n2. There are 2 blue balls.\n3. Add the two counts: 3 + 2 = 5.\n4. The answer is 5."
  }
]
```

SFT consumes `question` and `step_wise_answer`; GRPO additionally uses `answer` to extract the reference final answer. GRPO can also read a list of step strings, but translation and the documented SFT format use strings.

GRPO splits strings on non-empty lines, so headers count as lines too. Its continuation reward expects the `Step wise format:` marker followed by numbered lines. Inspect augmented examples for this structure: augmentation checks only that the response contains the word `step`, not that numbering or step counts are valid.

### Inference input

Each test file must be a JSON array containing `question` and `gold_answer`:

```json
[
  {
    "question": "What is 3 + 2?",
    "gold_answer": "5"
  }
]
```

Inference saves `question`, `gold_answer`, and `full_model_response`. It does **not** calculate accuracy or other benchmark metrics.

## Running the pipeline

Run commands from the repository root. Except for augmentation, paths and settings are configured by editing constants in the scripts; they are not command-line arguments or environment-variable overrides.

### 1. Generate step-wise solutions

Set `TOGETHER_API_KEY` in your shell, then run:

```bash
mkdir -p data/raw data/augmented data/train data/test outputs
python Data_Augment.py data/raw/math.json data/augmented/math.jsonl
```

Place your source JSON at `data/raw/math.json` first. Configure `AI_MODEL_NAME` in `Data_Augment.py` with a model available to your Together AI account. The checked-in model identifier is an experiment default, not a guarantee of current availability. The script sends each problem and solution to that API.

Output is opened in append mode. Re-running against the same output file appends records and does not automatically skip previously processed examples.

Convert JSONL to a JSON array before translation or SFT. This example also creates a file suitable for the GRPO loader:

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path("data/augmented/math.jsonl")
destination = Path("data/train/math.json")
records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
PY
```

### 2. Translate data (optional)

Edit these constants in `Translation.py`:

| Constant | Example value / purpose |
| --- | --- |
| `INPUT_DIR` | `"data/train"`; reads all `*.json` files |
| `OUTPUT_FILE` | `"outputs/translated_math_data.jsonl"` |
| `SRC_LANG` | `"eng_Latn"` |
| `TGT_LANG` | `"mar_Deva"` for Marathi or `"hin_Deva"` for Hindi |
| `MODEL_CKPT` | `"ai4bharat/indictrans2-en-indic-1B"` |

Create the output directory before running:

```bash
mkdir -p outputs
python Translation.py
```

Only `question` and `step_wise_answer` are translated; `answer` is copied unchanged. Convert the resulting JSONL to a JSON array using the preceding snippet with updated paths.

**Before using translated data for GRPO, restore and verify numbered step boundaries.** The translator joins translated sentences with spaces, so its output does not preserve the newline structure required by the staged curriculum. Select the matching Hindi/Marathi marker regex and the commented LaBSE embedder option in `HorizontalAxis_GRPO.py` when using those languages. Check the actual translated marker against the regex.

### 3. Run vertical-axis SFT

In `VerticalAxis_SFT.py`, set:

```python
DATASET_PATH = "data/train/math.json"
OUTPUT_DIR = "outputs/sft"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B"
```

`DATASET_PATH` must point to a **file**, despite the original `"/dataset"` default. The script creates a 90/10 train/validation split with seed 42. Use enough examples for both splits.

```bash
python VerticalAxis_SFT.py
```

The final adapter and tokenizer are saved to `outputs/sft/final_adapter/`; intermediate checkpoints are saved under the configured output directory.

### 4. Run horizontal-axis GRPO

In the separate GRPO environment, configure `HorizontalAxis_GRPO.py`:

```python
DATASET_DIR = "data/train"
SFT_CHECKPOINT_PATH = "outputs/sft/final_adapter"
OUTPUT_DIR = "outputs/grpo"
```

Also change `output_dir` inside `GRPOConfig` to `OUTPUT_DIR`; it is independently hardcoded to `"/output"` in the supplied script. Confirm your installed Unsloth version can load the selected SFT adapter and its base model.

```bash
python HorizontalAxis_GRPO.py
```

The script reads `*.json` files, constructs staged continuations, trains with a weighted sampler, and saves LoRA weights with `model.save_lora(OUTPUT_DIR)`. Checkpoints use the `GRPOConfig.output_dir` setting. Although `ADAPTER_PATH` is defined, it is not used by the final save call.

Each invocation performs one GRPO run. Any repeated SFT/GRPO schedule, language ordering, or checkpoint handoff must be arranged explicitly; the repository does not automate the paper's full experiment schedule.

### 5. Generate test responses

Set the following in `Inference.py`:

```python
CKPT_DIR = Path("outputs/sft/final_adapter")
DATA_DIR = Path("data/test")
OUTPUT_PATH = Path("outputs/evaluation_results.json")
BATCH_SIZE = 8
```

For GRPO evaluation, replace `CKPT_DIR` with a PEFT-compatible saved adapter directory. The directory must contain an adapter configuration and weights readable by `PeftModel.from_pretrained`; verify the Unsloth export format before using it here.

```bash
python Inference.py
```

The script loads the base model named in the adapter configuration, then applies the adapter. Generation uses sampling with temperature `0.3` and up to `1500` new tokens. Results require a separate scoring procedure to obtain benchmark metrics.

## Default training configuration

These values are read from the supplied scripts, not independently verified paper settings.

| Setting | Vertical-axis SFT | Horizontal-axis GRPO |
| --- | --- | --- |
| Model initialization | `Qwen/Qwen2.5-Math-7B` | Configured SFT checkpoint |
| LoRA configuration | Rank 16, alpha 64, dropout 0.05 | Loader maximum LoRA rank 64; adapter settings come from checkpoint |
| Learning rate | `3e-5` | `5e-6` |
| Epochs | 3 | 1 |
| Per-device batch size | 2 | 1 |
| Gradient accumulation | 16 | 4 |
| Optimizer | `adamw_torch` | `adamw_8bit` |
| Scheduler / warmup ratio | Cosine / 0.1 | Cosine / 0.1 |
| Sequence settings | `max_seq_length=1500` | Loader length 2048; prompt 1200; completion 1024 |
| Generations per prompt | — | 4 |
| Checkpoint interval | 50 steps | 100 steps |
| Reporting | Disabled (`report_to="none"`) | TensorBoard |

The GRPO script additionally passes `generation_kwargs={"max_tokens": 1200, "temperature": 0.6}`. Check how your installed backend resolves those settings relative to `max_completion_length` and the model context limit before a long run.

## Compatibility and reproducibility

The original package lockfile, GPU configuration, and full experiment artifacts were not supplied. Dependency files were assembled from imports and backend usage; **an end-to-end GPU run has not been validated for this release**.

- **TRL APIs:** SFT uses legacy `tokenizer`, `max_seq_length`, and `packing` arguments. Its dependency profile selects [TRL 0.11.4](https://huggingface.co/docs/trl/v0.11.4/en/sft_trainer) as an API starting point. GRPO uses a different interface, custom dataloader behavior, and Unsloth integration. Installing the latest GRPO packages alone does not guarantee compatibility. Recover the experiment's package versions or validate and adapt the trainer calls on your target GPU.
- **GRPO grouping:** verify that the custom sampler/dataloader and `num_generations=4` satisfy the installed trainer's prompt grouping and batch-size requirements. The dataloader override bypasses the trainer's default sampling behavior.
- **Precision and memory:** SFT enables BF16 whenever CUDA is available, and inference explicitly loads BF16 weights. Confirm BF16 support or adjust those settings. SFT loads the base model without an explicit quantization configuration despite calling `prepare_model_for_kbit_training`.
- **Prompt formatting:** SFT requires a tokenizer chat template; verify that the chosen model supplies the template used in your experiment. Inference uses a separate `### Question` / `### Answer` prompt format.
- **Reward interpretation:** correctness is based on numeric answer extraction and a string match in the response's final 200 characters. It is not a symbolic equivalence checker. Validate it for the answer types in your benchmark.
- **Randomness:** the SFT split has a fixed seed, but augmentation, curriculum sampling, and inference include stochastic behavior. Record seeds, model revisions, data splits, and generation settings for reproducibility.

After validating each environment, save its resolved versions alongside your experiment records:

```bash
python -m pip freeze > requirements-lock.txt
```

Use a distinct lockfile for each environment, and review it for local paths or private package URLs before publishing it.

