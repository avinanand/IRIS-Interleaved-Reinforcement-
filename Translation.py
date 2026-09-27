import json
import os
import glob
import torch
import gc
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIELDS_TO_TRANSLATE = ['question', 'step_wise_answer']
INPUT_DIR = "/dataset"
OUTPUT_FILE = "/output/translated_math_data.jsonl"
MODEL_CKPT = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"
TGT_LANG = "mar_Deva"
#TGT_LANG = "hin_Deva" #Uncomment for Hindi translation


def load_dataset(input_dir: str) -> list[dict]:
    all_data = []
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {input_dir}.")
    
    for file_path in tqdm(json_files, desc="Loading data files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {file_path}. Error: {e}")
    return all_data

def initialize_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CKPT,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = IndicProcessor(inference=True)
    return model, tokenizer, processor

def translate_text(text: str, model, tokenizer, processor) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sentences:
        return ""

    preprocessed_sents = processor.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    
    inputs = tokenizer(
        preprocessed_sents,
        padding="longest",
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
        )
    
    with tokenizer.as_target_tokenizer():
        decoded_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translated_sents = processor.postprocess_batch(decoded_tokens, lang=TGT_LANG)
    return " ".join(translated_sents)

def main():
    model, tokenizer, processor = initialize_model()
    dataset = load_dataset(INPUT_DIR)
    
    if not dataset:
        print("No data loaded. Exiting.")
        return

    print(f"Starting translation of {len(dataset)} items...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in tqdm(dataset, desc="Translating items"):
            translated_item = item.copy()
            for field in FIELDS_TO_TRANSLATE:
                translated_item[field] = translate_text(item.get(field), model, tokenizer, processor)
            
            f_out.write(json.dumps(translated_item, ensure_ascii=False) + "\n")

    # Clean up resources
    del model, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nTranslation complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()