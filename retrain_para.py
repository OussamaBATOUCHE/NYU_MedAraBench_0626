#!/usr/bin/env python3
# ============================================================
# Gemma 4 12B IT LoRA benchmark adaptation - DDP Optimized Version
# ============================================================

# =========================
# 0. Config & Cluster Setup
# =========================

import os
import torch

# Setup local rank configurations for DDP
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_main_process = local_rank == 0

# Initialize distributed training context before any heavy imports
if "LOCAL_RANK" in os.environ:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

# IMPORTANT: set before importing transformers/cuda modules
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

os.environ["HF_HOME"] = "/data/oussama/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/oussama/hf_cache/hub"
os.environ["HF_HUB_CACHE"] = "/data/oussama/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/oussama/hf_cache/transformers"

# Local/offline only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CSV_PATH = "data/cleaned/medarabench_train_with_prompts.csv"
MODEL_NAME_OR_PATH = "google/gemma-4-12B-it"

RESULTS_DIR = "lora_results"
MODELS_DIR = f"{RESULTS_DIR}/models"
LOG_DIR = "logs/lora"
OFFLOAD_DIR = "/data/oussama/offload_gemma_lora"

TEST_SIZE = 0.05
RANDOM_STATE = 42
MAX_LENGTH = 1024

NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# Individual GPU maximum usage allocations
GPU_MAX_MEMORY = "13GiB"
CPU_MAX_MEMORY = "350GiB"

# 3 LoRA configurations to try
LORA_RUNS = [
    {
        "name": "lora_r8_alpha16_lr2e-4",
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "lr": 2e-4,
    },
    {
        "name": "lora_r16_alpha32_lr2e-4",
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "lr": 2e-4,
    },
    {
        "name": "lora_r32_alpha64_lr1e-4",
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "lr": 1e-4,
    },
]


# =========================
# 1. Imports and folders
# =========================

import re
import gc
import json
import time
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from transformers import AutoModelForImageTextToText as ModelClass
    MODEL_CLASS_NAME = "AutoModelForImageTextToText"
except Exception:
    try:
        from transformers import AutoModelForMultimodalLM as ModelClass
        MODEL_CLASS_NAME = "AutoModelForMultimodalLM"
    except Exception:
        from transformers import AutoModelForCausalLM as ModelClass
        MODEL_CLASS_NAME = "AutoModelForCausalLM"

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

if is_main_process:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

log_file = Path(LOG_DIR) / f"gemma4_lora_fixed_{time.strftime('%Y%m%d_%H%M%S')}.log"

# Suppress logging spam on worker nodes, only display comprehensively on Main Process
logging.basicConfig(
    level=logging.INFO if is_main_process else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file) if is_main_process else logging.NullHandler(),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

logger.info(f"Logging to: {log_file}")
logger.info(f"Using model class: {MODEL_CLASS_NAME}")
logger.info(f"LOCAL_RANK: {local_rank} | Is Main Process: {is_main_process}")


# =========================
# 2. Utility functions
# =========================

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_answer(x):
    x = clean_value(x).upper()
    m = re.search(r"[A-F]", x)
    return m.group(0) if m else ""


def get_allowed_letters(row):
    group = clean_value(row.get("Group", "")).upper()
    letters = [x for x in group if x in list("ABCDEF")]
    return letters if letters else list("ABCDEF")


def get_max_memory():
    if not torch.cuda.is_available():
        return {"cpu": CPU_MAX_MEMORY}
    # Map specifically to the assigned process rank
    return {local_rank: GPU_MAX_MEMORY, "cpu": CPU_MAX_MEMORY}


def get_input_device(model):
    try:
        emb_device = model.get_input_embeddings().weight.device
        logger.info(f"Input embedding device: {emb_device}")
        return emb_device
    except Exception as e:
        logger.warning(f"Could not read input embedding device: {repr(e)}")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def move_inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }


# =========================
# 3. Processor / chat-template helpers
# =========================

def load_processor_and_tokenizer():
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME_OR_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor.tokenizer is None.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return processor, tokenizer


def build_chat_inputs(processor, prompt, device=None):
    prompt = str(prompt)
    messages = [{"role": "user", "content": prompt}]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        mm_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            mm_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    if device is not None:
        inputs = move_inputs_to_device(inputs, device)

    return inputs


def build_chat_prefix_ids(processor, prompt):
    inputs = build_chat_inputs(processor, prompt, device=None)
    return inputs["input_ids"][0].tolist()


# =========================
# 4. Model loading
# =========================

def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Force individual model shards directly into the isolated local rank GPU
    common_kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": local_rank}, 
        max_memory=get_max_memory(),
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )

    try:
        model = ModelClass.from_pretrained(
            MODEL_NAME_OR_PATH,
            dtype=torch.bfloat16,
            **common_kwargs,
        )
    except TypeError:
        model = ModelClass.from_pretrained(
            MODEL_NAME_OR_PATH,
            torch_dtype=torch.bfloat16,
            **common_kwargs,
        )

    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning(f"Could not enable gradient checkpointing: {repr(e)}")

    return model


# =========================
# 5. Exact one-letter prediction by scoring allowed letters
# =========================

@torch.inference_mode()
def score_letter(model, processor, tokenizer, prompt, letter, input_device):
    model.eval()
    prefix_ids = build_chat_prefix_ids(processor, prompt)
    variants = [letter, " " + letter, "\n" + letter]
    best_score = None

    for variant in variants:
        cand_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not cand_ids:
            continue

        input_ids = torch.tensor([prefix_ids + cand_ids], dtype=torch.long, device=input_device)
        attention_mask = torch.ones_like(input_ids, device=input_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        prefix_len = len(prefix_ids)

        log_probs = torch.log_softmax(logits[:, prefix_len - 1 : -1, :], dim=-1)
        target_ids = input_ids[:, prefix_len:]
        token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        score = token_log_probs.mean().item()

        if best_score is None or score > best_score:
            best_score = score

    return best_score if best_score is not None else float("-inf")


def predict_one_letter(model, processor, tokenizer, prompt, allowed_letters, input_device):
    scores = {}
    for letter in allowed_letters:
        scores[letter] = score_letter(
            model=model, processor=processor, tokenizer=tokenizer,
            prompt=prompt, letter=letter, input_device=input_device,
        )
    prediction = max(scores, key=scores.get)
    return prediction, prediction, scores


def evaluate_model(model, processor, tokenizer, df_eval, model_name, save_path):
    input_device = get_input_device(model)
    save_path = Path(save_path)
    
    if save_path.exists():
        save_path.unlink()

    records, preds, golds = [], [], []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"Evaluating {model_name}"):
        prompt = clean_value(row["Prompt"])
        allowed_letters = get_allowed_letters(row)

        pred, raw, scores = predict_one_letter(
            model=model, processor=processor, tokenizer=tokenizer,
            prompt=prompt, allowed_letters=allowed_letters, input_device=input_device,
        )

        gold = row["gold"]
        correct = pred == gold

        record = row.to_dict()
        record["prediction"] = pred
        record["raw_output"] = raw
        record["letter_scores"] = json.dumps(scores, ensure_ascii=False)
        record["correct"] = correct
        record["model"] = model_name

        records.append(record)
        preds.append(pred)
        golds.append(gold)

        pd.DataFrame([record]).to_csv(
            save_path, mode="a", header=not save_path.exists(), index=False, encoding="utf-8-sig",
        )

    out = pd.DataFrame(records)
    acc = accuracy_score(golds, preds)
    logger.info(f"{model_name} accuracy: {acc:.4f}")
    return out, acc


# =========================
# 6. Load data and fixed 90/10 split
# =========================

logger.info(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

df = df.dropna(subset=["Prompt", "Correct Answer"]).copy()
df["gold"] = df["Correct Answer"].apply(normalize_answer)
df = df[df["gold"].isin(list("ABCDEF"))].copy()

train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True,
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_split_path = Path(RESULTS_DIR) / f"train_{100-int(TEST_SIZE*100)}_percent.csv"
test_split_path = Path(RESULTS_DIR) / f"test_{int(TEST_SIZE*100)}_percent.csv"

if is_main_process:
    train_df.to_csv(train_split_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_split_path, index=False, encoding="utf-8-sig")


# =========================
# 7. Load processor/tokenizer
# =========================

processor, tokenizer = load_processor_and_tokenizer()


# =========================
# 8. Baseline evaluation
# =========================

logger.info("Loading baseline model...")
model = load_base_model()

baseline_csv = Path(RESULTS_DIR) / f"baseline_gemma4_12b_it_test_{int(TEST_SIZE*100)}_percent.csv"
baseline_acc = 0.0

if is_main_process:
    _, baseline_acc = evaluate_model(
        model=model, processor=processor, tokenizer=tokenizer,
        df_eval=test_df, model_name="gemma4_12b_it_baseline", save_path=baseline_csv,
    )

del model
clean_memory()

if torch.distributed.is_initialized():
    torch.distributed.barrier()


# =========================
# 9. Prepare LoRA training dataset
# =========================

def make_training_example(row):
    prompt = str(row["Prompt"])
    answer = normalize_answer(row["Correct Answer"])
    prefix_ids = build_chat_prefix_ids(processor, prompt)

    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[-MAX_LENGTH:]
        labels = labels[-MAX_LENGTH:]
        attention_mask = attention_mask[-MAX_LENGTH:]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_ds = Dataset.from_pandas(train_df[["Prompt", "Correct Answer"]].reset_index(drop=True))
train_ds = train_ds.map(make_training_example, remove_columns=train_ds.column_names)

def collate_fn(batch):
    return {
        "input_ids": pad_sequence([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], batch_first=True, padding_value=0),
        "labels": pad_sequence([torch.tensor(x["labels"], dtype=torch.long) for x in batch], batch_first=True, padding_value=-100),
    }


# =========================
# 10. Train and evaluate 3 LoRA runs
# =========================

summary_rows = [
    {
        "model": "gemma4_12b_it_baseline", "adapter_path": "", "train_size": 0, "test_size": len(test_df),
        "lora_r": "", "lora_alpha": "", "lora_dropout": "", "learning_rate": "", "epochs": "",
        "accuracy": baseline_acc, "predictions_csv": str(baseline_csv),
    }
]

for cfg in LORA_RUNS:
    run_name = cfg["name"]

    logger.info("=" * 80)
    logger.info(f"Starting LoRA run: {run_name}")

    run_output_dir = Path(RESULTS_DIR) / f"trainer_{run_name}"
    adapter_dir = Path(MODELS_DIR) / run_name
    run_log_dir = Path(LOG_DIR) / run_name
    pred_csv = Path(RESULTS_DIR) / f"{run_name}_test_{int(TEST_SIZE*100)}_percent.csv"

    model = load_base_model()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        logging_dir=str(run_log_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=cfg["lr"],
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False, # Essential flag preventing DDP crash loops
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, data_collator=collate_fn,
    )

    trainer.train()

    if is_main_process:
        logger.info(f"Saving LoRA adapter to: {adapter_dir}")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        with open(adapter_dir / "lora_run_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        logger.info(f"Evaluating adapted model: {run_name}")
        _, acc = evaluate_model(
            model=model, processor=processor, tokenizer=tokenizer,
            df_eval=test_df, model_name=run_name, save_path=pred_csv,
        )

        summary_rows.append({
            "model": run_name, "adapter_path": str(adapter_dir), "train_size": len(train_df), "test_size": len(test_df),
            "lora_r": cfg["r"], "lora_alpha": cfg["alpha"], "lora_dropout": cfg["dropout"], "learning_rate": cfg["lr"],
            "epochs": NUM_EPOCHS, "accuracy": acc, "predictions_csv": str(pred_csv),
        })

    del trainer
    del model
    clean_memory()

    # Synchronize all distributed workers before starting the next architecture configuration
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# =========================
# 11. Save final comparison
# =========================

if is_main_process:
    comparison = pd.DataFrame(summary_rows)
    comparison["accuracy"] = comparison["accuracy"].round(4)
    comparison_csv = Path(RESULTS_DIR) / "gemma4_12b_it_baseline_vs_3_lora_runs.csv"
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")

    logger.info("=" * 80)
    logger.info(f"DONE. Saved summary to: {comparison_csv}")
    print(comparison)