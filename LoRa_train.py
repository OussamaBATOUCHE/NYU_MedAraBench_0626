#!/usr/bin/env python3
# ============================================================
# Gemma 4 12B IT LoRA benchmark adaptation - fixed version
#
# Fixes the "always A" problem by:
# 1. Using Gemma chat template via AutoProcessor, like the validation code.
# 2. NEVER changing the content of the Prompt column.
# 3. Evaluating by scoring only the allowed letters from Group.
#    This gives exactly one letter without weird generated text.
#
# What it does:
# - Load CSV
# - Random 90/10 split
# - Test baseline Gemma on the fixed 10%
# - Train lightweight LoRA adapters on the 90%
# - Test each adapted model on the same fixed 10%
# - Save all outputs under lora_results/
# - Save logs under logs/nohop/
# ============================================================

# =========================
# 0. Config
# =========================

import os

# IMPORTANT: set before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["HF_HOME"] = "/data/oussama/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/oussama/hf_cache/hub"
os.environ["HF_HUB_CACHE"] = "/data/oussama/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/data/oussama/hf_cache/transformers"

# Local/offline only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CSV_PATH = "data/cleaned/medarabench_train_with_prompts.csv"
MODEL_NAME_OR_PATH = "google/gemma-4-12B-it"

# If offline model name does not resolve, use exact snapshot path:
# MODEL_NAME_OR_PATH = "/data/oussama/hf_cache/hub/models--google--gemma-4-12B-it/snapshots/<snapshot_hash>"

RESULTS_DIR = "lora_results"
MODELS_DIR = f"{RESULTS_DIR}/models"
LOG_DIR = "logs/lora"
OFFLOAD_DIR = "/data/oussama/offload_gemma_lora"

TEST_SIZE = 0.05
RANDOM_STATE = 42

MAX_LENGTH = 256  # very safe for 4 x 16GB P100 GPUs during QLoRA training

NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# For 4 x 16GB GPUs, leave more headroom for backward/training.
# GPU 0 usually receives extra tensors/activations, so reserve more free memory there.
GPU0_MAX_MEMORY = "5GiB"
OTHER_GPU_MAX_MEMORY = "9GiB"
CPU_MAX_MEMORY = "350GiB"

# 2 LoRA configurations to try
# Safer LoRA sweep for 4 x 16GB GPUs.
# r32 + full MLP targets is often too heavy for Gemma 12B QLoRA.
LORA_RUNS = [
    {"name": "lora_r4_alpha8_lr2e-4", "r": 4, "alpha": 8, "dropout": 0.05, "lr": 2e-4},
    {"name": "lora_r8_alpha16_lr2e-4", "r": 8, "alpha": 16, "dropout": 0.05, "lr": 2e-4},
]

# Base LoRA target suffixes. The script will later expand these to full module names
# and exclude vision/multimodal modules to save VRAM.
LORA_TARGET_SUFFIXES = ["q_proj", "v_proj"]


# =========================
# 1. Imports and folders
# =========================

import re
import gc
import json
import time
import logging
from pathlib import Path

import torch
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

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

log_file = Path(LOG_DIR) / f"gemma4_lora_fixed_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

logger.info(f"Logging to: {log_file}")
logger.info(f"Using model class: {MODEL_CLASS_NAME}")
logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
logger.info(f"torch cuda available: {torch.cuda.is_available()}")
logger.info(f"visible cuda device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")


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

    max_memory = {}

    for i in range(torch.cuda.device_count()):
        if i == 0:
            max_memory[i] = GPU0_MAX_MEMORY
        else:
            max_memory[i] = OTHER_GPU_MAX_MEMORY

    max_memory["cpu"] = CPU_MAX_MEMORY
    return max_memory


def find_text_lora_targets(model):
    """
    Return full module names for text/LLM LoRA targets only.

    Why this matters:
    Gemma 4 is loaded through a multimodal/image-text class. If we pass only
    ["q_proj", "v_proj"], PEFT may attach LoRA to every matching module,
    including vision/multimodal parts. For a text-only MCQ task, that wastes
    VRAM and can cause OOM.
    """
    bad_keywords = [
        "vision",
        "image",
        "mm",
        "multi_modal",
        "multimodal",
        "projector",
        "connector",
        "visual",
    ]

    targets = []
    for name, module in model.named_modules():
        lname = name.lower()

        if not any(name.endswith(suffix) for suffix in LORA_TARGET_SUFFIXES):
            continue

        if any(bad in lname for bad in bad_keywords):
            continue

        targets.append(name)

    if not targets:
        raise RuntimeError(
            "No text LoRA targets found. Inspect model.named_modules() for q_proj/v_proj names."
        )

    logger.info(f"Using {len(targets)} text-only LoRA target modules.")
    logger.info("First 30 LoRA targets:")
    for t in targets[:30]:
        logger.info(f"  {t}")

    return targets


def get_input_device(model):
    """
    For device_map='auto', input_ids must be placed on the same device
    as the input embedding layer.

    Important fix:
    The older version used the first CUDA device, usually cuda:0.
    But with multi-GPU sharding, Gemma's embedding layer may be on cuda:3.
    If input_ids are on cuda:0 and embeddings are on cuda:3, PyTorch crashes with:
    "Expected all tensors to be on the same device, but found cuda:3 and cuda:0".
    """
    try:
        emb_device = model.get_input_embeddings().weight.device
        logger.info(f"Input embedding device: {emb_device}")
        return emb_device
    except Exception as e:
        logger.warning(f"Could not read input embedding device: {repr(e)}")

    if torch.cuda.is_available():
        return torch.device("cuda:0")

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
        raise RuntimeError("processor.tokenizer is None. Gemma processor did not expose a tokenizer.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return processor, tokenizer


def build_chat_inputs(processor, prompt, device=None):
    """
    Build Gemma instruction/chat inputs.

    Important:
    - The content of Prompt is not changed.
    - We only wrap it in Gemma's official chat template.
    """
    prompt = str(prompt)

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        # Some multimodal processors expect text blocks.
        mm_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
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
    """
    Return only input_ids for chat-formatted Prompt.
    Used for training and log-probability scoring.
    """
    inputs = build_chat_inputs(processor, prompt, device=None)
    return inputs["input_ids"][0].tolist()


# =========================
# 4. Model loading
# =========================

def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    common_kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=get_max_memory(),
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )

    try:
        model = ModelClass.from_pretrained(
            MODEL_NAME_OR_PATH,
            dtype=torch.float16,
            **common_kwargs,
        )
    except TypeError:
        model = ModelClass.from_pretrained(
            MODEL_NAME_OR_PATH,
            torch_dtype=torch.float16,
            **common_kwargs,
        )

    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning(f"Could not enable gradient checkpointing: {repr(e)}")

    if hasattr(model, "hf_device_map"):
        logger.info(f"Device map: {model.hf_device_map}")

    return model


# =========================
# 5. Exact one-letter prediction by scoring allowed letters
# =========================

@torch.inference_mode()
def score_letter(model, processor, tokenizer, prompt, letter, input_device):
    """
    Score one candidate letter as the continuation after the chat-formatted Prompt.
    We score variants because tokenizers may represent the answer as:
    'A', ' A', or '\\nA'.
    """
    model.eval()

    prefix_ids = build_chat_prefix_ids(processor, prompt)

    variants = [
        letter,
        " " + letter,
        "\n" + letter,
    ]

    best_score = None

    for variant in variants:
        cand_ids = tokenizer(
            variant,
            add_special_tokens=False,
        )["input_ids"]

        if not cand_ids:
            continue

        input_ids = torch.tensor(
            [prefix_ids + cand_ids],
            dtype=torch.long,
            device=input_device,
        )

        attention_mask = torch.ones_like(input_ids, device=input_device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits

        prefix_len = len(prefix_ids)

        # logits at position t predict token t+1
        # candidate tokens are positions prefix_len .. end
        log_probs = torch.log_softmax(
            logits[:, prefix_len - 1 : -1, :],
            dim=-1,
        )

        target_ids = input_ids[:, prefix_len:]

        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Use average log-probability so variants with 2 tokens are not unfairly penalized.
        score = token_log_probs.mean().item()

        if best_score is None or score > best_score:
            best_score = score

    if best_score is None:
        best_score = float("-inf")

    return best_score


def predict_one_letter(model, processor, tokenizer, prompt, allowed_letters, input_device):
    """
    Return exactly one letter by comparing model scores for allowed letters.
    No free-form generation, so no 'Your answer:', no repeated prompt, no accidental A from Answer.
    """
    scores = {}

    for letter in allowed_letters:
        scores[letter] = score_letter(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prompt=prompt,
            letter=letter,
            input_device=input_device,
        )

    prediction = max(scores, key=scores.get)

    # Keep raw_output as exactly one letter.
    raw_output = prediction

    return prediction, raw_output, scores


def evaluate_model(model, processor, tokenizer, df_eval, model_name, save_path):
    """
    Evaluate model on df_eval and save row-by-row.

    Important behavior:
    - If save_path already exists and has exactly len(df_eval) rows, skip evaluation.
    - If save_path exists but is incomplete, resume from the rows/questions already saved.
    - Save immediately after every row, so you can monitor the CSV while the job is running.
    """
    input_device = get_input_device(model)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    expected_n = len(df_eval)

    # If full results already exist, skip evaluation
    if save_path.exists():
        try:
            existing = pd.read_csv(save_path, encoding="utf-8-sig")
            if len(existing) == expected_n:
                logger.info(
                    f"Skipping {model_name}: existing complete file found "
                    f"with {len(existing)}/{expected_n} rows: {save_path}"
                )

                if "correct" in existing.columns:
                    acc = existing["correct"].astype(int).mean()
                elif "prediction" in existing.columns and "gold" in existing.columns:
                    acc = accuracy_score(existing["gold"], existing["prediction"])
                else:
                    acc = float("nan")

                logger.info(f"{model_name} accuracy from existing file: {acc:.4f}")
                return existing, acc

            logger.info(
                f"Found incomplete existing file for {model_name}: "
                f"{len(existing)}/{expected_n} rows. Will resume."
            )
        except Exception as e:
            logger.warning(f"Could not read existing result file {save_path}: {repr(e)}. Starting fresh.")
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    # Track already processed rows. Prefer Question Number if available.
    processed_keys = set()
    records = []

    if not existing.empty:
        records = existing.to_dict("records")

        if "Question Number" in existing.columns:
            processed_keys = set(existing["Question Number"].astype(str))
        else:
            # Fallback: processed dataframe positions
            processed_keys = set(existing.index.astype(str))

    preds = []
    golds = []

    # Use existing predictions for final accuracy if resuming
    if not existing.empty and "prediction" in existing.columns and "gold" in existing.columns:
        preds.extend(existing["prediction"].astype(str).tolist())
        golds.extend(existing["gold"].astype(str).tolist())

    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"Evaluating {model_name}"):
        if "Question Number" in df_eval.columns:
            row_key = str(row["Question Number"])
        else:
            row_key = str(idx)

        if row_key in processed_keys:
            continue

        prompt = clean_value(row["Prompt"])
        allowed_letters = get_allowed_letters(row)

        pred, raw, scores = predict_one_letter(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prompt=prompt,
            allowed_letters=allowed_letters,
            input_device=input_device,
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
        processed_keys.add(row_key)

        # Save immediately after every row.
        # This rewrites the file from records to keep columns consistent and resume-safe.
        pd.DataFrame(records).to_csv(
            save_path,
            index=False,
            encoding="utf-8-sig",
        )

    out = pd.DataFrame(records)

    # Final sanity check
    if len(out) != expected_n:
        logger.warning(
            f"Evaluation file for {model_name} has {len(out)}/{expected_n} rows. "
            f"It may be incomplete."
        )

    if "correct" in out.columns and len(out) > 0:
        acc = out["correct"].astype(int).mean()
    else:
        acc = accuracy_score(golds, preds) if golds else float("nan")

    logger.info(f"{model_name} accuracy: {acc:.4f}")
    logger.info(f"Saved predictions row-by-row to: {save_path}")

    return out, acc

# =========================
# 6. Load data and fixed 90/10 split
# =========================

logger.info(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

required_cols = [
    "Question Number",
    "Question",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "Option E",
    "Option F",
    "Correct Answer",
    "Level",
    "Medical Specialty",
    "Group",
    "Prompt",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["Prompt", "Correct Answer"]).copy()
df["gold"] = df["Correct Answer"].apply(normalize_answer)
df = df[df["gold"].isin(list("ABCDEF"))].copy()

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_split_path = Path(RESULTS_DIR) / f"train_{100-int(TEST_SIZE*100)}_percent.csv"
test_split_path = Path(RESULTS_DIR) / f"test_{int(TEST_SIZE*100)}_percent.csv"

train_df.to_csv(train_split_path, index=False, encoding="utf-8-sig")
test_df.to_csv(test_split_path, index=False, encoding="utf-8-sig")

logger.info(f"Total rows: {len(df)}")
logger.info(f"Train rows: {len(train_df)}")
logger.info(f"Test rows: {len(test_df)}")
logger.info(f"Saved train split to: {train_split_path}")
logger.info(f"Saved test split to: {test_split_path}")


# =========================
# 7. Load processor/tokenizer
# =========================

processor, tokenizer = load_processor_and_tokenizer()
logger.info("Processor and tokenizer loaded.")


# =========================
# 8. Baseline evaluation
# =========================

logger.info("Loading baseline model...")
model = load_base_model()
logger.info("Baseline model loaded.")

baseline_csv = Path(RESULTS_DIR) / f"baseline_gemma4_12b_it_test_{int(TEST_SIZE*100)}_percent.csv"

_, baseline_acc = evaluate_model(
    model=model,
    processor=processor,
    tokenizer=tokenizer,
    df_eval=test_df,
    model_name="gemma4_12b_it_baseline",
    save_path=baseline_csv,
)

del model
clean_memory()


# =========================
# 9. Prepare LoRA training dataset
# =========================

def make_training_example(row):
    """
    Training format:
      input  = exact Prompt column, wrapped only in Gemma chat template
      target = Correct Answer letter

    The Prompt text itself is never changed.
    Loss is masked on the prompt/chat tokens and computed only on the answer.
    """
    prompt = str(row["Prompt"])
    answer = normalize_answer(row["Correct Answer"])

    prefix_ids = build_chat_prefix_ids(processor, prompt)

    answer_ids = tokenizer(
        answer + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:
        # Keep the end because it contains options/question ending and answer target.
        input_ids = input_ids[-MAX_LENGTH:]
        labels = labels[-MAX_LENGTH:]
        attention_mask = attention_mask[-MAX_LENGTH:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_ds = Dataset.from_pandas(
    train_df[["Prompt", "Correct Answer"]].reset_index(drop=True)
)

train_ds = train_ds.map(
    make_training_example,
    remove_columns=train_ds.column_names,
)

# Helpful diagnostics: confirm truncation really happened.
lengths = [len(x) for x in train_ds["input_ids"]]
logger.info(
    f"Training token lengths after truncation: min={min(lengths)}, "
    f"median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}"
)

def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    return {
        "input_ids": pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ),
        "attention_mask": pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        ),
        "labels": pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        ),
    }

logger.info(train_ds)


# =========================
# 10. Train and evaluate 3 LoRA runs
# =========================

summary_rows = [
    {
        "model": "gemma4_12b_it_baseline",
        "adapter_path": "",
        "train_size": 0,
        "test_size": len(test_df),
        "lora_r": "",
        "lora_alpha": "",
        "lora_dropout": "",
        "learning_rate": "",
        "epochs": "",
        "accuracy": baseline_acc,
        "predictions_csv": str(baseline_csv),
    }
]

for cfg in LORA_RUNS:
    run_name = cfg["name"]

    logger.info("=" * 80)
    logger.info(f"Starting LoRA run: {run_name}")
    logger.info(json.dumps(cfg, indent=2))

    run_output_dir = Path(RESULTS_DIR) / f"trainer_{run_name}"
    adapter_dir = Path(MODELS_DIR) / run_name
    run_log_dir = Path(LOG_DIR) / run_name
    pred_csv = Path(RESULTS_DIR) / f"{run_name}_test_{int(TEST_SIZE*100)}_percent.csv"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    clean_memory()
    logger.info("Loading fresh base model for this LoRA run...")
    model = load_base_model()

    logger.info("Preparing model for k-bit LoRA training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_text_lora_targets(model),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        logging_dir=str(run_log_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=cfg["lr"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    clean_memory()
    logger.info(f"Training {run_name}...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to: {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(adapter_dir / "lora_run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"Evaluating adapted model: {run_name}")
    _, acc = evaluate_model(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        df_eval=test_df,
        model_name=run_name,
        save_path=pred_csv,
    )

    summary_rows.append(
        {
            "model": run_name,
            "adapter_path": str(adapter_dir),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "lora_r": cfg["r"],
            "lora_alpha": cfg["alpha"],
            "lora_dropout": cfg["dropout"],
            "learning_rate": cfg["lr"],
            "epochs": NUM_EPOCHS,
            "accuracy": acc,
            "predictions_csv": str(pred_csv),
        }
    )

    del trainer
    del model
    clean_memory()


# =========================
# 11. Save final comparison
# =========================

comparison = pd.DataFrame(summary_rows)
comparison["accuracy"] = comparison["accuracy"].round(4)

comparison_csv = Path(RESULTS_DIR) / "gemma4_12b_it_baseline_vs_3_lora_runs.csv"
comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")

logger.info("=" * 80)
logger.info("DONE")
logger.info(f"Saved comparison to: {comparison_csv}")
logger.info("\n" + comparison.to_string(index=False))

print(comparison)
