# Qwen local CPU inference for MCQ benchmark CSVs
# Run this in a fresh Python process/notebook kernel if possible, so CPU thread env vars apply cleanly.

import os

# Force this process to ignore GPUs and use CPU thread pools.
# These are process-local; they will not affect other nodes/processes using the same venv.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_CPU_THREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _CPU_THREADS)

import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _configure_torch_cpu(cpu_threads=None):
    """Use all available CPU cores unless cpu_threads is explicitly provided."""
    threads = int(cpu_threads or os.cpu_count() or 1)

    # Keep environment variables aligned even if the function is called manually.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

    torch.set_num_threads(threads)

    # Inter-op threads can only be set before PyTorch parallel work starts.
    # If the notebook already used PyTorch, this may raise RuntimeError; safe to ignore.
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 2)))
    except RuntimeError:
        pass

    return threads


def run_qwen_local_on_questions(
    input_csv,
    output_csv,
    prompt_col="Prompt",
    model_id="Qwen/Qwen3.6-35B-A3B",
    question_id_col=None,
    gold_col=None,
    level_col="level",
    medical_specialty_col="medical_specialty",
    max_retries=5,
    save_every=1,
    max_new_tokens=10,
    cache_dir="/data/oussama/hf_cache",
    cpu_threads=None,
    torch_dtype="auto",
    enable_thinking=False,
):
    """
    Run Qwen locally on CPU using Hugging Face Transformers and save predictions.

    Output CSV columns:
    - question_id
    - gold
    - prediction
    - raw_output
    - correct
    - level
    - medical_specialty
    - model
    - stop_reason
    - refusal_category
    - refusal_type

    Notes:
    - This forces CPU usage in this process. It will not touch other jobs using the same venv.
    - Qwen 30B/35B models on CPU require a lot of RAM and will be slow.
      Test first with a small Qwen model if you only want to verify the pipeline.
    """

    used_threads = _configure_torch_cpu(cpu_threads)
    device = torch.device("cpu")

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"CPU only: {device}")
    print(f"PyTorch CPU threads: {torch.get_num_threads()} / requested: {used_threads}")
    print(f"Loading local Qwen model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        device_map=None,          # important: do not auto-place on GPU
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    qwen_model.to(device)
    qwen_model.eval()

    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if prompt_col not in df.columns:
        raise ValueError(f"Missing prompt column: {prompt_col}")

    if gold_col is None:
        raise ValueError(
            "Could not find the gold answer column. "
            "Please pass gold_col='your_column_name'."
        )

    output_columns = [
        "question_id",
        "gold",
        "prediction",
        "raw_output",
        "correct",
        "level",
        "medical_specialty",
        "model",
        "stop_reason",
        "refusal_category",
        "refusal_type",
    ]

    def clean_value(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    def normalize_gold(value):
        text = clean_value(value).upper()
        match = re.search(r"\b([A-F])\b", text)
        return match.group(1) if match else text[:1]

    def extract_prediction(raw_output, allowed_letters=None):
        raw_output = clean_value(raw_output).upper()

        if allowed_letters:
            allowed_letters = [x.upper() for x in allowed_letters]
        else:
            allowed_letters = ["A", "B", "C", "D", "E", "F"]

        if raw_output in allowed_letters:
            return raw_output

        pattern = r"\b(" + "|".join(map(re.escape, allowed_letters)) + r")\b"
        match = re.search(pattern, raw_output)
        if match:
            return match.group(1)

        if raw_output and raw_output[0] in allowed_letters:
            return raw_output[0]

        return ""

    def parse_qwen_output(decoded_text):
        """Remove Qwen thinking/special tokens and keep the answer text."""
        text = clean_value(decoded_text)

        # Qwen3 may emit thinking tags depending on model/template/version.
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove common chat/special tokens if they appear in decoded text.
        for token in [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<｜Assistant｜>",
            "<｜User｜>",
            "<｜end▁of▁sentence｜>",
        ]:
            text = text.replace(token, " ")

        return clean_value(text)

    def make_chat_text(messages):
        """Use Qwen chat template; fall back for older Qwen tokenizers without enable_thinking."""
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def call_model(prompt, allowed_letters=None, max_retries=5):
        """
        Calls Qwen locally on CPU.

        Returns:
        {
            "raw_output": str,
            "stop_reason": str,
            "refusal_category": None,
            "refusal_type": None,
        }
        """

        if allowed_letters is None:
            allowed_letters = ["A", "B", "C", "D", "E", "F"]

        allowed_text = ", ".join(allowed_letters)

        system_prompt = (
            "You are evaluating multiple-choice questions from a medical benchmark. "
            "Select the single best answer option. "
            "Return only one uppercase letter. "
            "Do not explain."
        )

        # /no_think is an extra Qwen3 safeguard; enable_thinking=False is also passed above.
        user_prompt = (
            f"{prompt}\n\n"
            f"Allowed options: {allowed_text}\n"
            "Return exactly one uppercase letter from the allowed options. "
            "No explanation.\n"
            "/no_think"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                chat_text = make_chat_text(messages)

                inputs = tokenizer(
                    chat_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    outputs = qwen_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                generated_tokens = outputs[0][input_len:]

                decoded = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=False,
                )

                raw_output = parse_qwen_output(decoded)

                if raw_output:
                    return {
                        "raw_output": raw_output,
                        "stop_reason": "local_generate_cpu",
                        "refusal_category": None,
                        "refusal_type": None,
                    }

                print("Empty local Qwen response:")
                print("decoded:", repr(decoded))

                raise RuntimeError("Qwen returned empty text output.")

            except Exception as e:
                last_error = e

                if attempt == max_retries:
                    break

                wait_time = min(2 ** attempt, 60)
                print(
                    f"Local Qwen error attempt {attempt}/{max_retries}: {repr(e)}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed after {max_retries} retries. Last error: {repr(last_error)}"
        )

    # Read already processed question IDs if results CSV exists
    processed_ids = set()

    if output_csv.exists():
        old_results = pd.read_csv(output_csv, encoding="utf-8-sig")

        if "question_id" in old_results.columns:
            processed_ids = set(old_results["question_id"].astype(str))
        elif "Question_id" in old_results.columns:
            processed_ids = set(old_results["Question_id"].astype(str))

        for col in output_columns:
            if col not in old_results.columns:
                old_results[col] = ""

        old_results = old_results[output_columns]
        results = old_results.to_dict("records")
    else:
        results = []

    print(f"Already processed: {len(processed_ids)} questions")

    newly_processed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question_id = (
            clean_value(row[question_id_col])
            if question_id_col is not None
            else str(idx)
        )

        if str(question_id) in processed_ids:
            continue

        gold = normalize_gold(row[gold_col])

        group = clean_value(row["Group"]).upper() if "Group" in df.columns else "ABCDEF"

        allowed_letters = [
            letter for letter in group
            if letter in ["A", "B", "C", "D", "E", "F"]
        ]

        if not allowed_letters:
            allowed_letters = ["A", "B", "C", "D", "E", "F"]

        prompt = clean_value(row[prompt_col])

        call_result = call_model(
            prompt=prompt,
            allowed_letters=allowed_letters,
            max_retries=max_retries,
        )

        raw_output = call_result["raw_output"]
        stop_reason = call_result["stop_reason"]
        refusal_category = call_result["refusal_category"]
        refusal_type = call_result["refusal_type"]

        prediction = extract_prediction(raw_output, allowed_letters)
        correct = int(prediction == gold)

        result = {
            "question_id": question_id,
            "gold": gold,
            "prediction": prediction,
            "raw_output": raw_output,
            "correct": correct,
            "level": clean_value(row[level_col]) if level_col in df.columns else "",
            "medical_specialty": (
                clean_value(row[medical_specialty_col])
                if medical_specialty_col in df.columns
                else ""
            ),
            "model": model_id,
            "stop_reason": stop_reason,
            "refusal_category": refusal_category,
            "refusal_type": refusal_type,
        }

        results.append(result)
        processed_ids.add(str(question_id))
        newly_processed += 1

        if save_every and newly_processed % save_every == 0:
            pd.DataFrame(results, columns=output_columns).to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
            )

    results_df = pd.DataFrame(results, columns=output_columns)

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return results_df
