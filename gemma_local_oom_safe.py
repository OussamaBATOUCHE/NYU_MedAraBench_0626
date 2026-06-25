"""
OOM-safer local Gemma runner for MCQ CSV evaluation.

Main changes vs the original:
- reserves GPU memory with max_memory so generation has space
- uses float16 by default for Tesla P100 / older GPUs
- supports CPU/disk offload
- disables KV cache during generation to reduce memory
- uses max_new_tokens=5 by default for MCQ letters
- clears CUDA cache after each question
- includes all missing imports
"""

import os

# Must be set before torch is heavily used. In notebooks, run this file after a kernel restart.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForMultimodalLM


def run_gemma4_local_on_questions(
    input_csv,
    output_csv,
    prompt_col="Prompt",
    model_id="google/gemma-3-27b-it",
    question_id_col=None,
    gold_col=None,
    level_col="level",
    medical_specialty_col="medical_specialty",
    max_retries=2,
    save_every=1,
    max_new_tokens=5,
    cache_dir="/data/oussama/hf_cache",
    offload_folder="/data/oussama/offload_gemma",
    gpu_memory="13GiB",
    cpu_memory="160GiB",
    torch_dtype=torch.float16,
    use_cache=False,
    clear_cache_each_question=True,
    trust_remote_code=True,
):
    """
    Run Gemma locally using Transformers and save predictions.

    This version is safer for 4 x 16GB GPUs because it does not allow
    Transformers to fill each GPU completely during model loading.

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
    """

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    Path(offload_folder).mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Select the correct Jupyter kernel / venv before running Gemma."
        )

    n_gpus = torch.cuda.device_count()
    max_memory = {i: gpu_memory for i in range(n_gpus)}
    max_memory["cpu"] = cpu_memory

    print(f"Loading local Gemma model: {model_id}")
    print(f"GPUs visible: {n_gpus}")
    print(f"max_memory: {max_memory}")
    print(f"dtype: {torch_dtype}")
    print(f"offload_folder: {offload_folder}")

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    # Newer Transformers prefers `dtype`; older versions use `torch_dtype`.
    common_load_kwargs = dict(
        cache_dir=cache_dir,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    try:
        gemma_model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            **common_load_kwargs,
        )
    except TypeError:
        gemma_model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **common_load_kwargs,
        )

    gemma_model.eval()

    print("Model loaded.")
    if hasattr(gemma_model, "hf_device_map"):
        print("Device map:")
        print(gemma_model.hf_device_map)

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

    def parse_gemma_output(decoded_text):
        """Clean generated text and remove common chat/special tokens."""
        text = decoded_text

        if hasattr(processor, "parse_response"):
            try:
                parsed = processor.parse_response(decoded_text)

                if isinstance(parsed, str):
                    text = parsed
                elif isinstance(parsed, dict):
                    text = (
                        parsed.get("content")
                        or parsed.get("answer")
                        or parsed.get("text")
                        or str(parsed)
                    )
                elif isinstance(parsed, list) and len(parsed) > 0:
                    first = parsed[0]
                    if isinstance(first, dict):
                        text = first.get("content") or first.get("text") or str(first)
                    else:
                        text = str(first)
                else:
                    text = str(parsed)
            except Exception:
                text = decoded_text

        text = clean_value(text)

        for token in [
            "<turn|>",
            "<end_of_turn>",
            "<start_of_turn>",
            "<bos>",
            "<eos>",
            "</s>",
        ]:
            text = text.replace(token, " ")

        return clean_value(text)

    def get_input_device():
        """
        For device_map='auto', put input tensors on the first CUDA device used.
        Do not move tensors to CPU/disk even if some layers are offloaded.
        """
        device_map = getattr(gemma_model, "hf_device_map", None)
        if isinstance(device_map, dict):
            cuda_devices = []
            for dev in device_map.values():
                dev = str(dev)
                if dev.isdigit():
                    cuda_devices.append(int(dev))
                elif dev.startswith("cuda:"):
                    cuda_devices.append(int(dev.split(":", 1)[1]))
            if cuda_devices:
                return torch.device(f"cuda:{min(cuda_devices)}")

        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    input_device = get_input_device()
    print(f"Input tensors will be moved to: {input_device}")

    def build_inputs(messages):
        """Apply chat template with fallbacks for different Transformers/Gemma versions."""
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception:
            # Some multimodal processors expect content as text blocks.
            mm_messages = []
            for message in messages:
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                mm_messages.append({"role": message["role"], "content": content})

            return processor.apply_chat_template(
                mm_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )

    def move_inputs_to_device(inputs, device):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def call_model(prompt, allowed_letters=None, max_retries=2):
        """
        Calls Gemma locally.

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

        user_prompt = (
            f"{prompt}\n\n"
            f"Allowed options: {allowed_text}\n"
            "Return exactly one uppercase letter from the allowed options. "
            "No explanation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None

        for attempt in range(1, max_retries + 1):
            inputs = None
            outputs = None
            try:
                if clear_cache_each_question:
                    gc.collect()
                    torch.cuda.empty_cache()

                inputs = build_inputs(messages)
                inputs = move_inputs_to_device(inputs, input_device)

                input_len = inputs["input_ids"].shape[-1]

                eos_id = None
                if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                    eos_id = processor.tokenizer.eos_token_id

                with torch.inference_mode():
                    outputs = gemma_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=use_cache,
                        pad_token_id=eos_id,
                    )

                generated_tokens = outputs[0][input_len:]

                if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                    decoded = processor.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=False,
                    )
                else:
                    decoded = processor.decode(
                        generated_tokens,
                        skip_special_tokens=False,
                    )

                raw_output = parse_gemma_output(decoded)

                if raw_output:
                    return {
                        "raw_output": raw_output,
                        "stop_reason": "local_generate",
                        "refusal_category": None,
                        "refusal_type": None,
                    }

                print("Empty local Gemma response:")
                print("decoded:", repr(decoded))
                raise RuntimeError("Gemma returned empty text output.")

            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                print(f"CUDA OOM attempt {attempt}/{max_retries}: {e}")
                print(
                    "Suggestion: restart kernel and rerun with gpu_memory='12GiB' "
                    "or lower, and keep max_new_tokens=5."
                )
                del inputs, outputs
                gc.collect()
                torch.cuda.empty_cache()

                # Retrying OOM without changing memory rarely helps. Stop early.
                break

            except Exception as e:
                last_error = e
                del inputs, outputs
                gc.collect()
                torch.cuda.empty_cache()

                if attempt == max_retries:
                    break

                wait_time = min(2 ** attempt, 60)
                print(
                    f"Local Gemma error attempt {attempt}/{max_retries}: {repr(e)}. "
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
        old_results = pd.DataFrame(columns=output_columns)
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

        if clear_cache_each_question:
            gc.collect()
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results, columns=output_columns)

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return results_df
