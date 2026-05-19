"""Data loading and preprocessing for SFT with optional general data mixing."""

import json
import os
from typing import Optional

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def load_and_prepare_data(
    tokenizer: PreTrainedTokenizer,
    data_dir: str,
    max_seq_length: int = 2048,
    system_prompt: Optional[str] = None,
    val_split: float = 0.02,
    seed: int = 42,
    num_files: Optional[int] = None,
    general_data_path: Optional[str] = None,
    general_mix_ratio: float = 0.2,
) -> tuple[Dataset, Dataset]:
    """
    Load filtered parquet files, optionally mix with general domain data.

    Args:
        tokenizer: Qwen3 tokenizer with chat_template.
        data_dir: path to filtered/*.parquet directory (DeepMath data).
        max_seq_length: max token length for truncation.
        system_prompt: optional system message.
        val_split: fraction for validation.
        seed: random seed for split.
        num_files: limit number of parquet files (None = all).
        general_data_path: path to preprocessed general data (parquet or jsonl).
        general_mix_ratio: fraction of general data in final mix (0.0 ~ 1.0).

    Returns:
        (train_dataset, val_dataset)
    """
    # ── 1. Load DeepMath data ──
    math_dataset = _load_math_data(data_dir, num_files, tokenizer, max_seq_length, system_prompt)
    print(f"[Data] DeepMath: {len(math_dataset)} samples")

    # ── 2. Load general data (optional) ──
    if general_data_path and general_mix_ratio > 0:
        general_dataset = _load_general_data(general_data_path, tokenizer, max_seq_length, system_prompt)
        print(f"[Data] General:  {len(general_dataset)} samples")

        # Compute target counts for desired ratio
        total_math = len(math_dataset)
        target_total = int(total_math / (1 - general_mix_ratio))
        target_general = target_total - total_math

        if len(general_dataset) < target_general:
            print(f"[Data] Warning: only {len(general_dataset)} general samples available, "
                  f"need {target_general} for {general_mix_ratio:.0%} mix. Using all available.")
            target_general = len(general_dataset)

        if len(general_dataset) > 0:
            general_dataset = general_dataset.shuffle(seed=seed).select(range(target_general))
            combined = concatenate_datasets([math_dataset, general_dataset])
            combined = combined.shuffle(seed=seed)
            actual_ratio = target_general / len(combined)
            print(f"[Data] Combined: {len(combined)} total "
                  f"(math={total_math}, general={target_general}, ratio={actual_ratio:.1%})")
        else:
            combined = math_dataset
            print(f"[Data] No general data available, using math only")
    else:
        combined = math_dataset

    # ── 3. Split ──
    split = combined.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


def _load_math_data(
    data_dir: str,
    num_files: Optional[int],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
) -> Dataset:
    """Load DeepMath-103K filtered parquet files."""
    parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    if num_files:
        parquet_files = parquet_files[:num_files]

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(os.path.join(data_dir, f))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    dataset = Dataset.from_pandas(df_all[["question", "solution"]])
    dataset = dataset.map(
        lambda batch: _preprocess_math_batch(batch, tokenizer, max_seq_length, system_prompt),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing math",
    )
    return dataset


def _load_general_data(
    path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
) -> Dataset:
    """Load preprocessed general data (parquet or jsonl)."""
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path}")

    # The preprocessed data should have a 'messages' column (JSON string of chat messages)
    if "messages" in df.columns:
        if isinstance(df["messages"].iloc[0], str):
            df["messages"] = df["messages"].apply(json.loads)

        records = []
        for _, row in df.iterrows():
            msgs = row["messages"]
            token_count = row.get("token_count", None)
            # Already token-count filtered during preprocessing, but re-check if needed
            records.append({"messages": msgs, "token_count": token_count})

        dataset = Dataset.from_list(records)
    elif "text" in df.columns:
        # Fallback: already has text column
        dataset = Dataset.from_pandas(df[["text"]])
        dataset = dataset.map(
            lambda batch: _tokenize_from_text_batch(batch, tokenizer, max_seq_length),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing general",
        )
        return dataset
    else:
        raise ValueError(f"Expected 'messages' or 'text' column in {path}, got {df.columns.tolist()}")

    # Tokenize from messages
    dataset = dataset.map(
        lambda batch: _preprocess_messages_batch(batch, tokenizer, max_seq_length, system_prompt),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing general",
    )
    return dataset


# ── Preprocessing helpers ────────────────────────────────────────────

def _preprocess_math_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
) -> dict:
    """Tokenize DeepMath batch and mask prompt tokens."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        solution = examples["solution"][i]
        messages = _build_messages(question, solution, system_prompt)
        ids, mask, labels = _tokenize_with_mask(messages, tokenizer, max_seq_length)
        model_inputs["input_ids"].append(ids)
        model_inputs["attention_mask"].append(mask)
        model_inputs["labels"].append(labels)

    return model_inputs


def _preprocess_messages_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
) -> dict:
    """Tokenize general data batch (already in messages format)."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]

        # If system prompt is provided and messages don't have one, prepend it
        if system_prompt and messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

        ids, mask, labels = _tokenize_with_mask(messages, tokenizer, max_seq_length)
        model_inputs["input_ids"].append(ids)
        model_inputs["attention_mask"].append(mask)
        model_inputs["labels"].append(labels)

    return model_inputs


def _tokenize_from_text_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
) -> dict:
    """Fallback: tokenize raw text."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for text in examples["text"]:
        tok = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
        model_inputs["input_ids"].append(tok["input_ids"])
        model_inputs["attention_mask"].append(tok["attention_mask"])
        model_inputs["labels"].append(tok["input_ids"][:])  # all tokens are learnable
    return model_inputs


def _tokenize_with_mask(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
) -> tuple[list, list, list]:
    """Tokenize chat messages and mask non-assistant tokens."""
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_tokens = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding=False)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, padding=False)

    prefix_len = len(prompt_tokens["input_ids"])
    labels = [-100] * prefix_len + full_tokens["input_ids"][prefix_len:]

    return full_tokens["input_ids"], full_tokens["attention_mask"], labels


def _build_messages(
    question: str,
    solution: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Build messages list for Qwen3 chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": solution})
    return messages


class DeepMathDataCollator:
    """Data collator that pads to the longest sequence in the batch."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        import torch

        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "labels": [f["labels"] for f in features],
        }

        max_len = max(len(ids) for ids in batch["input_ids"])
        max_len = min(max_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for key in batch:
            pad_value = -100 if key == "labels" else (0 if key == "attention_mask" else pad_id)
            batch[key] = torch.tensor(
                [seq[:max_len] + [pad_value] * (max_len - len(seq[:max_len])) for seq in batch[key]],
                dtype=torch.long,
            )

        return batch
