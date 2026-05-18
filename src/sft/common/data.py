"""Data loading and preprocessing for DeepMath-103K SFT."""

import os
import pandas as pd
from typing import Optional
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
) -> tuple[Dataset, Dataset]:
    """
    Load filtered parquet files, apply chat template, tokenize, split.

    Args:
        tokenizer: Qwen3 tokenizer with chat_template.
        data_dir: path to filtered/*.parquet directory.
        max_seq_length: max token length for truncation.
        system_prompt: optional system message.
        val_split: fraction for validation.
        seed: random seed for split.
        num_files: limit number of parquet files (None = all).

    Returns:
        (train_dataset, val_dataset) — both tokenized with input_ids, attention_mask, labels.
    """
    parquet_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    )
    if num_files:
        parquet_files = parquet_files[:num_files]

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(os.path.join(data_dir, f))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[Data] loaded {len(df_all)} samples from {len(parquet_files)} file(s)")

    dataset = Dataset.from_pandas(df_all[["question", "solution"]])

    dataset = dataset.map(
        lambda batch: _preprocess_batch(batch, tokenizer, max_seq_length, system_prompt),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    dataset = dataset.train_test_split(test_size=val_split, seed=seed)
    return dataset["train"], dataset["test"]


def _preprocess_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    system_prompt: Optional[str],
) -> dict:
    """Tokenize a batch and create labels with user/system parts masked."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        solution = examples["solution"][i]

        messages = _build_messages(question, solution, system_prompt)

        # Full text: system + user + assistant
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Prompt-only text: system + user + assistant header (for label masking)
        prompt_messages = messages[:-1]  # drop assistant
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_tokens = tokenizer(
            full_text, truncation=True, max_length=max_seq_length, padding=False
        )
        prompt_tokens = tokenizer(
            prompt_text, truncation=True, max_length=max_seq_length, padding=False
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        prefix_len = len(prompt_tokens["input_ids"])
        labels = [-100] * prefix_len + input_ids[prefix_len:]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

    return model_inputs


def _build_messages(
    question: str,
    solution: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Build messages list for Qwen3 chat template."""
    messages = []
    response = f"{solution}<|endoftext|>"
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})
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

        # Pad to max length in batch
        max_len = max(len(ids) for ids in batch["input_ids"])
        max_len = min(max_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for key in batch:
            padded = []
            for seq in batch[key]:
                seq = seq[:max_len]
                if key == "labels":
                    pad_value = -100
                elif key == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = pad_id
                padded.append(seq + [pad_value] * (max_len - len(seq)))
            batch[key] = torch.tensor(padded, dtype=torch.long)

        return batch
