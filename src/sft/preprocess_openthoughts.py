"""
Preprocess OpenThoughts-114k (default config, sharded parquet) for SFT mixing.

What it does:
  1. Read sharded parquet files from --input_dir
  2. Convert <|begin_of_thought|> → <think>, <|end_of_thought|> → </think>
  3. Strip <|begin_of_solution|> / <|end_of_solution|>
  4. Convert `from`/`value` conversation format → `role`/`content`
  5. Replace 200-word system prompt with a concise one
  6. Filter by token length (< max_seq_length)
  7. Save as single parquet with `messages` column

Usage:
    python preprocess_openthoughts.py
    python preprocess_openthoughts.py --max_samples 30000 --max_seq_length 2048
"""

import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# ── Tag conversion ───────────────────────────────────────────────────

TAG_REPLACEMENTS = [
    ("<|begin_of_thought|>", "<think>"),
    ("<|end_of_thought|>",   "</think>"),
    ("<|begin_of_solution|>", ""),
    ("<|end_of_solution|>",   ""),
]

OUR_SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Think step by step and output the final answer within \\boxed{}."
)


def convert_tags(text: str) -> str:
    """Replace OpenThoughts special tags with our format."""
    for old, new in TAG_REPLACEMENTS:
        text = text.replace(old, new)
    # Collapse 3+ consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def convert_conversations(convs: list[dict]) -> list[dict]:
    """
    Convert from OpenThoughts `from`/`value` format to `role`/`content`.

    Input:  [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
    Output: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = []
    for turn in convs:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        if role == "system":
            continue  # we'll prepend our own system prompt
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    return messages


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",
                   default="/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/eval/openthoughts")
    p.add_argument("--output_dir",
                   default="/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/OpenThought")
    p.add_argument("--tokenizer_path",
                   default="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total output samples")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load tokenizer ────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ── 2. Read sharded parquet files ────────────────────────────────
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".parquet"))
    print(f"Found {len(files)} parquet files in {args.input_dir}")

    all_records = []
    total_raw = 0
    skipped_tags = 0
    skipped_len = 0

    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        df = pd.read_parquet(fpath)
        total_raw += len(df)
        print(f"  {fname}: {len(df)} rows")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {fname}"):
            # ── Parse conversations ──
            convs = row["conversations"]
            messages = convert_conversations(convs)

            if len(messages) < 2:
                skipped_tags += 1
                continue

            # ── Tag conversion on assistant messages ──
            for msg in messages:
                if msg["role"] == "assistant":
                    msg["content"] = convert_tags(msg["content"])

            # ── Prepend our system prompt ──
            messages = [{"role": "system", "content": OUR_SYSTEM_PROMPT}] + messages

            # ── Tokenize & filter by length ──
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tok(text, truncation=True, max_length=args.max_seq_length + 1)
            if len(tokens["input_ids"]) >= args.max_seq_length:
                skipped_len += 1
                continue

            all_records.append({
                "messages": json.dumps(messages, ensure_ascii=False),
                "token_count": len(tokens["input_ids"]),
            })

            if args.max_samples and len(all_records) >= args.max_samples:
                break

        if args.max_samples and len(all_records) >= args.max_samples:
            break

    print(f"\nRaw: {total_raw}, Kept: {len(all_records)}, "
          f"Skipped (tags): {skipped_tags}, Skipped (len): {skipped_len}")

    # ── 3. Save ──
    out_df = pd.DataFrame(all_records)
    out_path = os.path.join(args.output_dir, "openthoughts_filtered.parquet")
    out_df.to_parquet(out_path, index=False)

    # Also save a small jsonl sample for inspection
    sample_path = os.path.join(args.output_dir, "openthoughts_sample.jsonl")
    sample_df = out_df.sample(n=min(5, len(out_df)), random_state=42)
    with open(sample_path, "w") as f:
        for _, row in sample_df.iterrows():
            f.write(row["messages"] + "\n")
    print(f"Saved: {out_path} ({len(out_df)} rows)")
    print(f"Sample: {sample_path}")

    token_counts = out_df["token_count"]
    print(f"Token count: min={token_counts.min()}, max={token_counts.max()}, "
          f"mean={token_counts.mean():.0f}")


if __name__ == "__main__":
    main()
