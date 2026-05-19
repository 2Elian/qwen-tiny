#!/usr/bin/env python3
"""
Evaluation script for DeepMath-103K SFT model.

Metrics:
  - EM (Exact Match):         \boxed{...} == final_answer
  - Normalized Match:         after LaTeX unwrap + whitespace normalization
  - SymPy Equivalence:        algebraic equivalence via sympy
  - Think Format Rate:        <think>...</think> tag completeness
  - Pass@k:                   at least 1 of k generations is correct

Stratified by: difficulty bands

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval.py --checkpoint /data1/nuist_llm/TrainLLM/attention-residuals-reproduction/src/sft/output/deepmath-sft-fsdp/checkpoint-500 --num_samples 100 --k 4
    CUDA_VISIBLE_DEVICES=1 python eval.py --checkpoint /data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base --num_samples 100 --k 4

Output: prints tables + saves eval_results.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATA = "/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/DeepMath-103K/filtered/train-00009-of-00010_filtered.parquet"
DEFAULT_MODEL = "/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base"
SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Think step by step before answering."

def extract_boxed(text: str) -> Optional[str]:
    """Extract the content inside the LAST \\boxed{...} from text.

    Handles nested braces up to 3 levels deep.
    Returns None if no \\boxed found.
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison.

    - Strip leading/trailing whitespace
    - Remove LaTeX math mode wrappers ($$...$$, \\(...\\), \\[...\\])
    - Remove leading \\displaystyle
    - Collapse multiple spaces
    - Strip trailing punctuation (. , ;)
    """
    if ans is None:
        return ""
    s = ans.strip()

    # Remove display math wrappers
    s = re.sub(r'^\$\$', '', s)
    s = re.sub(r'\$\$$', '', s)
    s = re.sub(r'^\\\[', '', s)
    s = re.sub(r'\\\]$', '', s)
    s = re.sub(r'^\\\(', '', s)
    s = re.sub(r'\\\)$', '', s)
    s = re.sub(r'^\$', '', s)
    s = re.sub(r'\$$', '', s)

    # Remove \\displaystyle
    s = s.replace('\\displaystyle', '')

    # Collapse whitespace
    s = ' '.join(s.split())

    # Strip trailing punctuation
    s = s.rstrip('.,; ')

    return s.strip()


def try_sympy_equiv(pred: str, gt: str) -> bool:
    """Check if two math expressions are algebraically equivalent using sympy.

    Returns True if sympy determines they are equal, False otherwise.
    Catches parse errors gracefully.
    """
    try:
        import sympy
    except ImportError:
        return False

    try:
        # Preprocess: sympy needs explicit multiplication
        # Replace common LaTeX patterns
        pred_s = pred.strip()
        gt_s = gt.strip()

        # Handle simple numeric equality
        try:
            pf = float(pred_s)
            gf = float(gt_s)
            return abs(pf - gf) < 1e-9
        except (ValueError, TypeError):
            pass

        # Try sympy parsing
        pred_expr = sympy.sympify(pred_s, evaluate=True)
        gt_expr = sympy.sympify(gt_s, evaluate=True)

        diff = sympy.simplify(pred_expr - gt_expr)
        return diff == 0
    except Exception:
        return False

def check_think_format(text: str) -> dict:
    """Check <think>...</think> tag compliance.

    Returns dict with:
      - has_think_open:  bool
      - has_think_close: bool
      - think_complete:  bool (both tags present and open appears before close)
    """
    has_open = '<think>' in text
    has_close = '</think>' in text

    complete = False
    if has_open and has_close:
        open_pos = text.find('<think>')
        close_pos = text.find('</think>')
        complete = (open_pos < close_pos)

    return {
        "has_think_open": has_open,
        "has_think_close": has_close,
        "think_complete": complete,
    }


def build_prompt(question: str, tokenizer) -> str:
    """Build a chat-formatted prompt for a math question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    question: str,
    k: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Generate k answers for a single question.

    Args:
        model: HF CausalLM model
        tokenizer: tokenizer
        question: raw question text
        k: number of generations
        temperature: sampling temperature (set > 0 for Pass@k diversity)
        top_p: nucleus sampling threshold
        max_new_tokens: max tokens to generate

    Returns:
        list of k generated texts (only the new part, not including prompt)
    """
    prompt = build_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(k > 1 and temperature > 0),
        temperature=temperature if (k > 1) else 1.0,
        top_p=top_p if (k > 1) else 1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(k):
        if k == 1:
            gen_ids = outputs[0][prompt_len:]
        else:
            gen_ids = outputs[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(text)

    return results

def evaluate_one_sample(
    model, tokenizer, row: pd.Series, k: int = 1, temperature: float = 0.8
) -> dict:
    """Evaluate one math question sample.

    Returns a dict with all metrics for this sample.
    """
    question = row["question"]
    gt_answer = str(row["final_answer"]).strip()
    difficulty = float(row["difficulty"])

    # Generate k answers
    generations = generate_answers(model, tokenizer, question, k=k, temperature=temperature)

    # Evaluate each generation
    gen_results = []
    for gen_text in generations:
        boxed = extract_boxed(gen_text)
        gen_norm = normalize_answer(boxed) if boxed else ""
        gt_norm = normalize_answer(gt_answer)

        em = (gen_norm == gt_norm)
        sympy_match = False
        if not em and gen_norm and gt_norm:
            sympy_match = try_sympy_equiv(gen_norm, gt_norm)

        fmt = check_think_format(gen_text)

        gen_results.append({
            "extracted_answer": boxed,
            "normalized_answer": gen_norm,
            "em": em,
            "normalized_match": (gen_norm == gt_norm),  # same as EM since both normalized
            "sympy_equiv": em or sympy_match,  # True if EM or sympy says equal
            "think_format": fmt,
            "generation_length": len(gen_text),
        })

    # Pass@k: at least one generation is correct
    em_any = any(g["em"] for g in gen_results)
    norm_any = any(g["normalized_match"] for g in gen_results)
    sympy_any = any(g["sympy_equiv"] for g in gen_results)

    return {
        "question": question,
        "gt_answer": gt_answer,
        "difficulty": difficulty,
        "topic": row.get("topic", ""),
        "token_count": int(row.get("token_count", 0)),
        "k": k,
        "generations": gen_results,
        "pass_at_k_em": em_any,
        "pass_at_k_normalized": norm_any,
        "pass_at_k_sympy": sympy_any,
        # For k=1, pass@1 = whether single generation is correct
        "em": gen_results[0]["em"] if gen_results else False,
        "normalized_match": gen_results[0]["normalized_match"] if gen_results else False,
        "sympy_match": gen_results[0]["sympy_equiv"] if gen_results else False,
        "think_complete": gen_results[0]["think_format"]["think_complete"] if gen_results else False,
    }

def compute_aggregate(results: list[dict], k: int) -> dict:
    """Compute aggregate metrics from a list of per-sample results."""
    n = len(results)

    # Overall metrics
    em_rate = sum(1 for r in results if r["em"]) / n
    norm_rate = sum(1 for r in results if r["normalized_match"]) / n
    sympy_rate = sum(1 for r in results if r["sympy_match"]) / n
    think_rate = sum(1 for r in results if r["think_complete"]) / n

    # Pass@k
    pass_k_em = sum(1 for r in results if r["pass_at_k_em"]) / n
    pass_k_norm = sum(1 for r in results if r["pass_at_k_normalized"]) / n
    pass_k_sympy = sum(1 for r in results if r["pass_at_k_sympy"]) / n

    # By difficulty
    diff_bands = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    by_difficulty = {}
    for lo, hi in diff_bands:
        band_results = [r for r in results if lo <= r["difficulty"] < hi]
        if not band_results:
            continue
        m = len(band_results)
        label = f"[{lo}, {hi})"
        by_difficulty[label] = {
            "count": m,
            "em": sum(1 for r in band_results if r["em"]) / m,
            "normalized": sum(1 for r in band_results if r["normalized_match"]) / m,
            "sympy": sum(1 for r in band_results if r["sympy_match"]) / m,
            "think_format": sum(1 for r in band_results if r["think_complete"]) / m,
        }
        if k > 1:
            by_difficulty[label][f"pass@{k}_em"] = \
                sum(1 for r in band_results if r["pass_at_k_em"]) / m

    # Answer extraction rate
    extracted_rate = sum(
        1 for r in results
        if r["generations"] and r["generations"][0]["extracted_answer"] is not None
    ) / n

    avg_gen_len = np.mean([
        r["generations"][0]["generation_length"] for r in results if r["generations"]
    ])

    return {
        "num_samples": n,
        "k": k,
        "em": em_rate,
        "normalized_match": norm_rate,
        "sympy_match": sympy_rate,
        "think_format_rate": think_rate,
        "answer_extraction_rate": extracted_rate,
        "avg_generation_length": float(avg_gen_len),
        f"pass@{k}": {
            "em": pass_k_em,
            "normalized": pass_k_norm,
            "sympy": pass_k_sympy,
        },
        "by_difficulty": by_difficulty,
    }

def print_results(metrics: dict, k: int):
    """Pretty-print evaluation results."""
    n = metrics["num_samples"]

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Samples:            {n}")
    print(f"  k (Pass@k):         {k}")
    print(f"  Answer extracted:   {metrics['answer_extraction_rate']:.2%}")
    print(f"  Avg generation len: {metrics['avg_generation_length']:.0f} chars")
    print()
    print("-" * 70)
    print(f"  {'Metric':<25s} {'Pass@1':>8s}", end="")
    if k > 1:
        print(f"  {'Pass@' + str(k):>8s}", end="")
    print()
    print("-" * 70)

    rows = [
        ("EM (Exact Match)", metrics["em"], metrics[f"pass@{k}"]["em"]),
        ("Normalized Match", metrics["normalized_match"], metrics[f"pass@{k}"]["normalized"]),
        ("SymPy Equiv", metrics["sympy_match"], metrics[f"pass@{k}"]["sympy"]),
    ]
    if k == 1:
        for label, p1, _ in rows:
            print(f"  {label:<25s} {p1:>7.2%}")
    else:
        for label, p1, pk in rows:
            print(f"  {label:<25s} {p1:>7.2%}  {pk:>7.2%}")

    print("-" * 70)
    print(f"  {'Think Format Rate':<25s} {metrics['think_format_rate']:>7.2%}")
    print()
    by_diff = metrics.get("by_difficulty", {})
    if by_diff:
        print("-" * 70)
        header = f"  {'Difficulty':<14s} {'Count':>5s} {'EM':>8s} {'Norm':>8s} {'SymPy':>8s} {'Think':>8s}"
        if k > 1:
            header += f" {'Pass@k':>8s}"
        print(header)
        print("-" * 70)
        for band, stats in sorted(by_diff.items()):
            line = (f"  {band:<14s} {stats['count']:>5d} {stats['em']:>7.2%} "
                    f"{stats['normalized']:>7.2%} {stats['sympy']:>7.2%} "
                    f"{stats['think_format']:>7.2%}")
            if k > 1:
                line += f" {stats.get(f'pass@{k}_em', 0):>7.2%}"
            print(line)
        print("-" * 70)
        print()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate math reasoning model on DeepMath-103K")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Path to filtered parquet file")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_MODEL,
                        help="Path to model checkpoint or HF model name")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of generations per sample for Pass@k")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for Pass@k")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate per answer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: ./eval_results.json)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    print(f"[Data] {len(df)} samples from {args.data}")

    # Sample
    if args.num_samples < len(df):
        df = df.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    print(f"[Data] using {len(df)} samples")

    print(f"[Model] loading from {args.checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    print(f"[Model] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B, "
          f"device={model.device}")

    print(f"\n[Eval] running {len(df)} samples, k={args.k} ...")
    t0 = time.time()

    results = []
    for i in tqdm(range(len(df)), desc="Evaluating"):
        row = df.iloc[i]
        result = evaluate_one_sample(
            model, tokenizer, row,
            k=args.k,
            temperature=args.temperature,
        )
        results.append(result)

    elapsed = time.time() - t0
    print(f"[Eval] done in {elapsed:.1f}s ({elapsed / len(df):.1f}s/sample)")
    metrics = compute_aggregate(results, args.k)
    print_results(metrics, args.k)

    output_path = args.output or os.path.join(
        os.path.dirname(__file__) or ".", "eval_results.json"
    )

    # Make results JSON-serializable
    save_data = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": args.data,
            "num_samples": len(df),
            "k": args.k,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "metrics": metrics,
        # Detailed per-sample results (for debugging)
        "samples": [
            {
                "question": r["question"],
                "gt_answer": r["gt_answer"],
                "difficulty": r["difficulty"],
                "topic": r["topic"],
                "em": r["em"],
                "normalized_match": r["normalized_match"],
                "sympy_match": r["sympy_match"],
                "think_complete": r["think_complete"],
                "pass_at_k_em": r["pass_at_k_em"] if args.k > 1 else None,
                "extracted_answer": r["generations"][0]["extracted_answer"] if r["generations"] else None,
                "gen_length": r["generations"][0]["generation_length"] if r["generations"] else 0,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"[Save] results -> {output_path}")


if __name__ == "__main__":
    main()
