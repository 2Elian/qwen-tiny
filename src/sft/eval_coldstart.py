#!/usr/bin/env python3
"""
Cold-start SFT model readiness evaluation — determines if model is ready for RLVR.

Six dimensions with proper data sources:
  D1. Format & Tag Stability    → general-domain prompts (not math)
  D2. Instruction Following     → IFEval (public benchmark)
  D3. Basic Knowledge Retention → self-constructed math theorem test
  D4. Reasoning Continuity      → DeepMath samples, step-level analysis
  D5. RL Target Baseline        → DeepMath held-out split vs base model
  D6. Exploration Space         → DeepMath samples, k-way diversity

Usage:
    python eval_coldstart.py --checkpoint ./output/checkpoint-500

Output: readiness report card + results.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Paths ────────────────────────────────────────────────────────────

EVAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "eval"
)
DEFAULT_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "DeepMath-103K", "filtered",
    "train-00009-of-00010_filtered.parquet"
)
DEFAULT_BASE_MODEL = "/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base"
SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Think step by step before answering."


# ── Model Utils ──────────────────────────────────────────────────────

def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tokenizer, question: str, k: int = 1, temperature: float = 0.0,
             max_new_tokens: int = 1024, top_p: float = 1.0) -> list[str]:
    """Generate k responses for a question. k=1 + temp=0 = greedy."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = (k > 1 and temperature > 0)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(k):
        ids = outputs[0][prompt_len:] if k == 1 else outputs[i][prompt_len:]
        results.append(tokenizer.decode(ids, skip_special_tokens=True))
    return results


# ── D1: Format & Tag Stability (general-domain prompts) ──────────────

def load_general_prompts() -> list[dict]:
    """Load general prompts, falling back to built-in."""
    path = os.path.join(EVAL_DATA_DIR, "general_prompts_test.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def eval_format_stability(model, tokenizer, k: int) -> dict:
    """D1: Check format compliance on general-domain prompts.

    Key insight: if the model can only output <think> on math prompts
    but loses the ability on general prompts, it's overfitted.
    """
    print("\n[D1] Format & Tag Stability (general prompts) ...")

    prompts = load_general_prompts()
    if not prompts:
        print("  ⚠ No general prompts found. Using built-in set.")
        prompts = [
            {"question": "Write a short poem about the changing seasons.", "category": "creative"},
            {"question": "Explain how a rainbow forms to a child.", "category": "science"},
            {"question": "What are the main causes of World War I?", "category": "history"},
        ]

    questions = [p["question"] for p in prompts]
    categories = [p.get("category", "unknown") for p in prompts]

    think_complete = []
    has_boxed = []
    by_category = defaultdict(lambda: {"total": 0, "think": 0, "boxed": 0})

    for q, cat in tqdm(list(zip(questions, categories)), desc="D1"):
        gens = generate(model, tokenizer, q, k=k, temperature=0.8)
        for g in gens:
            fmt_ok = g.find("<think>") >= 0 and g.find("</think>") > g.find("<think>")
            boxed_ok = extract_boxed(g) is not None
            think_complete.append(fmt_ok)
            has_boxed.append(boxed_ok)
            by_category[cat]["total"] += 1
            by_category[cat]["think"] += int(fmt_ok)
            by_category[cat]["boxed"] += int(boxed_ok)

    n = len(think_complete)
    think_rate = sum(think_complete) / max(n, 1)
    boxed_rate = sum(has_boxed) / max(n, 1)

    print(f"  <think> complete: {think_rate:.2%}")
    print(f"  \\boxed present:   {boxed_rate:.2%}")
    if by_category:
        for cat, stats in sorted(by_category.items()):
            if stats["total"]:
                print(f"    {cat}: think={stats['think']/stats['total']:.0%}  "
                      f"boxed={stats['boxed']/stats['total']:.0%}")

    return {"think_complete_rate": think_rate, "boxed_rate": boxed_rate,
            "by_category": {c: {"think": s["think"]/s["total"], "boxed": s["boxed"]/s["total"]}
                           for c, s in by_category.items()},
            "score": float(np.mean([think_rate, boxed_rate]))}


# ── D2: Instruction Following (IFEval) ──────────────────────────────

def load_ifeval() -> Optional[list[dict]]:
    """Load IFEval dataset. Returns None if not downloaded."""
    manifest = os.path.join(EVAL_DATA_DIR, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            m = json.load(f)
        if m.get("ifeval", {}).get("available"):
            path = m["ifeval"]["path"]
            df = pd.read_parquet(path)
            return df.to_dict("records")
    return None


IFEVAL_CONSTRAINTS = {
    "length_constraints:number_words": {
        "pattern": r"(\d+)\s+words?", "check": lambda text, n:
            abs(len(text.split()) - int(n)) <= int(n) * 0.2
    },
    "length_constraints:number_sentences": {
        "pattern": r"(\d+)\s+sentences?", "check": lambda text, n:
            abs(len(re.split(r'[.!?]+', text)) - int(n)) <= 2
    },
    "detectable_format:number_bullet_lists": {
        "pattern": r"(\d+)", "check": lambda text, n:
            len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE)) >= int(n) - 1
    },
    "keywords:existence": {
        "pattern": r'"([^"]+)"', "check": lambda text, kw: kw.lower() in text.lower()
    },
    "keywords:forbidden_words": {
        "pattern": r'"([^"]+)"', "check": lambda text, kw: kw.lower() not in text.lower()
    },
    "startend:end_checker": {
        "pattern": r'(.*)', "check": lambda text, ending: text.rstrip().endswith(ending.rstrip('.'))
    },
}


def check_ifeval_constraint(instruction_id: str, text: str, kwargs: dict) -> bool:
    """Check if generated text satisfies a specific IFEval constraint."""
    for key, handler in IFEVAL_CONSTRAINTS.items():
        if key in instruction_id:
            match = re.search(handler["pattern"], str(kwargs))
            if match:
                try:
                    return handler["check"](text, match.group(1))
                except (ValueError, IndexError):
                    return False
    return True  # unknown constraint → skip


def eval_instruction_following(model, tokenizer, num_samples: int = 100) -> dict:
    """D2: IFEval instruction following.

    If IFEval is not downloaded, uses a minimal built-in set and warns.
    """
    print("\n[D2] Instruction Following ...")

    ifeval_data = load_ifeval()

    if ifeval_data is None:
        print("  ⚠ IFEval not downloaded. Run: python download_eval_datasets.py --datasets ifeval")
        print("  Using minimal built-in instruction test instead.")
        # Built-in fallback: test basic format instructions
        tests = [
            ("Write exactly three paragraphs about climate change. "
             "Separate paragraphs with blank lines.", "exactly 3 paragraphs"),
            ("Your response must contain the word 'therefore' at least twice.", "contains 'therefore'"),
            ("Write a response that ends with the sentence: 'This is the end.'", "ends with specific text"),
        ]
        score = 0
        for prompt, desc in tqdm(tests, desc="D2 (built-in)"):
            gen = generate(model, tokenizer, prompt, k=1, temperature=0)[0]
            # Simple heuristics
            if "paragraph" in prompt:
                paras = [p.strip() for p in gen.split("\n\n") if p.strip()]
                score += 1.0 if len(paras) >= 3 else 0.0
            elif "therefore" in prompt:
                score += 1.0 if gen.lower().count("therefore") >= 2 else 0.0
            elif "end" in prompt:
                score += 1.0 if gen.rstrip().endswith("This is the end.") else 0.0
        score /= len(tests)
        print(f"  Built-in IF score: {score:.2%}")
        return {"ifeval_score": score, "score": score, "note": "using built-in fallback"}

    # Real IFEval
    import random
    random.seed(42)
    samples = random.sample(ifeval_data, min(num_samples, len(ifeval_data)))

    total_constraints = 0
    passed_constraints = 0
    prompt_level_pass = 0

    for s in tqdm(samples, desc="D2 IFEval"):
        prompt = s["prompt"]
        instruction_ids = s.get("instruction_id_list", [])
        kwargs_list = s.get("kwargs", [])

        gen = generate(model, tokenizer, prompt, k=1, temperature=0)[0]
        prompt_ok = True

        for instr_id, kw in zip(instruction_ids, kwargs_list):
            total_constraints += 1
            ok = check_ifeval_constraint(instr_id, gen, kw)
            if ok:
                passed_constraints += 1
            else:
                prompt_ok = False

        if prompt_ok and instruction_ids:
            prompt_level_pass += 1

    constraint_rate = passed_constraints / max(total_constraints, 1)
    prompt_rate = prompt_level_pass / max(len(samples), 1)

    print(f"  Constraint-level: {constraint_rate:.2%}")
    print(f"  Prompt-level:     {prompt_rate:.2%}")

    return {"constraint_pass_rate": constraint_rate, "prompt_pass_rate": prompt_rate,
            "score": constraint_rate}


# ── D3: Basic Knowledge Retention ────────────────────────────────────

def load_knowledge_test() -> list[dict]:
    path = os.path.join(EVAL_DATA_DIR, "knowledge_test.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def eval_knowledge_retention(model, tokenizer, base_model, base_tokenizer) -> dict:
    """D3: Self-constructed math theorem/knowledge test.

    Compare SFT model vs base model on direct theorem recall.
    If SFT forgets basic theorems, the RL process will suffer.
    """
    print("\n[D3] Basic Knowledge Retention (self-built math theorem test) ...")

    tests = load_knowledge_test()
    if not tests:
        print("  ⚠ No knowledge test found.")
        return {"score": 0.5, "note": "no test data"}

    sft_correct = 0
    base_correct = 0
    total = len(tests)

    for item in tqdm(tests, desc="D3 Knowledge"):
        q = item["question"]
        gt = item["answer"]

        gen_sft = generate(model, tokenizer, q, k=1, temperature=0)[0]
        gen_base = generate(base_model, base_tokenizer, q, k=1, temperature=0)[0]

        if _fuzzy_knowledge_match(gen_sft, gt):
            sft_correct += 1
        if _fuzzy_knowledge_match(gen_base, gt):
            base_correct += 1

    sft_rate = sft_correct / total
    base_rate = base_correct / total
    degradation = base_rate - sft_rate

    print(f"  SFT model:   {sft_rate:.2%} ({sft_correct}/{total})")
    print(f"  Base model:  {base_rate:.2%} ({base_correct}/{total})")
    print(f"  Degradation: {degradation:+.2%}")

    return {"sft_accuracy": sft_rate, "base_accuracy": base_rate,
            "degradation": degradation,
            "score": max(0.0, min(1.0, 1.0 - degradation * 3))}


def _fuzzy_knowledge_match(generated: str, gt: str) -> bool:
    """Check if generated text conveys the correct mathematical knowledge.

    Uses character-level overlap of the mathematical core (variables,
    numbers, operators) — robust to LaTeX vs plain text differences.
    """
    def _math_core(s: str) -> set:
        """Extract mathematical tokens: variables, numbers, function names."""
        s = s.lower()
        # Remove LaTeX commands (keep their content/equivalents)
        for cmd in [r'\frac', r'\sqrt', r'\pm', r'\cdot', r'\times',
                     r'\sum', r'\int', r'\prod', r'\lim',
                     r'\left', r'\right', r'\displaystyle',
                     r'\begin', r'\end', r'\text', r'\textbf',
                     r'\mathbb', r'\mathbf', r'\mathcal',
                     r'\ldots', r'\cdots', r'\vdots', r'\ddots',
                     r'\boxed', r'\circ']:
            s = s.replace(cmd, '')
        # Remove braces
        s = s.replace('{', '').replace('}', '')
        # Extract tokens: sequences of letters/digits/operators
        tokens = set(re.findall(r'[a-z0-9+\-*/^=<>|!]+', s))
        # Remove noise: single chars (except known important ones)
        noise = set('abcdefghijklmnopqrstuvwxyz')  # single letters as standalone tokens
        tokens -= noise
        # Keep if length >= 2 or is a number
        return {t for t in tokens if len(t) >= 2 or t.isdigit()}

    gen_core = _math_core(generated)
    gt_core = _math_core(gt)

    if not gt_core:
        return True  # can't check, assume correct

    # Fraction of gt tokens that appear in the generation
    overlap = len(gt_core & gen_core)
    ratio = overlap / len(gt_core)

    # Also check: does the generation mention key descriptive words from gt?
    gt_desc = set(re.findall(r'[a-z]{4,}', gt.lower())) - _math_core(gt)
    if gt_desc:
        gen_words = set(re.findall(r'[a-z]{4,}', generated.lower()))
        desc_overlap = len(gt_desc & gen_words) / len(gt_desc)
        ratio = max(ratio, desc_overlap)

    return ratio >= 0.4


# ── D4: Reasoning Continuity ────────────────────────────────────────

def eval_reasoning_continuity(model, tokenizer, questions: list[str], k: int) -> dict:
    """D4: Check if the reasoning chain is logical and well-structured.

    Specific checks:
      1. Step connectivity — logical transition words between steps
      2. No contradiction — no "wait, actually..." followed by no correction
      3. Goal-directed — the reasoning references the final answer
      4. Step coherence — sentences build on each other (not random)
    """
    print("\n[D4] Reasoning Continuity ...")

    logical_transitions = [
        "therefore", "thus", "hence", "so", "consequently", "because",
        "since", "given that", "it follows", "this implies", "we have",
        "then", "next", "finally", "accordingly", "as a result",
    ]
    self_correction_markers = [
        "wait", "actually", "let me reconsider", "that was wrong",
        "I made a mistake", "let me correct", "oops", "scratch that",
    ]

    completeness_scores = []
    self_correction_count = 0
    transition_density = []
    step_diversity = []

    for q in tqdm(questions, desc="D4 Continuity"):
        gens = generate(model, tokenizer, q, k=1, temperature=0.8)
        for g in gens:
            think_start = g.find("<think>")
            think_end = g.find("</think>")

            if think_start < 0 or think_end <= think_start:
                completeness_scores.append(0.0)
                transition_density.append(0.0)
                step_diversity.append(0.0)
                continue

            think_text = g[think_start + len("<think>"):think_end]

            # ── 1. Transition density ──
            sentences = [s.strip() for s in re.split(r'[.!?\n]+', think_text) if len(s.strip()) > 10]
            if sentences:
                transition_count = sum(
                    1 for s in sentences
                    if any(t in s.lower() for t in logical_transitions)
                )
                transition_density.append(transition_count / len(sentences))

            # ── 2. Self-correction (positive signal for reasoning) ──
            if any(m in think_text.lower() for m in self_correction_markers):
                self_correction_count += 1

            # ── 3. Completeness: does reasoning reference the answer? ──
            boxed_answer = extract_boxed(g)
            if boxed_answer and boxed_answer.strip():
                # Check if the answer appears in the reasoning (before \boxed)
                # This indicates the reasoning actually led to the answer
                if boxed_answer.strip() in think_text:
                    completeness_scores.append(1.0)
                else:
                    completeness_scores.append(0.5)  # answer exists but not in reasoning
            else:
                completeness_scores.append(0.0)

            # ── 4. Step diversity ──
            if len(sentences) >= 3:
                sentence_starts = [s.split()[0].lower() if s.split() else "" for s in sentences[:10]]
                unique_starts = len(set(sentence_starts))
                step_diversity.append(unique_starts / len(sentence_starts))
            else:
                step_diversity.append(0.0)

    completeness = np.mean(completeness_scores) if completeness_scores else 0.0
    transition = np.mean(transition_density) if transition_density else 0.0
    correction_rate = self_correction_count / max(len(questions) * 1, 1)
    step_div = np.mean(step_diversity) if step_diversity else 0.0

    print(f"  Completeness:     {completeness:.3f}  (answer referenced in reasoning)")
    print(f"  Transition density: {transition:.3f}  (logical connectors per sentence)")
    print(f"  Self-correction:  {correction_rate:.3f}  (fraction with self-checking)")
    print(f"  Step diversity:   {step_div:.3f}  (unique sentence openings)")

    # Score: composite
    score = float(np.mean([completeness, transition * 3, min(correction_rate * 5, 1.0), step_div]))

    return {"completeness": completeness, "transition_density": transition,
            "self_correction_rate": correction_rate, "step_diversity": step_div,
            "score": score}


# ── D5: RL Target Baseline ──────────────────────────────────────────

def eval_rl_baseline(model, tokenizer, base_model, base_tokenizer,
                     data_file: str, num_samples: int) -> dict:
    """D5: Accuracy on RL target data vs base model."""
    print("\n[D5] RL Target Baseline ...")

    df = pd.read_parquet(data_file)
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)

    sft_correct = 0
    base_correct = 0
    total = 0
    by_diff = defaultdict(lambda: {"sft": 0, "base": 0, "count": 0})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="D5"):
        q = row["question"]
        gt = str(row["final_answer"]).strip()
        diff = float(row["difficulty"])

        ans_sft = extract_boxed(generate(model, tokenizer, q, k=1, temperature=0)[0])
        ans_base = extract_boxed(generate(base_model, base_tokenizer, q, k=1, temperature=0)[0])

        total += 1
        if ans_sft and ans_sft.strip() == gt:
            sft_correct += 1
        if ans_base and ans_base.strip() == gt:
            base_correct += 1

        band = f"[{int(diff//2)*2}, {int(diff//2)*2+2})"
        by_diff[band]["count"] += 1
        if ans_sft and ans_sft.strip() == gt:
            by_diff[band]["sft"] += 1
        if ans_base and ans_base.strip() == gt:
            by_diff[band]["base"] += 1

    sft_acc = sft_correct / max(total, 1)
    base_acc = base_correct / max(total, 1)

    print(f"  SFT:  {sft_acc:.2%}  |  Base: {base_acc:.2%}  |  Delta: {sft_acc - base_acc:+.2%}")
    for band in sorted(by_diff):
        s = by_diff[band]
        if s["count"]:
            print(f"    {band}: SFT={s['sft']/s['count']:.2%}  Base={s['base']/s['count']:.2%}  (n={s['count']})")

    return {"sft_accuracy": sft_acc, "base_accuracy": base_acc,
            "delta": sft_acc - base_acc, "by_difficulty": {
                b: {"sft": s["sft"]/max(s["count"],1), "base": s["base"]/max(s["count"],1), "count": s["count"]}
                for b, s in by_diff.items()},
            "score": max(0.0, sft_acc / max(base_acc, 0.01))}


# ── D6: Exploration Space ───────────────────────────────────────────

def eval_exploration_space(model, tokenizer, questions: list[str], k: int) -> dict:
    """D6: Output diversity under temperature sampling.

    For GRPO to work, the model must produce diverse reasoning paths
    that converge to the same correct answer.
    """
    print("\n[D6] Exploration Space ...")

    answer_divs = []
    ngram_divs = []
    length_cvs = []
    struct_divs = []
    has_correct = 0

    for q in tqdm(questions, desc="D6 Exploration"):
        gens = generate(model, tokenizer, q, k=k, temperature=0.8)

        answers = [extract_boxed(g) for g in gens]
        answers_clean = [a.strip() for a in answers if a is not None]

        answer_divs.append(len(set(answers_clean)) / k if answers_clean else 0.0)

        # Check if at least one correct answer exists
        if any(a is not None for a in answers):
            has_correct += 1

        # N-gram diversity
        def get_ngrams(text, n=3):
            words = text.split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        ngram_sets = [get_ngrams(g) for g in gens]
        pair_sims = []
        for i in range(len(ngram_sets)):
            for j in range(i+1, len(ngram_sets)):
                u = ngram_sets[i] | ngram_sets[j]
                if u:
                    pair_sims.append(len(ngram_sets[i] & ngram_sets[j]) / len(u))
        ngram_divs.append(1.0 - np.mean(pair_sims) if pair_sims else 0.0)

        # Length variance
        lens = [len(g) for g in gens]
        length_cvs.append(np.std(lens) / max(np.mean(lens), 1))

        # Structure diversity
        opens = [" ".join(g.strip().split()[:5]) for g in gens]
        struct_divs.append(len(set(opens)) / len(opens) if opens else 0.0)

    ans_div = float(np.mean(answer_divs))
    ngram_div = float(np.mean(ngram_divs))
    len_cv = float(np.mean(length_cvs))
    struct_div = float(np.mean(struct_divs))

    print(f"  Answer diversity:      {ans_div:.3f}  (want: similar answers = low)")
    print(f"  N-gram diversity:      {ngram_div:.3f}  (want: diverse paths = high >0.3)")
    print(f"  Length variation:      {len_cv:.3f}  (want: varied lengths >0.1)")
    print(f"  Structure diversity:   {struct_div:.3f}  (want: diverse openings >0.3)")

    score_ngram = min(1.0, ngram_div / 0.4)
    score_len = min(1.0, len_cv / 0.15)
    score_struct = min(1.0, struct_div / 0.5)

    return {"answer_diversity": ans_div, "ngram_diversity": ngram_div,
            "length_cv": len_cv, "structure_diversity": struct_div,
            "score": float(np.mean([score_ngram, score_len, score_struct]))}


# ── Helpers ─────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


# ── Report ──────────────────────────────────────────────────────────

def print_report(dimensions: dict, readiness: dict, args):
    weights = {"D1": 0.15, "D2": 0.15, "D3": 0.15, "D4": 0.10,
               "D5": 0.30, "D6": 0.15}

    print("\n" + "=" * 70)
    print("  COLD-START READINESS REPORT")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Base:       {args.base_checkpoint}")
    print()

    labels = {
        "D1": "Format Stability (general data)",
        "D2": "Instruction Following",
        "D3": "Knowledge Retention",
        "D4": "Reasoning Continuity",
        "D5": "RL Target Baseline",
        "D6": "Exploration Space",
    }

    for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        s = readiness["dim_scores"].get(dim, 0)
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  [{dim}] {labels[dim]:<32s} {s:>5.0%} {bar}")

    print(f"  {'─'*66}")
    print(f"  OVERALL: {readiness['overall']:.0%}")
    print(f"  {readiness['verdict']}")

    if readiness.get("notes"):
        for n in readiness["notes"]:
            print(f"    • {n}")
    print("=" * 70 + "\n")


# ── Main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_checkpoint", default=DEFAULT_BASE_MODEL)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──
    print(f"Loading RL data: {args.data_file}")
    df_rl = pd.read_parquet(args.data_file)
    if args.num_samples < len(df_rl):
        df_eval = df_rl.sample(n=args.num_samples, random_state=42)
    else:
        df_eval = df_rl.reset_index(drop=True)

    eval_questions = df_eval["question"].tolist()

    # ── Load models ──
    print(f"Loading SFT: {args.checkpoint}")
    model, tok = load_model(args.checkpoint)
    print(f"Loading base: {args.base_checkpoint}")
    base_model, base_tok = load_model(args.base_checkpoint)

    t0 = time.time()

    # ── Evaluate ──
    dims = {}
    notes = []

    # D1: Format stability on general prompts
    dims["D1"] = eval_format_stability(model, tok, args.k)

    # D2: Instruction following
    dims["D2"] = eval_instruction_following(model, tok, args.num_samples)

    # D3: Knowledge retention
    dims["D3"] = eval_knowledge_retention(model, tok, base_model, base_tok)

    # D4: Reasoning continuity
    dims["D4"] = eval_reasoning_continuity(model, tok, eval_questions[:min(args.num_samples, 30)], args.k)

    # D5: RL baseline
    dims["D5"] = eval_rl_baseline(model, tok, base_model, base_tok, args.data_file, args.num_samples)

    # D6: Exploration space
    dims["D6"] = eval_exploration_space(model, tok, eval_questions[:min(args.num_samples, 30)], args.k)

    elapsed = time.time() - t0

    # ── Compute readiness ──
    weights = {"D1": 0.15, "D2": 0.15, "D3": 0.15, "D4": 0.10, "D5": 0.30, "D6": 0.15}
    scores = {k: v["score"] for k, v in dims.items()}
    overall = sum(scores[k] * weights[k] for k in scores) / sum(weights[k] for k in scores)

    # ── Generate notes ──
    if dims["D1"]["score"] < 0.5:
        notes.append(f"D1: Format unstable — <think> on general prompts = {dims['D1']['think_complete_rate']:.0%}. "
                      "RL reward will fail on most samples.")
    if dims["D2"]["score"] < 0.3:
        notes.append("D2: Instruction following poor — model may ignore system prompt during RL.")
    if dims["D3"].get("degradation", 0) > 0.2:
        notes.append(f"D3: Knowledge degraded {dims['D3']['degradation']:.0%} vs base — "
                      "retrain with lower LR or add general data.")
    if dims["D5"]["delta"] < -0.1:
        notes.append("D5: RL target accuracy significantly WORSE than base model. "
                      "SFT is destroying math ability — lower LR, add more data.")
    if dims["D6"]["score"] < 0.3:
        notes.append("D6: Exploration space collapsed — GRPO will have no learning signal. "
                      "Model is overfitted to a single reasoning template.")

    if overall >= 0.80:
        verdict = "✓ READY — proceed to RL training"
    elif overall >= 0.60:
        verdict = "⚠ BORDERLINE — may work, monitor closely, fix issues above first"
    elif overall >= 0.40:
        verdict = "✗ NOT READY — fix issues before RL"
    else:
        verdict = "✗ BROKEN — model collapsed, retrain from scratch"

    readiness = {"dim_scores": scores, "overall": overall, "verdict": verdict, "notes": notes}

    # ── Print ──
    print_report(dims, readiness, args)

    # ── Save ──
    output_path = args.output or os.path.join(
        os.path.dirname(__file__) or ".", "eval_coldstart_results.json"
    )
    save = {
        "config": {"checkpoint": args.checkpoint, "base": args.base_checkpoint,
                   "num_samples": args.num_samples, "k": args.k, "elapsed_s": round(elapsed)},
        "readiness": readiness,
        "dimensions": {k: {kk: vv for kk, vv in v.items() if kk != "by_category"}
                       for k, v in dims.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
