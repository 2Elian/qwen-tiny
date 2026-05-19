"""
DeepMath-103K multi-source reward function for verl RLVR training.

Returns a dict with 'score' (float) and per-source breakdowns that
verl automatically logs to wandb via reward_extra_info.

Design principle:
  - answer_correct = 1.0 is the dominant signal (weight 1.0)
  - format compliance and reasoning quality are additive bonuses (weight ~0.1 each)
  - overlong penalty is subtractive
  - the total score is capped at [0, 1.2] roughly

Usage in verl config:
    custom_reward_function.path=deepmath_reward.py
    custom_reward_function.name=compute_score
"""

import re
from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict:
    """
    Multi-source reward function for DeepMath-103K.

    Args:
        data_source: dataset name (unused)
        solution_str: the full model-generated text
        ground_truth: the correct answer string
        extra_info: extra fields from the parquet (difficulty, topic, etc.)

    Returns:
        dict with:
            - score: float, the combined reward
            - answer_correct: 0/1
            - format_score: float, format compliance
            - reasoning_score: float, reasoning quality
            - efficiency_score: float, length penalty/bonus
            ... (each logged separately in wandb via reward_extra_info)
    """
    # ── 1. Extract answer from \boxed{...} ───────────────────────
    boxed = _extract_boxed(solution_str)

    # ── 2. Core: answer correctness ─────────────────────────────
    answer_correct = 0.0
    if boxed is not None:
        pred_norm = _normalize(boxed)
        gt_norm = _normalize(str(ground_truth))
        if pred_norm == gt_norm:
            answer_correct = 1.0
        elif _sympy_equiv(pred_norm, gt_norm):
            answer_correct = 1.0

    # ── 3. Format compliance ────────────────────────────────────
    fmt = _check_format(solution_str)

    # Sub-signals
    has_think_open = 1.0 if fmt["has_think_open"] else 0.0
    has_think_close = 1.0 if fmt["has_think_close"] else 0.0
    think_order_ok = 1.0 if fmt["think_order_ok"] else 0.0 # TODO 这个奖励分数没用
    has_boxed = 1.0 if (boxed is not None) else 0.0

    # Aggregate format score: all 4 conditions → max bonus
    format_score = 0.0
    if has_think_open:
        format_score += 0.025
    if has_think_close:
        format_score += 0.025
    if think_order_ok:
        format_score += 0.025
    if has_boxed:
        format_score += 0.025
    # format_score ∈ [0, 0.1]

    # ── 4. Reasoning quality ────────────────────────────────────
    reasoning = _check_reasoning_quality(solution_str, fmt)

    think_non_empty = 1.0 if reasoning["think_non_empty"] else 0.0
    reasoning_ratio = reasoning["reasoning_ratio"]  # 0~1, fraction of text in <think>
    has_self_verification = 1.0 if reasoning["has_self_verify"] else 0.0

    # Aggregate reasoning score
    reasoning_score = 0.0
    if think_non_empty and reasoning_ratio > 0.1:
        reasoning_score += 0.025
    if reasoning_ratio > 0.3:  # reasoning is at least 30% of response
        reasoning_score += 0.025
    if has_self_verification:
        reasoning_score += 0.05
    # reasoning_score ∈ [0, 0.1]

    # ── 5. Efficiency ───────────────────────────────────────────
    # Penalize overly long responses (encourage conciseness)
    # Use difficulty from extra_info if available
    difficulty = (extra_info or {}).get("difficulty", 4.0)
    response_len = len(solution_str)

    # Expected length scales with difficulty:
    #   difficulty 1-3:  ~800 chars
    #   difficulty 4-6: ~1500 chars
    #   difficulty 7-10: ~2500 chars
    if difficulty <= 3:
        target_len = 800
    elif difficulty <= 6:
        target_len = 1500
    else:
        target_len = 2500

    # Overlong penalty: negative if significantly over target
    overlong_ratio = response_len / max(target_len, 1)
    if overlong_ratio > 2.0:
        efficiency_score = -0.1
    elif overlong_ratio > 1.5:
        efficiency_score = -0.05
    elif overlong_ratio > 1.0:
        efficiency_score = 0.0
    else:
        # Bonus for being concise (but only if answer is correct)
        efficiency_score = 0.02 if answer_correct > 0.5 else 0.0

    # ── 6. Language consistency ─────────────────────────────────
    # Penalize mixed Chinese/English (our data is English-only)
    chinese_chars = len(re.findall(r'[一-鿿]', solution_str))
    total_chars = max(len(solution_str), 1)
    chinese_ratio = chinese_chars / total_chars
    # Penalty for heavy mixing (> 20% Chinese characters)
    lang_penalty = -0.1 if chinese_ratio > 0.2 else 0.0

    # ── 7. Combine ──────────────────────────────────────────────
    total_score = (
        answer_correct * 1.0      # core: 0 or 1
        + format_score             # up to 0.1
        + reasoning_score          # up to 0.1
        + efficiency_score         # -0.1 to 0.02
        + lang_penalty             # 0 or -0.1
    )
    total_score = max(0.0, min(1.2, total_score))

    return {
        "score": total_score,
        # Per-source breakdowns (auto-logged to wandb by verl)
        "answer_correct": answer_correct,
        "format_score": format_score,
        "reasoning_score": reasoning_score,
        "efficiency_score": efficiency_score,
        "lang_penalty": lang_penalty,
        "chinese_ratio": round(chinese_ratio, 3),
        # Atomic format signals
        "has_think_open": has_think_open,
        "has_think_close": has_think_close,
        "think_order_ok": think_order_ok,
        "has_boxed": has_boxed,
        # Atomic reasoning signals
        "think_non_empty": think_non_empty,
        "reasoning_ratio": reasoning_ratio,
        "has_self_verification": has_self_verification,
        # Context
        "response_length": response_len,
        "overlong_ratio": round(overlong_ratio, 2),
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _extract_boxed(text: str) -> str | None:
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def _normalize(s: str) -> str:
    s = s.strip()
    for env in ["$$", "$", "\\(", "\\)", "\\[", "\\]"]:
        if s.startswith(env):
            s = s[len(env):]
        if s.endswith(env):
            s = s[:-len(env)]
    s = s.replace("\\displaystyle", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\n", "").replace(" ", "")
    return s.strip()


def _sympy_equiv(a: str, b: str) -> bool:
    try:
        import sympy
        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        return False


def _check_format(text: str) -> dict:
    """Check <think>...</think> and \\boxed{} format compliance."""
    has_open = "<think>" in text
    has_close = "</think>" in text
    order_ok = False
    if has_open and has_close:
        open_pos = text.find("<think>")
        close_pos = text.find("</think>")
        order_ok = (open_pos < close_pos)
    return {
        "has_think_open": has_open,
        "has_think_close": has_close,
        "think_order_ok": order_ok,
    }


def _check_reasoning_quality(text: str, fmt: dict) -> dict:
    """Check reasoning quality metrics."""
    # Extract text within <think>...</think>
    think_text = ""
    if fmt["think_order_ok"]:
        start = text.find("<think>") + len("<think>")
        end = text.find("</think>")
        think_text = text[start:end]

    think_non_empty = len(think_text.strip()) > 10

    # Reasoning ratio: what fraction of the total response is inside <think>
    total_len = max(len(text), 1)
    reasoning_ratio = len(think_text) / total_len

    # Self-verification check: common verification phrases
    verify_patterns = [
        r"(?i)let me (check|verify|double.check|confirm)",
        r"(?i)we can verify",
        r"(?i)to verify",
        r"(?i)as a sanity check",
        r"(?i)let.?s (check|verify|confirm)",
        r"(?i)check(ing)?.?( the)? (answer|result|solution)",
        r"(?i)this (matches|confirms|is consistent)",
    ]
    has_self_verify = any(
        re.search(pat, think_text) for pat in verify_patterns
    )

    return {
        "think_non_empty": think_non_empty,
        "reasoning_ratio": round(reasoning_ratio, 3),
        "has_self_verify": has_self_verify,
        "think_length": len(think_text),
    }


# ── For standalone testing ──────────────────────────────────────────

if __name__ == "__main__":
    # Test with a mock response
    test_response = """<think>
Okay, let me think about this step by step.
The limit as x approaches infinity of sqrt(x) * (cbrt(x+1) - cbrt(x-1)).

We can use the binomial expansion for the cube roots:
cbrt(x+1) = x^{1/3} * (1 + 1/x)^{1/3} = x^{1/3} * (1 + 1/(3x) + ...)
cbrt(x-1) = x^{1/3} * (1 - 1/x)^{1/3} = x^{1/3} * (1 - 1/(3x) + ...)

The difference: cbrt(x+1) - cbrt(x-1) ~ x^{1/3} * (2/(3x)) = (2/3) * x^{-2/3}

Multiply by sqrt(x) = x^{1/2}:
x^{1/2} * (2/3) * x^{-2/3} = (2/3) * x^{-1/6} -> 0 as x -> infinity.

Let me verify: the expansion is correct and the limit indeed approaches 0.
</think>

Therefore, the limit is 0.

\\boxed{0}"""

    result = compute_score(
        data_source="deepmath-103k",
        solution_str=test_response,
        ground_truth="0",
        extra_info={"difficulty": 4.5},
    )
    for k, v in result.items():
        print(f"  {k:<25s} = {v}")
