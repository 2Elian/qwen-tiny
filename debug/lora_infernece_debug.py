#!/usr/bin/env python3
"""
Debug script for Qwen3 + LoRA adapter inference (non-streaming, single-turn).
Usage:
    python debug_inference.py --gpu 0 --enable_thinking
    python debug_inference.py --gpu 0 --lora_path ./output/lora-nl2sql/qkvogud/final
    python debug_inference.py --gpu 0 --question "Your SQL question here"
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


# ── ANSI 颜色 ──────────────────────────────────────────────
class Color:
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 + LoRA Debug Inference (single-turn, non-streaming)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base",
        help="Path to the pretrained base model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/src/lora/output/lora-nl2sql/qkvogud/checkpoint-1030",
        help="Path to the LoRA adapter weights (skip to use base model only)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID (e.g., '0', '0,1', 'cpu')"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (0 = greedy)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode (default: off)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="User question (if not provided, a default SQL example will be used)"
    )
    return parser.parse_args()


def setup_device(gpu_str):
    if gpu_str.lower() == 'cpu':
        print(f"{Color.YELLOW}Using device: cpu{Color.RESET}")
        return torch.device('cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    if torch.cuda.is_available():
        print(f"{Color.YELLOW}Using device: cuda:{gpu_str}{Color.RESET}")
        return torch.device('cuda:0')
    else:
        print(f"{Color.RED}Warning: CUDA not available, falling back to CPU{Color.RESET}")
        return torch.device('cpu')


def load_model_and_tokenizer(model_path, lora_path, device):
    print(f"{Color.DIM}Loading tokenizer from {model_path}...{Color.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"{Color.DIM}Loading base model from {model_path}...{Color.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None,
        trust_remote_code=True
    )
    if device.type == 'cpu':
        model = model.to(device)

    if lora_path and os.path.isdir(lora_path):
        print(f"{Color.DIM}Loading LoRA adapter from {lora_path}...{Color.RESET}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"{Color.GREEN}LoRA adapter loaded!{Color.RESET}")
    elif lora_path:
        print(f"{Color.YELLOW}Warning: LoRA path '{lora_path}' not found, using base model only.{Color.RESET}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{Color.GREEN}Model loaded! Total: {total / 1e9:.2f}B, Trainable: {trainable / 1e6:.2f}M{Color.RESET}")
    return model, tokenizer


def format_messages(messages, tokenizer, enable_thinking=False):
    """Format messages using chat template or fallback."""
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": enable_thinking}
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
    # Fallback (rarely used)
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n"
    formatted += "Assistant: "
    return formatted


def find_think_end_token_id(tokenizer):
    """Find token id for </think>."""
    for token_str in ["</think>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    if hasattr(tokenizer, 'all_special_tokens') and hasattr(tokenizer, 'all_special_ids'):
        for tok, tid in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
            if "<think>" in tok or "</think>" in tok:
                return tid
    print(f"{Color.YELLOW}Warning: Could not find </think> token, using hardcoded id 151648{Color.RESET}")
    return 151648


class ThinkEndStoppingCriteria(StoppingCriteria):
    """Stop generation when </think> is generated (used when thinking mode is off)."""
    def __init__(self, tokenizer, prompt_length):
        super().__init__()
        self.prompt_length = prompt_length
        self.think_end_ids = set()
        for token_str in ["</think>"]:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != tokenizer.unk_token_id:
                self.think_end_ids.add(tid)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.prompt_length:
            return False
        last_token_id = input_ids[0, -1].item()
        return last_token_id in self.think_end_ids


def run_single_inference(model, tokenizer, args, device):
    """Run one inference on a fixed or user-provided question, print result."""
    # Default SQL example
    default_question = """
Database Schema
###
CREATE TABLE donations
(
    donationid TEXT not null primary key,
    projectid TEXT,
    is_teacher_acct TEXT, -- description: whether donor is also a teacher; value: 'f' or 't'
    foreign key (projectid) references projects(projectid)
);

CREATE TABLE projects
(
    projectid TEXT not null primary key,
    teacher_acctid TEXT,
    school_city TEXT -- description: city where the school is located
);

###
Question: 
How many teachers have made some type of donation for projects in Chicago?

Hint:
in Chicago refers to school_city = 'Chicago'; teachers refers to is_teacher_acct = 't'
"""

    user_question = args.question if args.question else default_question

    messages = [{"role": "user", "content": user_question}]
    formatted_input = format_messages(messages, tokenizer, args.enable_thinking)

    inputs = tokenizer([formatted_input], return_tensors="pt").to(device)
    prompt_length = inputs['input_ids'].shape[1]

    # Build eos token ids
    eos_ids = set()
    eos_ids.add(tokenizer.eos_token_id)                         # 151643
    eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))  # 151645
    think_end_token_id = find_think_end_token_id(tokenizer)
    if not args.enable_thinking and think_end_token_id is not None:
        eos_ids.add(think_end_token_id)
    eos_ids = list(eos_ids)

    # Stopping criteria for </think> when thinking mode off
    if not args.enable_thinking:
        stop_criteria = ThinkEndStoppingCriteria(tokenizer, prompt_length)
        stopping_criteria = StoppingCriteriaList([stop_criteria])
    else:
        stopping_criteria = None

    use_sample = args.temperature > 0
    gen_kwargs = dict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=args.max_new_tokens,
        do_sample=use_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        repetition_penalty=args.repetition_penalty,
        **({"stopping_criteria": stopping_criteria} if stopping_criteria else {}),
    )
    if use_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    print(f"{Color.CYAN}{Color.BOLD}Generating...{Color.RESET}")
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    # Decode only the new tokens (skip prompt)
    generated_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Simple post-processing to clean up leftover think tags if any
    if not args.enable_thinking:
        # Remove any stray </think> that might have been kept as eos
        response = response.replace("</think>", "").strip()
    else:
        # Keep think tags for visibility
        pass

    # Print final answer
    print(f"\n{Color.BOLD}{Color.GREEN}Assistant:{Color.RESET} {Color.CYAN}{response}{Color.RESET}")
    print()


def main():
    args = parse_args()
    device = setup_device(args.gpu)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_path, device)
    run_single_inference(model, tokenizer, args, device)


if __name__ == "__main__":
    main()