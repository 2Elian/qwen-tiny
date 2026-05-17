#!/usr/bin/env python3
"""
Interactive CLI for Qwen3 + LoRA adapter inference with streaming output.
Usage:
    python inference_lora.py --gpu 3
    python inference_lora.py --gpu 0 --lora_path ./output/lora-nl2sql/qkvogud/final
    python inference_lora.py --gpu 0 --enable_thinking    # 开启思考模式
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import re


# ── ANSI 颜色 ──────────────────────────────────────────────
class Color:
    GREEN   = "\033[92m"   # 用户输入
    CYAN    = "\033[96m"   # 模型输出
    YELLOW  = "\033[93m"   # 系统/提示
    RED     = "\033[91m"   # 错误
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


def clear_terminal():
    """清除终端屏幕并把光标移到左上角。"""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 + LoRA Interactive CLI")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b",
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
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=10,
        help="Maximum number of conversation turns to keep (-1 for unlimited)"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode (default: off for LoRA model; the adapter was trained without think tags)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Repetition penalty to suppress looping (1.0 = off)"
    )
    return parser.parse_args()


def setup_device(gpu_str):
    """Setup device based on GPU argument."""
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
    """Load base model, tokenizer, and optionally LoRA adapter."""
    print(f"{Color.DIM}Loading tokenizer from {model_path}...{Color.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print(f"{Color.DIM}Loading base model from {model_path}...{Color.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None,
        trust_remote_code=True
    )

    if device.type == 'cpu':
        model = model.to(device)

    # ── 加载 LoRA adapter ──
    if lora_path and os.path.isdir(lora_path):
        print(f"{Color.DIM}Loading LoRA adapter from {lora_path}...{Color.RESET}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"{Color.GREEN}LoRA adapter loaded!{Color.RESET}")
    elif lora_path:
        print(f"{Color.YELLOW}Warning: LoRA path '{lora_path}' not found, using base model only.{Color.RESET}")

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    # Count trainable (LoRA) vs total parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{Color.GREEN}Model loaded! Total: {total / 1e9:.2f}B, Trainable: {trainable / 1e6:.2f}M{Color.RESET}")

    return model, tokenizer


def find_think_end_token_id(tokenizer):
    """
    找到 Qwen3 的  token id。
    尝试多种方式，确保找到正确的 id。
    """
    # 方法1: convert_tokens_to_ids
    for token_str in ["</think>", "</think>"]:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return token_id

    # 方法2: 从 special_tokens_map 查找
    if hasattr(tokenizer, 'special_tokens_map'):
        for key, val in tokenizer.special_tokens_map.items():
            if "<think>" in str(val) or "</think>" in str(val):
                token_id = tokenizer.convert_tokens_to_ids(val)
                if token_id is not None and token_id != tokenizer.unk_token_id:
                    return token_id

    # 方法3: 遍历所有 special tokens
    if hasattr(tokenizer, 'all_special_tokens') and hasattr(tokenizer, 'all_special_ids'):
        for tok, tid in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
            if "<think>" in tok or "</think>" in tok:
                return tid

    # 方法4: 硬编码 Qwen3 的常见 token id (151648)
    print(f"{Color.YELLOW}Warning: Could not find  token in vocabulary, trying hardcoded id 151648{Color.RESET}")
    return 151648


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
                try:
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"{Color.RED}Warning: Chat template failed ({e}), using fallback format{Color.RESET}")
        except Exception as e:
            print(f"{Color.RED}Warning: Chat template failed ({e}), using fallback format{Color.RESET}")

    # Fallback
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


class ThinkEndStoppingCriteria(StoppingCriteria):
    """
    兜底停止条件：如果模型生成了  标记，立即停止。
    这防止了当 eos_token_id 没有正确包含 think_end token 时的无限生成。
    """

    def __init__(self, tokenizer, prompt_length):
        super().__init__()
        self.tokenizer = tokenizer
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


class InteractiveStreamer(TextStreamer):
    """Custom streamer that prints tokens as they're generated with color."""

    def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.generated_text = ""
        self.first_token = True
        self.in_think_block = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        processed = text
        if self.first_token:
            sys.stdout.write(f"{Color.CYAN}Assistant: {Color.RESET}{Color.CYAN}")
            sys.stdout.flush()
            self.first_token = False

        i = 0
        output_buf = ""
        while i < len(processed):
            chunk = processed[i:]

            if not self.in_think_block:
                think_start = chunk.find("<think>")
                if think_start != -1:
                    before = processed[i:i + think_start]
                    output_buf += before
                    sys.stdout.write(f"{Color.CYAN}{before}{Color.RESET}")
                    sys.stdout.flush()
                    self.in_think_block = True
                    i += think_start + 7
                    sys.stdout.write(f"{Color.DIM}{Color.GREEN}<think>{Color.RESET}")
                    sys.stdout.flush()
                    continue
                else:
                    sys.stdout.write(f"{Color.CYAN}{chunk}{Color.RESET}")
                    sys.stdout.flush()
                    output_buf += chunk
                    i = len(processed)
            else:
                think_end = chunk.find("</think>")
                if think_end != -1:
                    think_content = processed[i:i + think_end]
                    sys.stdout.write(f"{Color.DIM}{Color.GREEN}{think_content}</think>{Color.RESET}")
                    sys.stdout.flush()
                    output_buf += think_content + "</think>"
                    self.in_think_block = False
                    i += think_end + 8
                else:
                    sys.stdout.write(f"{Color.DIM}{Color.GREEN}{chunk}{Color.RESET}")
                    sys.stdout.flush()
                    output_buf += chunk
                    i = len(processed)

        self.generated_text += output_buf

        if stream_end:
            sys.stdout.write(f"{Color.RESET}\n\n")
            sys.stdout.flush()
            self.first_token = True
            self.in_think_block = False

    def get_generated_text(self):
        return self.generated_text.strip()


def get_user_input():
    """
    获取用户输入，支持多行。
    单行：直接输入内容，按 Enter 提交。
    多行：输入  换行，然后输入内容，最后输入  换行提交。
    """
    sys.stdout.write(f"{Color.GREEN}{Color.BOLD}You: {Color.RESET}{Color.GREEN}")
    sys.stdout.flush()
    line = input()
    sys.stdout.write(Color.RESET)
    sys.stdout.flush()

    if line.strip() == "":
        return ""

    if line.strip() == "<<<":
        lines = []
        while True:
            sys.stdout.write(f"{Color.GREEN}... {Color.RESET}{Color.GREEN}")
            sys.stdout.flush()
            line = input()
            sys.stdout.write(Color.RESET)
            sys.stdout.flush()
            if line.strip() == ">>>":
                break
            lines.append(line)
        return "\n".join(lines)

    return line


def chat_loop(model, tokenizer, args, device):
    """Main interactive chat loop."""
    think_end_token_id = find_think_end_token_id(tokenizer)
    print(f"{Color.DIM}Think end token id: {think_end_token_id}{Color.RESET}")

    print(f"\n{'='*60}")
    print(f"{Color.BOLD}{Color.GREEN}Interactive Chat Started!{Color.RESET}")
    print(f"Device: {device}")
    print(f"Thinking mode: {'ON' if args.enable_thinking else 'OFF'}")
    print(f"{Color.YELLOW}Commands:{Color.RESET}")
    print(f"  /clear   - Clear conversation history & terminal")
    print(f"  /exit    - Quit the program")
    print(f"  /help    - Show this help")
    print(f"{Color.YELLOW}Multi-line input:{Color.RESET}")
    print(f"  <<< + Enter  - Start multi-line mode")
    print(f"  >>> + Enter  - End multi-line mode & submit")
    print(f"{'='*60}\n")

    messages = [{"role": "system", "content": "You are a SQL expert. Given a database schema and a natural language question, think step by step and generate the correct SQL query."}]

    while True:
        try:
            user_input = get_user_input().strip()

            if user_input.lower() == '/exit':
                print(f"{Color.YELLOW}Goodbye!{Color.RESET}")
                break
            elif user_input.lower() == '/clear':
                messages = []
                clear_terminal()
                print(f"{Color.YELLOW}Conversation history & terminal cleared.{Color.RESET}\n")
                continue
            elif user_input.lower() == '/help':
                print(f"{Color.YELLOW}Commands:{Color.RESET}")
                print(f"  /clear   - Clear conversation history & terminal")
                print(f"  /exit    - Quit the program")
                print(f"  /help    - Show this help")
                print(f"{Color.YELLOW}Multi-line input:{Color.RESET}")
                print(f"  <<< + Enter  - Start multi-line mode")
                print(f"  >>> + Enter  - End multi-line mode & submit\n")
                continue
            elif not user_input:
                continue

            # 限制历史长度
            if args.max_history_turns > 0:
                max_messages = args.max_history_turns * 2
                if len(messages) >= max_messages:
                    messages = messages[-max_messages + 2:]

            messages.append({"role": "user", "content": user_input})

            formatted_input = format_messages(messages, tokenizer, args.enable_thinking)

            inputs = tokenizer(
                [formatted_input],
                return_tensors="pt"
            ).to(device)

            prompt_length = inputs['input_ids'].shape[1]

            streamer = InteractiveStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            eos_ids = set()
            eos_ids.add(tokenizer.eos_token_id)                         # 151643: <|endoftext|>
            eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))  # 151645: chat template 消息结束符
            if not args.enable_thinking:
                if think_end_token_id is not None:
                    eos_ids.add(think_end_token_id)
            eos_ids = list(eos_ids)

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
                streamer=streamer,
                **({"stopping_criteria": stopping_criteria} if stopping_criteria else {}),
            )
            if use_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)

            response = streamer.get_generated_text()

            if not response:
                response = "(Model did not generate a response after think block.)"

            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}Interrupted. Type /exit to quit or continue chatting.{Color.RESET}\n")
            continue
        except torch.cuda.OutOfMemoryError:
            print(f"\n{Color.RED}Error: GPU out of memory. Try reducing max_new_tokens or clearing history with /clear{Color.RESET}\n")
            if len(messages) > 0 and messages[-1]["role"] == "user":
                messages = messages[:-1]
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n{Color.RED}Error: {e}{Color.RESET}\n")
            if len(messages) > 0 and messages[-1]["role"] == "user":
                messages = messages[:-1]
            continue


def main():
    args = parse_args()

    device = setup_device(args.gpu)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_path, device)

    chat_loop(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
