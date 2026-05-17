#!/usr/bin/env python3
"""
LoRA fine-tuning for Qwen3 on NL2SQL CoT data.
Usage:
    python main.py
    python main.py --config my_config.yaml
"""

import os
import sys
import yaml
import argparse
import math
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_messages(row: dict) -> list[dict]:
    query = str(row["query"]).strip()
    thoughts = str(row.get("thoughts") or row.get("thinking_process") or "").strip()
    answer = str(row["answer"])
    if thoughts == "None" or thoughts == "" or thoughts == "nan":
        return None
    if "<think>" not in thoughts or "</think>" not in thoughts:
        print("警告: 思考过程中缺少 <think> 或 </think> 标签，跳过")
        return None
    thoughts = thoughts.replace("<answer>", "").replace("</answer>", "").strip()
    response = f"{thoughts}\n{answer}<|endoftext|>"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a SQL expert. Given a database schema and a natural language question, "
                "think step by step and generate the correct SQL query."
            ),
        },
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]
    return messages


def preprocess(examples: dict, tokenizer, max_seq_length: int) -> dict:
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    skip_count = 0

    for i in range(len(examples["query"])):
        row = {
            "query": examples["query"][i],
            "answer": examples["answer"][i],
            "thoughts": examples.get("thoughts", [None] * len(examples["query"]))[i]
                        if "thoughts" in examples
                        else examples.get("thinking_process", [None] * len(examples["query"]))[i],
        }
        messages = build_messages(row)
        if messages is None:
            skip_count += 1
            continue

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ) # TODO 这里最后要不要加<|endoftext|>
        tokenized = tokenizer(
            text, truncation=True, max_length=max_seq_length, padding=False,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        user_messages = messages[:-1]
        user_text = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        user_tokenized = tokenizer(user_text, truncation=True, max_length=max_seq_length)
        prefix_len = len(user_tokenized["input_ids"])
        labels = [-100] * prefix_len + input_ids[prefix_len:]
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

    if skip_count > 0:
        print(f"  [Data] Skipped {skip_count} invalid samples")

    return model_inputs

class TrainingVisualCallback(TrainerCallback):
    """
    继承TrainerCallback 做一个训练画图:
      - train_loss (每个 logging_step)
      - eval_loss (每个 eval)
      - learning_rate
      - gradient norm
      - train_samples_per_second
    训练结束后画图保存到 output_dir/plots/
    """
    def __init__(self):
        self.train_losses = []       # (step, loss)
        self.eval_losses = []        # (step, loss)
        self.learning_rates = []     # (step, lr)
        self.grad_norms = []         # (step, norm)
        self.train_speeds = []       # (step, samples/sec)
        self.train_epochs = []       # (step, epoch)
        self.eval_epochs = []        # (step, epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step

        if "loss" in logs:
            self.train_losses.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.learning_rates.append((step, logs["learning_rate"]))
        if "grad_norm" in logs:
            self.grad_norms.append((step, logs["grad_norm"]))
        if "train_samples_per_second" in logs:
            self.train_speeds.append((step, logs["train_samples_per_second"]))
        if "epoch" in logs:
            if "eval_loss" in logs:
                self.eval_epochs.append((step, logs["epoch"]))
            else:
                self.train_epochs.append((step, logs["epoch"]))

    def on_train_end(self, args, state, control, **kwargs):
        output_dir = Path(args.output_dir)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Plots] Saving training plots to {plots_dir}/")
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, losses, "b-", linewidth=1.5, label="Train Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "train_loss.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ train_loss.png ({len(self.train_losses)} points)")
        if self.eval_losses:
            steps, losses = zip(*self.eval_losses)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, losses, "r-o", linewidth=1.5, markersize=4, label="Eval Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Eval Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "eval_loss.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ eval_loss.png ({len(self.eval_losses)} points)")
        if self.train_losses and self.eval_losses:
            fig, ax = plt.subplots(figsize=(10, 5))
            t_steps, t_losses = zip(*self.train_losses)
            e_steps, e_losses = zip(*self.eval_losses)
            ax.plot(t_steps, t_losses, "b-", linewidth=1, alpha=0.7, label="Train Loss")
            ax.plot(e_steps, e_losses, "r-o", linewidth=1.5, markersize=4, label="Eval Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Train vs Eval Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "train_vs_eval_loss.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ train_vs_eval_loss.png")
        if self.learning_rates:
            steps, lrs = zip(*self.learning_rates)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, lrs, "g-", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "learning_rate.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ learning_rate.png ({len(self.learning_rates)} points)")
        if self.grad_norms:
            steps, norms = zip(*self.grad_norms)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, norms, "m-", linewidth=1, alpha=0.7, label="Grad Norm")
            # 加一条 rolling average
            if len(norms) > 10:
                window = min(20, len(norms) // 5)
                rolling_avg = np.convolve(norms, np.ones(window) / window, mode="valid")
                rolling_steps = steps[window - 1:]
                ax.plot(rolling_steps, rolling_avg, "k--", linewidth=2,
                        label=f"Rolling Avg (w={window})")
            ax.set_xlabel("Step")
            ax.set_ylabel("Gradient L2 Norm")
            ax.set_title("Gradient Norm")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "grad_norm.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ grad_norm.png ({len(self.grad_norms)} points)")
        if self.train_speeds:
            steps, speeds = zip(*self.train_speeds)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, speeds, "c-", linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Samples/sec")
            ax.set_title("Training Speed")
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "train_speed.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ train_speed.png ({len(self.train_speeds)} points)")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            axes[0, 0].plot(steps, losses, "b-", linewidth=1)
            axes[0, 0].set_title("Train Loss")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].grid(True, alpha=0.3)
        if self.eval_losses:
            steps, losses = zip(*self.eval_losses)
            axes[0, 1].plot(steps, losses, "r-o", linewidth=1.5, markersize=3)
            axes[0, 1].set_title("Eval Loss")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].grid(True, alpha=0.3)
        if self.train_losses and self.eval_losses:
            t_s, t_l = zip(*self.train_losses)
            e_s, e_l = zip(*self.eval_losses)
            axes[0, 2].plot(t_s, t_l, "b-", linewidth=1, alpha=0.7, label="Train")
            axes[0, 2].plot(e_s, e_l, "r-o", linewidth=1.5, markersize=3, label="Eval")
            axes[0, 2].set_title("Train vs Eval")
            axes[0, 2].legend(fontsize=8)
            axes[0, 2].set_xlabel("Step")
            axes[0, 2].grid(True, alpha=0.3)
        if self.learning_rates:
            steps, lrs = zip(*self.learning_rates)
            axes[1, 0].plot(steps, lrs, "g-", linewidth=1)
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].grid(True, alpha=0.3)
        if self.grad_norms:
            steps, norms = zip(*self.grad_norms)
            axes[1, 1].plot(steps, norms, "m-", linewidth=1, alpha=0.7)
            if len(norms) > 10:
                w = min(20, len(norms) // 5)
                rolling = np.convolve(norms, np.ones(w) / w, mode="valid")
                axes[1, 1].plot(steps[w - 1:], rolling, "k--", linewidth=2)
            axes[1, 1].set_title("Gradient Norm")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].grid(True, alpha=0.3)
        if self.train_speeds:
            steps, speeds = zip(*self.train_speeds)
            axes[1, 2].plot(steps, speeds, "c-", linewidth=1)
            axes[1, 2].set_title("Train Speed (samples/sec)")
            axes[1, 2].set_xlabel("Step")
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(plots_dir / "dashboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ dashboard.png (combined)")

        # ── 保存原始数据为 CSV ──
        import csv
        metrics_csv = plots_dir / "metrics.csv"
        all_steps = set()
        for s, _ in (self.train_losses + self.eval_losses +
                     self.learning_rates + self.grad_norms + self.train_speeds):
            all_steps.add(s)
        all_steps = sorted(all_steps)

        step_map = {}
        for s, v in self.train_losses:
            step_map.setdefault(s, {})["train_loss"] = v
        for s, v in self.eval_losses:
            step_map.setdefault(s, {})["eval_loss"] = v
        for s, v in self.learning_rates:
            step_map.setdefault(s, {})["learning_rate"] = v
        for s, v in self.grad_norms:
            step_map.setdefault(s, {})["grad_norm"] = v
        for s, v in self.train_speeds:
            step_map.setdefault(s, {})["train_speed"] = v

        with open(metrics_csv, "w", newline="") as f:
            fields = ["step", "train_loss", "eval_loss", "learning_rate", "grad_norm", "train_speed"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in all_steps:
                row = {"step": s}
                row.update(step_map.get(s, {}))
                writer.writerow(row)
        print(f"  ✓ metrics.csv (raw data)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[Config] loaded from {args.config}")

    model_path = cfg["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg["model"].get("trust_remote_code", True)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[Tokenizer] loaded, vocab_size={tokenizer.vocab_size}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg["model"].get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    model = prepare_model_for_kbit_training(model)
    print(f"[Model] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["train_file"])
    print(f"[Data] loaded {len(df)} samples from {data_cfg['train_file']}")
    print(f"[Data] columns: {list(df.columns)}")
    if "thinking_process" in df.columns and "thoughts" not in df.columns:
        df = df.rename(columns={"thinking_process": "thoughts"})

    val_split = data_cfg.get("val_split", 0.05)
    val_size = int(len(df) * val_split)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    val_df = df.iloc[:val_size].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df[["query", "thoughts", "answer"]])
    val_dataset = Dataset.from_pandas(val_df[["query", "thoughts", "answer"]])
    max_seq_length = data_cfg.get("max_seq_length", 2048)
    print(f"[Data] tokenizing (max_seq_length={max_seq_length})...")

    train_dataset = train_dataset.map(
        lambda x: preprocess(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )
    print(f"[Data] train={len(train_dataset)}, val={len(val_dataset)}")

    t_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        max_grad_norm=t_cfg["max_grad_norm"],
        bf16=t_cfg["bf16"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        eval_strategy=t_cfg["eval_strategy"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        report_to=t_cfg.get("report_to", "none"),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 4),
        seed=t_cfg.get("seed", 42),
        remove_unused_columns=False,
        logging_dir=os.path.join(t_cfg["output_dir"], "tensorboard"),
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    viz_callback = TrainingVisualCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[viz_callback],
    )
    effective_batch = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
    print("\n" + "=" * 60)
    print("Starting LoRA fine-tuning...")
    print(f"  Model:         {model_path}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Epochs:        {t_cfg['num_train_epochs']}")
    print(f"  Batch size:    {t_cfg['per_device_train_batch_size']} x "
          f"{t_cfg['gradient_accumulation_steps']} = {effective_batch}")
    print(f"  Learning rate: {t_cfg['learning_rate']}")
    print(f"  LoRA rank:     {lora_cfg['r']}")
    print(f"  Output dir:    {t_cfg['output_dir']}")
    print(f"  TensorBoard:   tensorboard --logdir {t_cfg['output_dir']}/tensorboard")
    print("=" * 60 + "\n")

    trainer.train()
    save_path = os.path.join(t_cfg["output_dir"], "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n[Done] LoRA adapter saved to {save_path}")
    plots_dir = os.path.join(t_cfg["output_dir"], "plots")
    print(f"\n[Plots] 查看训练曲线: {plots_dir}/")


if __name__ == "__main__":
    main()
