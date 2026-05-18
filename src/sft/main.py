#!/usr/bin/env python3
"""
SFT training on DeepMath-103K with DeepSpeed ZeRO-3.

Usage:
    CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 main.py
    torchrun --nproc_per_node=2 main.py --config my_config.yaml

Features:
  - ZeRO-3: shards optimizer states, gradients, and parameters across all GPUs.
  - CPU offload for optimizer states and parameters (configurable in ds_zero3.json).
  - Supports single-node multi-GPU and multi-node setups.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# Shared modules live in the same package
sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, load_and_prepare_data, DeepMathDataCollator, TrainingVisualCallback


def parse_args():
    parser = argparse.ArgumentParser(description="SFT with DeepSpeed ZeRO-3")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torchrun)")
    parser.add_argument("--deepspeed", type=str, default="ds_zero3.json",
                        help="DeepSpeed config file (JSON)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config paths relative to this script
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    if not config_path.exists():
        config_path = Path(args.config)

    cfg = load_config(str(config_path))
    print(f"[Config] {config_path}")

    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}, pad_token={tokenizer.pad_token}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    print(f"[Model] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    data_cfg = cfg["data"]
    train_dataset, val_dataset = load_and_prepare_data(
        tokenizer=tokenizer,
        data_dir=data_cfg["data_dir"],
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        val_split=data_cfg.get("val_split", 0.02),
        seed=cfg["training"].get("seed", 42),
        num_files=data_cfg.get("num_files"),
    )
    print(f"[Data] train={len(train_dataset)}, val={len(val_dataset)}")

    t_cfg = cfg["training"]
    output_dir = t_cfg["output_dir"]

    # Resolve deepspeed config path
    ds_config_path = script_dir / args.deepspeed
    if not ds_config_path.exists():
        ds_config_path = Path(args.deepspeed)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        max_grad_norm=t_cfg["max_grad_norm"],
        bf16=t_cfg.get("bf16", True),
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg.get("save_steps", 500),
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg.get("eval_steps", 500),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=t_cfg.get("report_to", "none"),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 4),
        seed=t_cfg.get("seed", 42),
        remove_unused_columns=False,
        deepspeed=str(ds_config_path),
        ddp_backend="nccl",
        logging_dir=os.path.join(output_dir, "tensorboard"),
    )

    data_collator = DeepMathDataCollator(
        tokenizer=tokenizer,
        max_length=data_cfg.get("max_seq_length", 2048),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[TrainingVisualCallback()],
    )

    eff_batch = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
    n_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    print("\n" + "=" * 60)
    print("[DeepSpeed ZeRO-3 Training]")
    print(f"  Model:        {model_cfg['name_or_path']}")
    print(f"  Train:        {len(train_dataset)} samples")
    print(f"  Val:          {len(val_dataset)} samples")
    print(f"  Epochs:       {t_cfg['num_train_epochs']}")
    print(f"  Batch/GPU:    {t_cfg['per_device_train_batch_size']}")
    print(f"  GradAccum:    {t_cfg['gradient_accumulation_steps']}")
    print(f"  Effective:    {eff_batch} (per step) / {eff_batch * n_gpus} (global)")
    print(f"  LR:           {t_cfg['learning_rate']}")
    print(f"  Max seq len:  {data_cfg.get('max_seq_length', 2048)}")
    print(f"  DeepSpeed:    {ds_config_path}")
    print(f"  Output:       {output_dir}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final checkpoint
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[Done] model saved to {final_path}")


if __name__ == "__main__":
    main()
