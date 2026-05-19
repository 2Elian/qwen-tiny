#!/usr/bin/env python3
"""
SFT training on DeepMath-103K with PyTorch FSDP (Fully Sharded Data Parallel).

Usage:
    torchrun --nproc_per_node=8 main_fsdp.py
    torchrun --nproc_per_node=8 main_fsdp.py --config my_config.yaml

Features:
  - FSDP "full_shard" — shards model parameters, gradients, and optimizer states
    across all GPUs (analogous to ZeRO-3).
  - Auto-wrapping of transformer layers for efficient communication.
  - CPU offload via fsdp_config.
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

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, load_and_prepare_data, DeepMathDataCollator, TrainingVisualCallback


def parse_args():
    parser = argparse.ArgumentParser(description="SFT with FSDP")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--fsdp_mode", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op", "hybrid_shard", "offload"],
                        help="FSDP sharding strategy")
    # General data mixing
    parser.add_argument("--general_data_path", type=str, default=None,
                        help="Path to preprocessed general data (parquet/jsonl)")
    parser.add_argument("--general_mix_ratio", type=float, default=0.2,
                        help="Fraction of general data in training mix (0.0~1.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    if not config_path.exists():
        config_path = Path(args.config)

    cfg = load_config(str(config_path))
    print(f"[Config] {config_path}")

    # ── 1. Tokenizer ──────────────────────────────────────────────
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}")

    # ── 2. Model ──────────────────────────────────────────────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    # For FSDP, we don't move the model to any device — FSDP handles it.
    # We need to make sure the model is initialized on "meta" or "cpu" for FSDP to shard.
    # HuggingFace Trainer with fsdp="full_shard" will handle the wrapping.
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model.gradient_checkpointing_enable()
    print(f"[Model] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ── 3. Data ───────────────────────────────────────────────────
    data_cfg = cfg["data"]
    train_dataset, val_dataset = load_and_prepare_data(
        tokenizer=tokenizer,
        data_dir=data_cfg["data_dir"],
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        val_split=data_cfg.get("val_split", 0.02),
        seed=cfg["training"].get("seed", 42),
        num_files=data_cfg.get("num_files"),
        general_data_path=args.general_data_path or data_cfg.get("general_data_path"),
        general_mix_ratio=args.general_mix_ratio if args.general_data_path else data_cfg.get("general_mix_ratio", 0.0),
    )
    print(f"[Data] train={len(train_dataset)}, val={len(val_dataset)}")

    # ── 4. Training ───────────────────────────────────────────────
    t_cfg = cfg["training"]
    output_dir = t_cfg["output_dir"]

    # FSDP config passed to TrainingArguments
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": False,
        "fsdp_use_orig_params": False,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_sync_module_states": True,
    }

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
        # ── FSDP flags ──
        fsdp=args.fsdp_mode,
        fsdp_config=fsdp_config,
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

    # ── 5. Run ────────────────────────────────────────────────────
    eff_batch = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
    n_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    print("\n" + "=" * 60)
    print("[FSDP Training]")
    print(f"  Model:        {model_cfg['name_or_path']}")
    print(f"  FSDP mode:    {args.fsdp_mode}")
    print(f"  Train:        {len(train_dataset)} samples")
    print(f"  Val:          {len(val_dataset)} samples")
    print(f"  Epochs:       {t_cfg['num_train_epochs']}")
    print(f"  Batch/GPU:    {t_cfg['per_device_train_batch_size']}")
    print(f"  GradAccum:    {t_cfg['gradient_accumulation_steps']}")
    print(f"  Effective:    {eff_batch} (per step) / {eff_batch * n_gpus} (global)")
    print(f"  LR:           {t_cfg['learning_rate']}")
    print(f"  Max seq len:  {data_cfg.get('max_seq_length', 2048)}")
    print(f"  Output:       {output_dir}")
    print("=" * 60 + "\n")

    trainer.train()

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[Done] model saved to {final_path}")


if __name__ == "__main__":
    main()
