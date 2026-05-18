#!/usr/bin/env python3
"""
SFT training on DeepMath-103K with DeepSpeed Tensor Parallelism + ZeRO-3.

Usage:
    torchrun --nproc_per_node=8 main_tp_zero3.py
    torchrun --nproc_per_node=8 main_tp_zero3.py --config my_config.yaml

Architecture:
  - Tensor Parallelism (TP): splits each transformer layer's weight matrices across
    `tp_size` GPUs within the same TP group. Reduces per-GPU memory and communication
    for large models.
  - ZeRO-3: shards optimizer states, gradients, and parameters across DP groups
    (different TP groups do not share data-parallel state).

  With 8 GPUs and tp_size=2:
    - 4 DP groups (rank 0-1, 2-3, 4-5, 6-7)
    - Within each DP group, 2 GPUs collaborate via TP on one layer's computation
    - ZeRO-3 shards across the 4 DP groups

  This approach bypasses HuggingFace Trainer to give full control over the
  DeepSpeed engine initialization with TP enabled.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import deepspeed

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, load_and_prepare_data, DeepMathDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="SFT with DeepSpeed TP + ZeRO-3")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set by torchrun / deepspeed launcher")
    parser.add_argument("--tp_size", type=int, default=2,
                        help="Tensor parallelism group size")
    parser.add_argument("--deepspeed_config", type=str, default="ds_tp_zero3.json",
                        help="DeepSpeed config (JSON, generated from YAML)")
    return parser.parse_args()


def build_deepspeed_config(cfg: dict, tp_size: int, output_dir: str) -> dict:
    """Build a DeepSpeed config dict with TP + ZeRO-3."""

    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",

        "tensor_parallel": {
            "enabled": True,
            "tp_size": tp_size,
            "tp_grain": "layer",
        },

        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },

        "bf16": {
            "enabled": cfg["training"].get("bf16", True),
        },

        "gradient_clipping": cfg["training"].get("max_grad_norm", 1.0),

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg["training"]["learning_rate"],
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": cfg["training"].get("weight_decay", 0.01),
            },
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": cfg["training"]["learning_rate"],
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        },

        "wall_clock_breakdown": False,
    }

    return ds_config


def get_dataloader(dataset, batch_size: int, tokenizer, max_length: int,
                   sampler=None, shuffle: bool = False, num_workers: int = 4):
    """Create a DataLoader with DeepMathDataCollator."""
    collator = DeepMathDataCollator(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )


def main():
    args = parse_args()

    # ── 0. Init distributed ───────────────────────────────────────
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # ── 1. Config ─────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    cfg = load_config(str(config_path))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    t_cfg = cfg["training"]
    output_dir = t_cfg["output_dir"]

    # ── 2. Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 3. Data ───────────────────────────────────────────────────
    train_dataset, val_dataset = load_and_prepare_data(
        tokenizer=tokenizer,
        data_dir=data_cfg["data_dir"],
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        val_split=data_cfg.get("val_split", 0.02),
        seed=t_cfg.get("seed", 42),
        num_files=data_cfg.get("num_files"),
    )

    train_sampler = DistributedSampler(train_dataset)
    train_loader = get_dataloader(
        train_dataset,
        batch_size=t_cfg["per_device_train_batch_size"],
        tokenizer=tokenizer,
        max_length=data_cfg.get("max_seq_length", 2048),
        sampler=train_sampler,
        num_workers=t_cfg.get("dataloader_num_workers", 4),
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = get_dataloader(
        val_dataset,
        batch_size=t_cfg["per_device_eval_batch_size"],
        tokenizer=tokenizer,
        max_length=data_cfg.get("max_seq_length", 2048),
        sampler=val_sampler,
        num_workers=t_cfg.get("dataloader_num_workers", 4),
    )

    # ── 4. Model (load on CPU, DeepSpeed will shard it) ──────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model.gradient_checkpointing_enable()

    # ── 5. DeepSpeed Engine (TP + ZeRO-3) ────────────────────────
    ds_config = build_deepspeed_config(cfg, args.tp_size, output_dir)

    if rank == 0:
        print(f"[DeepSpeed TP+ZeRO-3] tp_size={args.tp_size}, world_size={world_size}")
        print(f"  DP groups: {world_size // args.tp_size}")
        print(f"  Config: {json.dumps(ds_config, indent=2)[:2000]}")

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )

    # ── 6. Training loop ──────────────────────────────────────────
    epochs = t_cfg["num_train_epochs"]
    grad_accum_steps = t_cfg["gradient_accumulation_steps"]
    global_step = 0
    model_engine.train()

    if rank == 0:
        eff_batch = t_cfg["per_device_train_batch_size"] * grad_accum_steps * world_size
        print("\n" + "=" * 60)
        print("[TP + ZeRO-3 Training]")
        print(f"  Model:        {model_cfg['name_or_path']}")
        print(f"  TP size:      {args.tp_size}")
        print(f"  World size:   {world_size}")
        print(f"  Train:        {len(train_dataset)} samples")
        print(f"  Val:          {len(val_dataset)} samples")
        print(f"  Epochs:       {epochs}")
        print(f"  Batch/GPU:    {t_cfg['per_device_train_batch_size']}")
        print(f"  GradAccum:    {grad_accum_steps}")
        print(f"  Global batch: {eff_batch}")
        print(f"  LR:           {t_cfg['learning_rate']}")
        print(f"  Max seq len:  {data_cfg.get('max_seq_length', 2048)}")
        print(f"  Output:       {output_dir}")
        print("=" * 60 + "\n")

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        steps_in_epoch = 0

        for micro_step, batch in enumerate(train_loader):
            batch = {k: v.cuda(local_rank) for k, v in batch.items()}

            loss = model_engine(batch)["loss"]
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            steps_in_epoch += 1
            global_step += 1

            if rank == 0 and global_step % t_cfg["logging_steps"] == 0:
                avg_loss = total_loss / max(steps_in_epoch, 1)
                lr = model_engine.get_lr()[0]
                print(f"  [Step {global_step}] loss={avg_loss:.4f} lr={lr:.2e}")

        # ── End of epoch: eval & save ─────────────────────────────
        model_engine.eval()
        eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda(local_rank) for k, v in batch.items()}
                loss = model_engine(batch)["loss"]
                eval_loss += loss.item()
                eval_steps += 1

        avg_train_loss = total_loss / max(steps_in_epoch, 1)
        avg_eval_loss = eval_loss / max(eval_steps, 1)

        if rank == 0:
            print(f"  [Epoch {epoch+1}/{epochs}] train_loss={avg_train_loss:.4f} "
                  f"eval_loss={avg_eval_loss:.4f}")

            ckpt_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model_engine.save_checkpoint(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  [Save] checkpoint -> {ckpt_dir}")

        model_engine.train()

    # ── 7. Final save ─────────────────────────────────────────────
    if rank == 0:
        final_path = os.path.join(output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        model_engine.save_checkpoint(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\n[Done] model saved to {final_path}")


if __name__ == "__main__":
    main()
