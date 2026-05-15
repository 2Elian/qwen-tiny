#!/usr/bin/env python3
# author: 2Elian
# github: https://github.com/2Elian/qwen-tiny
"""
LoRA fine-tuning for Qwen3 on NL2SQL CoT data.
Usage:
    python main.py                          # 使用默认config.yaml
    python main.py --config my_config.yaml  # 指定你的配置文件
"""

import os
import sys
import yaml
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_messages(row: dict) -> list[dict]:
    """
    将 CSV 一行映射为 chat messages。

    CSV 列:
      - query:    user input (Database Schema + Question + Hint)
      - answer:   SQL(```sql...``` 格式)
      - thoughts: 思考过程（<think>...</think> + <answer>...</answer>)

    训练目标: 让模型输出 thoughts(包含中文的think + answer)
    """
    query = row["query"].strip()
    thoughts = row["thoughts"].strip()

    # 清理 thoughts 中的 <answer>...</answer> 标签，保留纯内容
    # 因为模型输出格式是 <think>...</think> 后跟回答
    thoughts = thoughts.replace("<answer>", "").replace("</answer>", "").strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a SQL expert. Given a database schema and a natural language question, "
                "think step by step and generate the correct SQL query."
            ),
        },
        {"role": "user", "content": query},
        {"role": "assistant", "content": thoughts},
    ]
    return messages


def preprocess(examples: dict, tokenizer, max_seq_length: int) -> dict:
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    ignore_data_count = 0
    for i in range(len(examples["query"])):
        if examples["thoughts"][i] is None:
            ignore_data_count+=1
            print(f"跳过无效的数据，当前无效的数据数量为：{ignore_data_count}")
            continue
        row = {
            "query": examples["query"][i],
            "answer": examples["answer"][i],
            "thoughts": examples["thoughts"][i],
        }
        
        messages = build_messages(row)

        # 用 chat template 编码 --> 聊天模板: <|im_start|> <|im_end|>
        """
        example:
            chat_template --> 进入到这个函数里面首先会找到模型对应的聊天模板：chat_template = self.get_chat_template(chat_template, tools)
                聊天模板是加载的/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b/tokenizer_config.json
            template_kwargs --> 找到对应的结束token和pad token
            rendered_chat --> 如果不是batch的话 返回值就是这个
                '<|im_start|>system\nYou are a SQL expert. Given a database schema and a natural language question, think step by step and generate the correct SQL query.<|im_end|>\n<|im_start|>user\nDatabase Schema\n###\nCREATE TABLE Sales Orders\n(\n\tOrderNumber TEXT constraint "Sales Orders_pk" primary key,\n\tWarehouseCode TEXT, -- examples: `WARE-PUJ1005`| `Warehouse Code` description: Warehouse Code value description: if the warehouse code is the same, it means this product is shipped from the same ware house\n\t_StoreID INTEGER constraint "Sales Orders_Store Locations_StoreID_fk" references "Store Locations"(StoreID), --\n);\n\nCREATE TABLE Store Locations\n(\n\tStoreID INTEGER constraint "Store Locations_pk" primary key,\n\tLatitude REAL, -- description: Latitude\n\tLongitude REAL, -- description: Longitude value description: coordinates or detailed position: (Latitude, Longitude)\n);\n\n###\nQuestion: \nAt what Latitude and Longitude is the store that has used the WARE-PUJ1005 warehouse the fewest times? \n\nHint:\nWARE-PUJ1005 is the WarehouseCode; fewest times refers to Min (Count(WarehouseCode))<|im_end|>\n<|im_start|>assistant\n<think>\n1. **理解需求**：用户需要找到使用仓库代码为\'WARE-PUJ1005\'次数最少的商店的经纬度。\n\n2. **分析表结构**：\n   - `Sales Orders`表包含订单信息，其中`_StoreID`外键关联到`Store Locations`表的`StoreID`。\n   - `Store Locations`表存储商店的地理位置信息。\n\n3. **确定连接条件**：通过`Sales Orders._StoreID`与`Store Locations.StoreID`进行JOIN操作，关联两个表。\n\n4. **过滤条件**：筛选出`WarehouseCode`为\'WARE-PUJ1005\'的订单记录。\n\n5. **聚合统计**：按`_StoreID`分组，统计每个商店使用该仓库的次数。\n\n6. **排序与限制**：按使用次数升序排列，取第一条记录（即次数最少的商店）。\n\n7. **处理可能的问题**：确保分组后能正确获取对应的经纬度，且无重复或错误。\n</think>\n\n```sql\nSELECT \n    "Store Locations".Latitude, \n    "Store Locations".Longitude \nFROM \n    "Sales Orders" \nJOIN \n    "Store Locations" ON "Sales Orders"._StoreID = "Store Locations".StoreID \nWHERE \n    "Sales Orders".WarehouseCode = \'WARE-PUJ1005\' \nGROUP BY \n    "Sales Orders"._StoreID \nORDER BY \n    COUNT(*) ASC \nLIMIT 1;\n```<|im_end|>\n'
        """
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        # 此时的len(input_ids)就是输入的token的数量(还没做padding)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # 构建 labels --> 只在 assistant 回复部分计算 loss
        user_messages = messages[:-1]  # system + user
        user_text = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        ) # add_generation_prompt=True代表在结尾添加<|im_start|>assistant\n。可以对比text和user_text看看 --> 但是要注意：推理的时候一定要add_generation_prompt=True，和训练保持对齐
        user_tokenized = tokenizer(user_text, truncation=True, max_length=max_seq_length)
        prefix_len = len(user_tokenized["input_ids"])

        # labels --> prefix 部分设为 -100
        labels = [-100] * prefix_len + input_ids[prefix_len:]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

    return model_inputs


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
    print(f"[Tokenizer] loaded from {model_path}, vocab_size={tokenizer.vocab_size}")
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
    print(f"[Model] loaded, params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

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

    val_split = data_cfg.get("val_split", 0.05)
    val_size = int(len(df) * val_split)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    val_df = df.iloc[:val_size].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # tokenize
    max_seq_length = data_cfg.get("max_seq_length", 2048)
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
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("Starting LoRA fine-tuning...")
    print(f"  Model:        {model_path}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Epochs:        {t_cfg['num_train_epochs']}")
    print(f"  Batch size:    {t_cfg['per_device_train_batch_size']} x {t_cfg['gradient_accumulation_steps']} (grad accum) = {t_cfg['per_device_train_batch_size'] * t_cfg['gradient_accumulation_steps']}")
    print(f"  Learning rate: {t_cfg['learning_rate']}")
    print(f"  LoRA rank:     {lora_cfg['r']}")
    print(f"  Output dir:    {t_cfg['output_dir']}")
    print("=" * 60 + "\n")

    trainer.train()
    save_path = os.path.join(t_cfg["output_dir"], "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n[Done] LoRA adapter saved to {save_path}")


if __name__ == "__main__":
    main()
