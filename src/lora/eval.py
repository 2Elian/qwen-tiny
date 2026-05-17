#!/usr/bin/env python3
"""
NL2SQL Evaluation Script
评估 Qwen3 模型在 text-to-SQL 任务上的准确性。

评估指标:
  1. Exact Match (EM)     — 预测 SQL 与 ground truth 完全一致（归一化后）
  2. Partial Match (PM)   — SQL 关键组件匹配（SELECT/WHERE/JOIN/GROUP BY/ORDER BY）
  3. Token-level F1       — SQL token 级别的 precision/recall/F1
  4. Execution Accuracy   — 两条 SQL 执行结果是否相同（需要数据库，可选）

用法:
  # 评估原始模型
  python eval.py --model /data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b

  # 评估 LoRA 微调后的模型
  python eval.py --model /data1/nuist_llm/TrainLLM/attention-residuals-reproduction/src/lora/output/lora-nl2sql/qkvogud/checkpoint-1030 \
                 --base_model /data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b

  # 限制评估数量
  python eval.py --model xxx --max_samples 100

  # 保存详细结果
  python eval.py --model xxx --output results.json
"""

import os
import re
import json
import argparse
import sys
from collections import Counter
from typing import Optional

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════
# 1. SQL 归一化 & 组件解析
# ════════════════════════════════════════════════════════════════

def normalize_sql(sql: str) -> str:
    """
    归一化 SQL：去除无关差异，保留语义等价性。
    - 去除 markdown 代码块标记
    - 统一空白
    - 统一大小写（关键字大写，标识符保留）
    - 去除尾部分号
    - 去除表别名 (AS T1 → 去掉)
    """
    # 去除 markdown 代码块
    sql = re.sub(r'```sql\s*', '', sql)
    sql = re.sub(r'```\s*', '', sql)
    sql = sql.strip()

    # 去除尾部分号
    sql = sql.rstrip(';').strip()

    # 统一空白（多个空格/换行 → 单空格）
    sql = re.sub(r'\s+', ' ', sql).strip()

    # 关键字大写
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'ON',
        'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'FULL', 'CROSS',
        'GROUP', 'BY', 'ORDER', 'ASC', 'DESC', 'HAVING', 'LIMIT',
        'OFFSET', 'UNION', 'ALL', 'DISTINCT', 'AS', 'COUNT', 'SUM',
        'AVG', 'MIN', 'MAX', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'BETWEEN', 'LIKE', 'EXISTS', 'IS', 'NULL', 'TRUE', 'FALSE',
        'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE',
        'CREATE', 'TABLE', 'ALTER', 'DROP', 'INDEX', 'VIEW',
        'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'NOT', 'TEXT',
        'INTEGER', 'REAL', 'BLOB', 'NUMERIC', 'INT', 'BIGINT',
    ]
    # 先占位保护字符串字面量
    strings = []
    def save_string(m):
        strings.append(m.group(0))
        return f"__STR{len(strings)-1}__"
    sql = re.sub(r"'[^']*'", save_string, sql, flags=re.IGNORECASE)

    # 大写化关键字（简单匹配，不处理子串）
    for kw in sorted(keywords, key=len, reverse=True):
        sql = re.sub(r'\b' + kw + r'\b', kw, sql, flags=re.IGNORECASE)

    # 恢复字符串
    for i, s in enumerate(strings):
        sql = sql.replace(f"__STR{i}__", s)

    return sql.strip()


def extract_sql_components(sql: str) -> dict:
    """
    从 SQL 中提取关键组件，用于 component matching。
    """
    components = {
        "select_cols": [],
        "from_tables": [],
        "where_conditions": [],
        "join_conditions": [],
        "group_by": [],
        "order_by": [],
        "having": [],
        "limit": None,
        "aggregates": [],
        "distinct": False,
    }

    sql_norm = normalize_sql(sql)

    # DISTINCT
    components["distinct"] = "DISTINCT" in sql_norm.upper()

    # SELECT 列
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM\b', sql_norm, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_str = select_match.group(1)
        # 分割列（注意处理嵌套函数）
        cols = split_sql_cols(select_str)
        components["select_cols"] = [normalize_col(c.strip()) for c in cols]

        # 聚合函数
        agg_funcs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
        for func in agg_funcs:
            if func + '(' in select_str.upper():
                components["aggregates"].append(func)

    # FROM 表
    from_match = re.search(r'FROM\s+(.*?)(?:\s+WHERE\b|\s+GROUP\b|\s+ORDER\b|\s+LIMIT\b|\s+HAVING\b|\s*$)',
                           sql_norm, re.IGNORECASE | re.DOTALL)
    if from_match:
        from_str = from_match.group(1)
        # 提取表名（包括别名）
        tables = re.findall(r'(\w+(?:\.\w+)?)\s*(?:AS\s+)?(\w+)?', from_str, re.IGNORECASE)
        components["from_tables"] = [t[0] for t in tables]

        # JOIN 条件
        join_matches = re.findall(
            r'(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN\s+(\w+)\s*(?:AS\s+)?(\w+)?\s+ON\s+(.*?)(?=\s+(?:LEFT|RIGHT|INNER|FULL|CROSS|WHERE|GROUP|ORDER|LIMIT|HAVING|$))',
            sql_norm, re.IGNORECASE
        )
        for join_table, alias, on_cond in join_matches:
            components["join_conditions"].append(normalize_col(on_cond.strip()))

    # WHERE 条件
    where_match = re.search(
        r'WHERE\s+(.*?)(?:\s+GROUP\b|\s+ORDER\b|\s+LIMIT\b|\s+HAVING\b|\s*$)',
        sql_norm, re.IGNORECASE | re.DOTALL
    )
    if where_match:
        where_str = where_match.group(1)
        # 按 AND/OR 分割
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_str, flags=re.IGNORECASE)
        components["where_conditions"] = [normalize_col(c.strip()) for c in conditions if c.strip()]

    # GROUP BY
    group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+HAVING\b|\s+ORDER\b|\s+LIMIT\b|\s*$)',
                            sql_norm, re.IGNORECASE | re.DOTALL)
    if group_match:
        cols = split_sql_cols(group_match.group(1))
        components["group_by"] = [normalize_col(c.strip()) for c in cols]

    # ORDER BY
    order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT\b|\s*$)',
                            sql_norm, re.IGNORECASE | re.DOTALL)
    if order_match:
        order_str = order_match.group(1).strip()
        components["order_by"] = [normalize_col(c.strip()) for c in split_sql_cols(order_str)]

    # HAVING
    having_match = re.search(r'HAVING\s+(.*?)(?:\s+ORDER\b|\s+LIMIT\b|\s*$)',
                             sql_norm, re.IGNORECASE | re.DOTALL)
    if having_match:
        components["having"] = [normalize_col(having_match.group(1).strip())]

    # LIMIT
    limit_match = re.search(r'LIMIT\s+(\d+)', sql_norm, re.IGNORECASE)
    if limit_match:
        components["limit"] = int(limit_match.group(1))

    return components


def split_sql_cols(s: str) -> list[str]:
    """按逗号分割 SQL 列，但不分割嵌套括号内的逗号。"""
    cols = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            cols.append(''.join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        cols.append(''.join(current).strip())
    return cols


def normalize_col(col: str) -> str:
    """归一化列引用：去除别名前缀、统一大小写。"""
    col = re.sub(r'\bT\d+\.', '', col)  # 去除 T1. T2. 等别名
    col = re.sub(r'\b\w+\.', '', col)   # 去除所有 table. 前缀（保守）
    return col.strip().upper()


def sql_tokens(sql: str) -> list[str]:
    """将 SQL 分解为 token 列表。"""
    sql = normalize_sql(sql)
    # 按非字母数字字符分割
    tokens = re.findall(r'\w+|[^\s\w]', sql)
    return [t.upper() for t in tokens]


# ════════════════════════════════════════════════════════════════
# 2. 评估指标
# ════════════════════════════════════════════════════════════════

def exact_match(pred: str, gold: str) -> bool:
    """归一化后的精确匹配。"""
    return normalize_sql(pred) == normalize_sql(gold)


def token_f1(pred: str, gold: str) -> dict:
    """Token 级别的 Precision / Recall / F1。"""
    pred_tokens = sql_tokens(pred)
    gold_tokens = sql_tokens(gold)

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # 交集 = 每个 token 取 min(count)
    intersection = sum((pred_counter & gold_counter).values())

    precision = intersection / len(pred_tokens) if pred_tokens else 0.0
    recall = intersection / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def component_match(pred: str, gold: str) -> dict:
    """
    SQL 组件匹配：分别评估 SELECT / WHERE / JOIN / GROUP BY / ORDER BY。
    返回每个组件的 Jaccard 相似度。
    """
    pred_comp = extract_sql_components(pred)
    gold_comp = extract_sql_components(gold)

    results = {}
    for key in ["select_cols", "where_conditions", "join_conditions",
                 "group_by", "order_by", "aggregates"]:
        pred_set = set(pred_comp.get(key, []))
        gold_set = set(gold_comp.get(key, []))

        if not pred_set and not gold_set:
            results[key] = 1.0  # 都为空，视为匹配
        elif not pred_set or not gold_set:
            results[key] = 0.0
        else:
            intersection = pred_set & gold_set
            union = pred_set | gold_set
            results[key] = len(intersection) / len(union) if union else 0.0

    # DISTINCT 匹配
    results["distinct"] = 1.0 if pred_comp["distinct"] == gold_comp["distinct"] else 0.0

    # LIMIT 匹配
    results["limit"] = 1.0 if pred_comp["limit"] == gold_comp["limit"] else 0.0

    return results


# ════════════════════════════════════════════════════════════════
# 3. 模型推理
# ════════════════════════════════════════════════════════════════

def load_model(model_path: str, base_model_path: Optional[str] = None, device: str = "auto"):
    """加载模型，支持 LoRA 和原始模型。"""
    print(f"[Model] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if base_model_path is None else base_model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if base_model_path:
        # LoRA 模型：先加载 base model，再加载 adapter
        print(f"[Model] Loading base model from {base_model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        from peft import PeftModel
        print(f"[Model] Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 合并 LoRA 以加速推理
        print("[Model] LoRA adapter merged.")
    else:
        print(f"[Model] Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Model] Ready. Parameters: {param_count:.2f}B")
    return model, tokenizer


def generate_sql(model, tokenizer, query: str, max_new_tokens: int = 1024) -> str:
    """给定 query，让模型生成 SQL。"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a SQL expert. Given a database schema and a natural language question, "
                "think step by step and generate the correct SQL query. "
                "Output the SQL query inside ```sql ... ``` code blocks."
            ),
        },
        {"role": "user", "content": query},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只取新生成的 token
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


def extract_sql_from_response(response: str) -> str:
    """从模型输出中提取 SQL。"""
    # 尝试从 ```sql ... ``` 中提取
    match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 尝试从 <think>...</think> 后面提取
    think_end = response.rfind("</think>")
    if think_end != -1:
        after_think = response[think_end + len("</think>"):].strip()
        # 再试一次 code block
        match = re.search(r'```sql\s*(.*?)\s*```', after_think, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # 没有 code block，取第一行非空内容作为 SQL
        for line in after_think.split('\n'):
            line = line.strip()
            if line and line.upper().startswith('SELECT'):
                return line
        return after_think

    # 最后 fallback：找第一个 SELECT 语句
    match = re.search(r'(SELECT\b.*?)(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return response.strip()


# ════════════════════════════════════════════════════════════════
# 4. 主评估流程
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NL2SQL Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (LoRA adapter dir or base model dir)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path (required if --model is a LoRA adapter)")
    parser.add_argument("--data", type=str,
                        default="/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/nl2sql/cot-qa-cold-start.csv",
                        help="Eval dataset path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (None = all)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max new tokens for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (currently only 1 supported)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for detailed results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda:0/cpu)")
    args = parser.parse_args()

    # ── 加载模型 ──
    model, tokenizer = load_model(args.model, args.base_model, args.device)

    # ── 加载数据 ──
    print(f"[Data] Loading from {args.data}...")
    df = pd.read_csv(args.data)
    # cot-qa.csv 列: query, answer, thinking_process
    # cot-qa-cold-start.csv 列: query, answer, thoughts
    # 统一列名
    if "thinking_process" in df.columns:
        df = df.rename(columns={"thinking_process": "thoughts"})

    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"[Data] {len(df)} samples to evaluate\n")

    # ── 评估 ──
    results = []
    total_em = 0
    total_f1_sum = 0.0
    total_component_scores = {}
    component_keys = ["select_cols", "where_conditions", "join_conditions",
                      "group_by", "order_by", "aggregates", "distinct", "limit"]

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = row["query"]
        gold_sql = row["answer"]
        thoughts = row.get("thoughts", "")

        # 模型推理
        response = generate_sql(model, tokenizer, query, args.max_new_tokens)
        pred_sql = extract_sql_from_response(response)

        # 计算指标
        em = exact_match(pred_sql, gold_sql)
        f1 = token_f1(pred_sql, gold_sql)
        comp = component_match(pred_sql, gold_sql)

        total_em += int(em)
        total_f1_sum += f1["f1"]
        for k in component_keys:
            total_component_scores[k] = total_component_scores.get(k, 0.0) + comp.get(k, 0.0)

        results.append({
            "idx": idx,
            "query": query[:200] + "..." if len(query) > 200 else query,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": em,
            "token_f1": f1["f1"],
            "component_scores": comp,
        })

        # 打印部分结果
        status = "✓" if em else "✗"
        print(f"\n  [{status}] Sample {idx}")
        print(f"    Gold:  {gold_sql[:120]}...")
        print(f"    Pred:  {pred_sql[:120]}...")
        print(f"    EM={em}  F1={f1['f1']:.3f}")

    # ── 汇总 ──
    n = len(df)
    avg_f1 = total_f1_sum / n if n > 0 else 0
    avg_components = {k: v / n for k, v in total_component_scores.items()}

    print("\n" + "=" * 60)
    print(f"  Evaluation Results ({n} samples)")
    print("=" * 60)
    print(f"  Exact Match Accuracy:  {total_em}/{n} = {total_em/n*100:.2f}%")
    print(f"  Avg Token F1:          {avg_f1:.4f}")
    print()
    print("  Component Matching (Jaccard):")
    for k, v in avg_components.items():
        print(f"    {k:20s}: {v:.4f}")
    print("=" * 60)

    # ── 保存详细结果 ──
    if args.output:
        output_data = {
            "model": args.model,
            "base_model": args.base_model,
            "data": args.data,
            "num_samples": n,
            "exact_match_accuracy": total_em / n if n > 0 else 0,
            "avg_token_f1": avg_f1,
            "component_matching": avg_components,
            "details": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[Output] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
