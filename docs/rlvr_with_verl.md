# 使用 verl 训练 RLVR 模型：GRPO / DAPO / GSPO 完整指南

> 对应源码: `src/verl-grpo/verl/`
> 我们自己的数据: `data/DeepMath-103K/filtered/`


# 奖励函数设计的评估维度

```text
一、答案正确性（主导信号，权重 1.0）                                                                                                                                                                                 
   
  ┌────────────────┬───────┬───────────────────────────────────────────────────────────────┐                                                                                                                           
  │      信号      │ 类型  │                             说明                              │
  ├────────────────┼───────┼───────────────────────────────────────────────────────────────┤                                                                                                                           
  │ answer_correct │ 0 / 1 │ \boxed{...} 提取结果与 ground_truth 匹配（精确 + SymPy 等价） │
  └────────────────┴───────┴───────────────────────────────────────────────────────────────┘                                                                                                                           
                                                                                                                                                                                                                       
  二、格式合规（总分 ≤ 0.1，每个 0.025）                                                                                                                                                                               
                                                                                                                                                                                                                       
  ┌─────────────────┬──────────────────────────────┐                                                                                                                                                                   
  │    原子信号     │             说明             │              
  ├─────────────────┼──────────────────────────────┤                                                                                                                                                                   
  │ has_think_open  │ 是否有 <think>               │              
  ├─────────────────┼──────────────────────────────┤                                                                                                                                                                   
  │ has_think_close │ 是否有 </think>              │                                                                                                                                                                   
  ├─────────────────┼──────────────────────────────┤                                                                                                                                                                   
  │ think_order_ok  │ <think> 是否在 </think> 之前 │                                                                                                                                                                   
  ├─────────────────┼──────────────────────────────┤                                                                                                                                                                   
  │ has_boxed       │ 是否有 \boxed{...}           │              
  └─────────────────┴──────────────────────────────┘                                                                                                                                                                   
                                                                  
  三、推理质量（总分 ≤ 0.1）                                                                                                                                                                                           
                                                                  
  ┌─────────────────────────────────────────────────────────────┬────────┐                                                                                                                                             
  │                            条件                             │  分值  │
  ├─────────────────────────────────────────────────────────────┼────────┤                                                                                                                                             
  │ think_non_empty 且 reasoning_ratio > 0.1                    │ +0.025 │
  ├─────────────────────────────────────────────────────────────┼────────┤                                                                                                                                             
  │ reasoning_ratio > 0.3（推理占回复 30% 以上）                │ +0.025 │                                                                                                                                             
  ├─────────────────────────────────────────────────────────────┼────────┤                                                                                                                                             
  │ has_self_verification（检测到自检关键词如 "let me verify"） │ +0.05  │                                                                                                                                             
  └─────────────────────────────────────────────────────────────┴────────┘                                                                                                                                             
                                                                  
  对应的原子信号：                                                                                                                                                                                                     
                                                                  
  ┌───────────────────────┬─────────────────────────────┐                                                                                                                                                              
  │         信号          │            说明             │         
  ├───────────────────────┼─────────────────────────────┤                                                                                                                                                              
  │ think_non_empty       │ think 内容 > 10 字符        │         
  ├───────────────────────┼─────────────────────────────┤                                                                                                                                                              
  │ reasoning_ratio       │ think 文本长度 / 总回复长度 │                                                                                                                                                              
  ├───────────────────────┼─────────────────────────────┤                                                                                                                                                              
  │ has_self_verification │ 是否包含 self-check 语句    │                                                                                                                                                              
  └───────────────────────┴─────────────────────────────┘                                                                                                                                                              
                                                                  
  四、效率（-0.1 ~ +0.02）                                                                                                                                                                                             
                                                                  
  ┌─────────────────────────────────┬───────┐                                                                                                                                                                          
  │              条件               │ 分值  │                     
  ├─────────────────────────────────┼───────┤                                                                                                                                                                          
  │ overlong_ratio > 2.0            │ -0.1  │                     
  ├─────────────────────────────────┼───────┤                                                                                                                                                                          
  │ overlong_ratio > 1.5            │ -0.05 │                                                                                                                                                                          
  ├─────────────────────────────────┼───────┤                                                                                                                                                                          
  │ overlong_ratio ≤ 1.0 且答案正确 │ +0.02 │                                                                                                                                                                          
  └─────────────────────────────────┴───────┘                                                                                                                                                                          
                                                                  
  overlong_ratio = response_length / target_len，target_len 根据 difficulty 分档（800 / 1500 / 2500）。                                                                                                                
                                                                  
  五、语言一致性（0 或 -0.1）                                                                                                                                                                                          
                                                                  
  ┌────────────────────┬──────┐                                                                                                                                                                                        
  │        条件        │ 分值 │                                   
  ├────────────────────┼──────┤                                                                                                                                                                                        
  │ 中文字符占比 > 20% │ -0.1 │                                   
  ├────────────────────┼──────┤                                                                                                                                                                                        
  │ 否则               │ 0    │                                                                                                                                                                                        
  └────────────────────┴──────┘                             
```

---

## 一、verl 架构概览

### 1.1 整体架构

verl (Volcano Engine RL) 是一个基于 Ray 的分布式 RLHF 训练框架。训练时启动多个 Ray Actor：

```
┌─────────────────────────────────────────────────────────────┐
│                      Ray Cluster                            │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Driver   │  │  Actor   │  │ Rollout  │  │    Ref      │ │
│  │ (main)    │  │ Worker   │  │  (vLLM/  │  │  (policy)   │ │
│  │           │  │ (FSDP)   │  │  SGLang) │  │             │ │
│  └─────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬───────┘ │
│        │              │              │              │         │
│        │      ┌───────┴───────┐      │              │         │
│        └──────┤ Reward Manager├──────┘              │         │
│               └───────────────┘                     │         │
└─────────────────────────────────────────────────────────────┘
```

- **Driver (main_ppo.py)**: 协调训练循环，调度 rollout → reward → advantage → train
- **Actor Worker**: 被训练的模型，用 FSDP/Megatron 做分布式训练
- **Rollout**: 推理引擎 (vLLM 或 SGLang)，高效生成 answers
- **Ref Policy**: 冻结的参考模型，用于计算 KL 散度约束
- **Reward Manager**: 计算每条 response 的 reward

### 1.2 GRPO 训练流程

GRPO (Group Relative Policy Optimization) 的核心思想是**去掉 critic 模型**，用同一 prompt 的多条 responses 的 group 内相对比较来估计 advantage。每一步训练循环如下：

```
第 1 步: Rollout (生成)
  每道题用 Rollout 引擎生成 n 条 response（如 n=16）
  → 得到一个 batch，每个 prompt 有 n 条 responses

第 2 步: Reward (打分)
  用 reward function 给每条 response 打分
  → 每道题得到 n 个 scalar reward {r_1, r_2, ..., r_n}

第 3 步: Advantage Estimation (GRPO 核心)
  对于同一道题的 n 个 reward：
    advantage_i = (r_i - mean(r_1..r_n)) / std(r_1..r_n)
  → advantage > 0: 这条回答比平均好，鼓励
  → advantage < 0: 这条回答比平均差，抑制

第 4 步: Policy Update
  用 PPO-clip 或 GSPO 等 loss 更新 actor 参数
  同时加 KL 散度惩罚，防止偏离 ref policy 太远

第 5 步: 重复
```

### 1.3 GRPO vs DAPO vs GSPO 的区别

| 方法 | Advantage | Policy Loss | 关键特点 |
|------|-----------|-------------|---------|
| **GRPO** | `(r - mean)/std` group 内标准化 | PPO-clip (token-level) | 原始版本，最常用 |
| **DAPO** | 同 GRPO | PPO-clip + overlong penalty | 额外的超长惩罚，reward manager 层处理 |
| **GSPO** | 同 GRPO | **序列级**重要性采样 + clip | `clip_ratio` 极小 (3e-4)，loss 在 sequence-level 聚合 |

**关键区别 — GSPO 的 loss 公式**:

```
标准 PPO:  loss = -min(A * ratio, A * clip(ratio, 1-ε, 1+ε))
GSPO:      loss = -min(A * s_i, A * clip(s_i, 1-ε_low, 1+ε_high))

其中 s_i = sg[s(θ)] * π_θ(y_i,t|x) / sg[π_θ(y_i,t|x)]
     s(θ) = (π_θ(y|x)/π_old(y|x))^(1/|y|)   ← 序列级重要性比

关键参数: clip_ratio_low=3e-4, clip_ratio_high=4e-4 (极小!)
```

**GSPO 的意义**: 序列级 clip 比 token 级 clip 更稳定，适合长文本推理。极小 clip 范围让更新非常保守，训练更平滑。

**DAPO 是在 reward manager 层面做文章**: 多了 overlong penalty（对超过 max_len 的回答额外扣分），reward 计算更精细。

---

## 二、数据格式

### 2.1 verl 要求的 Parquet 格式

verl 读取的数据必须是 **parquet** 格式，每行包含以下**必需字段**：

```python
{
    "data_source": "deepmath-103k",           # 数据集名称 (str)
    "prompt": [                                # OpenAI 格式的 chat messages
        {"role": "system", "content": "You are a helpful math assistant..."},
        {"role": "user", "content": "Evaluate the limit: ..."},
    ],
    "ability": "math",                         # 能力标签 (str)
    "reward_model": {                          # 奖励配置
        "style": "rule",                       # "rule" 表示规则型 reward
        "ground_truth": "0",                   # 标准答案
    },
    "extra_info": {                            # 可选额外信息
        "difficulty": 4.5,
        "topic": "Mathematics -> Precalculus -> Limits",
        "split": "train",
        "index": 0,
    },
}
```

**关键点**:
- `prompt` 字段是 **chat message 列表**，不是纯文本。训练时 verl 会自动调用 tokenizer 的 `apply_chat_template`
- `reward_model.ground_truth` 是 reward function 会收到的标准答案
- `extra_info` 可以存任意额外信息（会传给 reward function）

### 2.2 将我们的 DeepMath-103K 数据转为 verl 格式

这是数据预处理脚本（保存为 `preprocess_deepmath_for_verl.py`）：

```python
#!/usr/bin/env python3
"""
将 DeepMath-103K filtered 数据转换为 verl 要求的 parquet 格式。

Usage:
    python preprocess_deepmath_for_verl.py \
        --data_dir /path/to/filtered \
        --output_dir ~/data/deepmath \
        --train_val_split 0.95
"""

import os
import argparse
import pandas as pd
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a helpful math reasoning assistant. "
    "Think step by step and output the final answer within \\boxed{}."
)


def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path to filtered/*.parquet directory")
    parser.add_argument("--output_dir", default="~/data/deepmath",
                        help="Output directory for train.parquet / test.parquet")
    parser.add_argument("--train_val_split", type=float, default=0.95)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (for quick experiments)")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load all filtered parquet files ──
    parquet_files = sorted(
        f for f in os.listdir(args.data_dir) if f.endswith(".parquet")
    )
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(os.path.join(args.data_dir, f))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} samples from {len(parquet_files)} files")

    if args.max_samples:
        df_all = df_all.sample(n=args.max_samples, random_state=42)
        print(f"Sampled {len(df_all)} samples")

    # ── 2. Build HF Dataset ──
    records = []
    for i, row in df_all.iterrows():
        records.append({
            "data_source": "deepmath-103k",
            "prompt": build_prompt(row["question"]),
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(row["final_answer"]).strip(),
            },
            "extra_info": {
                "difficulty": float(row["difficulty"]),
                "topic": str(row.get("topic", "")),
                "split": "train",
                "index": i,
                "question": row["question"],  # 保留原始问题，方便调试
            },
        })

    dataset = Dataset.from_list(records)

    # ── 3. Split ──
    split_dataset = dataset.train_test_split(
        test_size=1 - args.train_val_split, seed=42
    )
    train_ds = split_dataset["train"]
    test_ds = split_dataset["test"]

    # ── 4. Save ──
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
    print(f"Train: {len(train_ds)} → {train_path}")
    print(f"Test:  {len(test_ds)} → {test_path}")

    # Save example for reference
    import json
    with open(os.path.join(output_dir, "train_example.json"), "w") as f:
        json.dump(train_ds[0], f, indent=2, ensure_ascii=False)
    print(f"Example saved to {output_dir}/train_example.json")


if __name__ == "__main__":
    main()
```

**运行**:
```bash
# 全量数据
python preprocess_deepmath_for_verl.py \
    --data_dir /data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/DeepMath-103K/filtered \
    --output_dir ~/data/deepmath

# 快速实验（1000条）
python preprocess_deepmath_for_verl.py \
    --data_dir /data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/DeepMath-103K/filtered \
    --output_dir ~/data/deepmath_1k \
    --max_samples 1000
```

---

## 三、奖励函数 (Reward Function)

### 3.1 数学推理的 Rule-based Reward

verl 已内置 MATH 的 reward 函数 `math_reward.py`，提取 `\boxed{...}` 与 ground truth 比较。对于 DeepMath-103K (也是 `\boxed{}` 格式)，可以直接复用。

如果我们想自定义 reward（比如加上格式奖励），写一个 `deepmath_reward.py`：

```python
# deepmath_reward.py — 自定义 reward function
"""
DeepMath-103K 的 reward function:
  - 正确答案 (extracted from \\boxed{}) = ground_truth → score = 1.0
  - 答案格式正确 (有 \\boxed{}) 但内容错误 → score = 0.1
  - 格式错误 (没有 \\boxed{}) → score = 0.0

Usage in verl config:
    custom_reward_function.path=deepmath_reward.py
    custom_reward_function.name=compute_score
"""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    verl 会调用此函数，传入:
        data_source: str   — "deepmath-103k"
        solution_str: str  — 模型生成的完整文本
        ground_truth: str  — 标准答案（来自 parquet 的 reward_model.ground_truth）
        extra_info: dict   — 额外信息（来自 parquet 的 extra_info）
    返回: float — reward score
    """
    # ── 1. 提取 \\boxed{...} 中的答案 ──
    boxed_match = _extract_boxed(solution_str)
    if boxed_match is None:
        return 0.0  # 格式错误：没有 \\boxed{}

    answer = _normalize(boxed_match)
    gt = _normalize(str(ground_truth))

    # ── 2. 判断正确性 ──
    if answer == gt:
        return 1.0

    # ── 3. SymPy 等价判断（可选，处理 1/2 vs 0.5 等） ──
    if _sympy_equiv(answer, gt):
        return 1.0

    # ── 4. 格式正确但答案错误 ──
    return 0.1


def _extract_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...} 的内容"""
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def _normalize(s: str) -> str:
    """标准化答案字符串"""
    s = s.strip()
    # 去 LaTeX 包装
    for env in ['$$', '$', '\\(', '\\)', '\\[', '\\]']:
        if s.startswith(env):
            s = s[len(env):]
        if s.endswith(env):
            s = s[:-len(env)]
    s = s.replace('\\displaystyle', '')
    return ' '.join(s.split()).strip()


def _sympy_equiv(a: str, b: str) -> bool:
    """用 sympy 判断两个数学表达式是否等价"""
    try:
        import sympy
        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        return False
```

### 3.2 在 verl 中使用自定义 Reward

```bash
# 在训练脚本中添加:
EXTRA=(
    # ... other params ...
    custom_reward_function.path=deepmath_reward.py
    custom_reward_function.name=compute_score
)
```

或者如果直接复用一个 compute_score 函数，不需要指定 name（默认用 `compute_score` 函数名）。

### 3.3 Reward 设计的注意事项

1. **返回 float**: 可以返回 0/1 二值，也可以返回连续值（如 0.0 ~ 1.0 的分数）。GRPO 在 group 内做标准化，所以分数尺度不重要，**相对顺序**才是关键。

2. **避免全 0 或全 1 的 group**: 如果同一题的所有 n 条回答 reward 都一样（全 0 或全 1），GRPO 的 advantage = 0，这个 group 就没有信号。解法：
   - 增加 n（如 n=16 或 n=32）降低全同概率
   - 增加 reward 粒度（如加入格式分 0.1）

3. **性能**: reward function 在训练循环中频繁调用。避免网络 I/O 或重型计算。如果必须用 reward model，verl 支持 `reward_model.style="model"` 模式，启动单独的 reward model worker。

---

## 四、GRPO 训练脚本

### 4.1 基于 verl 的 DeepMath-103K GRPO 训练脚本

```bash
#!/usr/bin/env bash
# run_deepmath_grpo.sh — GRPO 训练 Qwen3-0.6B on DeepMath-103K
#
# 用法:
#   bash run_deepmath_grpo.sh
#   MODEL_PATH=./output/deepmath-sft/final bash run_deepmath_grpo.sh  # 从 SFT checkpoint 开始
#
set -xeuo pipefail

########################### 可调参数 ###########################
MODEL_PATH=${MODEL_PATH:-/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/deepmath/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/deepmath/test.parquet}
NGPUS=${NGPUS:-8}

# 数据
TRAIN_BATCH_SIZE=64          # 全局 batch（多少个不同的 prompt）
MAX_PROMPT_LENGTH=512         # prompt 最大 token 数（我们数据问题较短）
MAX_RESPONSE_LENGTH=2048      # response 最大 token 数
ROLLOUT_N=${ROLLOUT_N:-8}     # 每个 prompt 生成 n 个 response

# 算法
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
ACTOR_LR=2e-6
KL_LOSS_COEF=0.001
ENTROPY_COEFF=0

# 训练
TOTAL_EPOCHS=5
SAVE_FREQ=20
TEST_FREQ=5

# 硬件
ROLLOUT_TP=1                  # vLLM tensor parallelism（0.6B 模型用 1 即可）
ROLLOUT_GPU_MEM_UTIL=0.5      # vLLM GPU 内存利用率

PROJECT_NAME=deepmath-grpo
EXPERIMENT_NAME=qwen3_06b_grpo_$(date +%Y%m%d_%H%M)
########################### 参数数组 ###########################

DATA=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${TEST_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_RESPONSE_LENGTH * ROLLOUT_N))
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_RESPONSE_LENGTH * ROLLOUT_N))
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_RESPONSE_LENGTH * ROLLOUT_N))
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# 指定自定义 reward function（用 math_reward 内置的，兼容 \\boxed{} 格式）
# 如果你的 reward function 在单独文件，改这里
REWARD=(
    custom_reward_function.path=/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/src/sft/deepmath_reward.py
    custom_reward_function.name=compute_score
)

EXTRA=(
    actor_rollout_ref.rollout.enforce_eager=True     # vLLM eager mode（避免 compile 问题）
    actor_rollout_ref.rollout.free_cache_engine=True  # 释放 vLLM cache
)

########################### 启动 ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@"
```

### 4.2 GSPO 训练脚本 (关键差异)

GSPO 与 GRPO 的差异主要在 actor 配置上：

```bash
ACTOR=(
    # ── GSPO 特有参数 ──
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo        # ← 使用 GSPO loss
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean # ← sequence-level 聚合
    actor_rollout_ref.actor.clip_ratio_low=3e-4               # ← 极小的 clip
    actor_rollout_ref.actor.clip_ratio_high=4e-4
    actor_rollout_ref.actor.clip_ratio_c=10.0                 # ← 上下界分离因子
    actor_rollout_ref.actor.use_kl_loss=False                 # ← GSPO 通常不用 KL loss
    # ── 通用参数（同 GRPO） ──
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # ...
)
```

完整脚本参考: `src/verl-grpo/examples/gspo_trainer/run_qwen3_8b_fsdp.sh`

### 4.3 DAPO 配置 (关键差异)

DAPO 的核心在 **reward manager** 层面：

```bash
# 使用 DAPO reward manager（比默认的 naive reward manager 多了 overlong penalty）
REWARD=(
    reward.reward_manager.name=dapo
    reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
    reward.reward_kwargs.overlong_buffer_cfg.enable=True
    reward.reward_kwargs.overlong_buffer_cfg.len=256        # 超过 max_len-256 开始扣分
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    reward.reward_kwargs.overlong_buffer_cfg.log=True       # 记录 overlong 日志
    custom_reward_function.path=deepmath_reward.py
    custom_reward_function.name=compute_score
)
```

---

## 五、关键注意事项

### 5.1 必须用 SFT Warm-up

**RLVR 直接从 base model 开始几乎不可能成功**。模型需要先通过 SFT 学会：
- 使用 `<think>` 格式
- 在 `\boxed{}` 中输出答案
- 基本的数学推理能力

**推荐流程**:
```
Step 1: SFT (如 main.py) → 得到 checkpoint
Step 2: 用 SFT checkpoint 作为 RL 的初始模型
  MODEL_PATH=/path/to/sft/checkpoint bash run_deepmath_grpo.sh
```

### 5.2 Rollout n 的选择

| n | 每组样本数 | group 内方差 | GRPO 信号质量 | 计算成本 |
|---|----------|-------------|-------------|---------|
| 4 | 少 | 小 | 差（容易全同） | 低 |
| 8 | 中 | 中 | 一般 | 中 |
| 16 | 多 | 大 | 好 | 高 |
| 32 | 很多 | 很大 | 很好 | 很高 |

**建议**: 从 n=8 开始（与我们 eval 的 k=4 不同，这里 n 越大 training signal 越好）。

### 5.3 Max Response Length 设置

我们的数据 token count 均值 1607，最大 2047。所以：
- `max_prompt_length=512`（问题较短，prompt 一般 < 300 tokens）
- `max_response_length=2048`（略大于数据中的平均 token 数，给 RL 探索空间）

**注意**: DAPO 的 overlong buffer 会让模型学会简洁回答。如果 prompt+response 总长度经常超过 `max_prompt_length + max_response_length - overlong_buffer_len`，模型会因扣分而逐渐缩短回答。

### 5.4 GPU 显存估算

Qwen3-0.6B + 8 GPU + FSDP + ZeRO-3 full_shard：

| 组件 | 显存占用 (per GPU) |
|------|-------------------|
| Actor (FSDP shard) | ~0.15 GB |
| Optimizer states | ~0.30 GB (Adam) |
| Activations (grad ckpt) | ~2-5 GB |
| vLLM rollout | ~3-5 GB |
| Ref model | ~0.15 GB |
| **总计** | ~6-12 GB |

对于 24GB/32GB GPU，配置合理。对于 80GB GPU 则非常宽松，可以增大 batch 或 n。

### 5.5 常见错误

**Q1: 训练时 loss 不下降，reward 不提升**

可能原因：
- 没有 SFT warm-up（模型不会输出 `\boxed{}` 格式）
- 学习率太大（GRPO 通常用 `1e-6 ~ 2e-6`，比 SFT 小一个数量级）
- KL coeff 太小，模型偏离太远导致崩溃

**Q2: "All rewards in group are same" Warning**

某个 group 的所有 n 条回答得到了相同的 reward（全 0 或全 1），GRPO advantage = 0。少量出现正常，大量出现则：
- 增大 n（增加多样性）
- 增加 reward 粒度（如格式 +0.1）
- 增大 temperature

**Q3: vLLM OOM**

减小 `rollout_gpu_mem_util` 或增大 `tensor_model_parallel_size`。

**Q4: 从 SFT checkpoint 开始训练，模型很快退化**

SFT checkpoint 的 `tokenizer_config` 可能没有正确的 `chat_template`。用 `tokenizer.apply_chat_template()` 验证一下 prompt 格式是否正确。

### 5.6 训练监控

verl 默认输出到 wandb。关键曲线：

| 曲线 | 含义 | 期望趋势 |
|------|------|---------|
| `actor/pg_loss` | policy gradient loss | 逐渐收敛（不太重要） |
| `actor/ppo_kl` | 与 ref policy 的 KL 散度 | 缓慢上升，控制在 < 0.1 |
| `reward/mean` | 平均 reward | 持续上升 |
| `reward/std` | reward 的标准差 | 保持 > 0 (有区分度) |
| `perf/mfu` | 模型 FLOPs 利用率 | 30-50% (FSDP 通信开销) |

---

## 六、文件清单

按照本指南，你需要创建以下文件：

```
src/sft/
├── deepmath_reward.py             # 自定义 reward function
├── preprocess_deepmath_for_verl.py # 数据预处理脚本
├── run_deepmath_grpo.sh           # GRPO 训练脚本
├── run_deepmath_gspo.sh           # GSPO 训练脚本
└── run_deepmath_dapo.sh           # DAPO 训练脚本
```

**完整流程**:

```bash
# 1. 预处理数据
python preprocess_deepmath_for_verl.py \
    --data_dir /data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/DeepMath-103K/filtered \
    --output_dir ~/data/deepmath

# 2. SFT（如果还没有训过）
cd src/sft
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py

# 3. GRPO（从 SFT checkpoint 开始）
MODEL_PATH=./output/deepmath-sft/final bash run_deepmath_grpo.sh
```
