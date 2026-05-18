# PyTorch FSDP 原理 + 参数详解 + 代码使用指南

> 对应代码: `src/sft/main_fsdp.py`

---

## 一、FSDP 原理

### 1.1 什么是 FSDP

FSDP (Fully Sharded Data Parallel) 是 PyTorch 原生提供的分布式训练策略，功能等价于 DeepSpeed ZeRO-3。它由 Meta 的 FairScale 项目演进而来，自 PyTorch 1.12 起合入 `torch.distributed.fsdp`。

**核心思想**: 将模型参数、梯度和优化器状态**分片 (shard)** 到所有 GPU 上，每个 GPU 只持有 1/N 的完整副本。需要计算时临时通过 all-gather 收集完整参数，用完立即释放。

```
┌──────────────────────────────────────────────────┐
│              传统 Data Parallel (DDP)              │
│  GPU0: [完整模型] [完整梯度] [完整优化器]           │
│  GPU1: [完整模型] [完整梯度] [完整优化器]           │
│  GPU2: [完整模型] [完整梯度] [完整优化器]           │
│  GPU3: [完整模型] [完整梯度] [完整优化器]           │
│  → 每张卡存完全相同的副本，显存浪费 N 倍            │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│                  FSDP (Full Shard)                │
│  GPU0: [¼ 参数] [¼ 梯度] [¼ 优化器]               │
│  GPU1: [¼ 参数] [¼ 梯度] [¼ 优化器]               │
│  GPU2: [¼ 参数] [¼ 梯度] [¼ 优化器]               │
│  GPU3: [¼ 参数] [¼ 梯度] [¼ 优化器]               │
│  → 每张卡只存 1/N，需要时通过通信临时拼出完整参数    │
└──────────────────────────────────────────────────┘
```

### 1.2 FSDP 的分片策略 (Sharding Strategy)

PyTorch FSDP 提供 4 种分片策略，由 `ShardingStrategy` 枚举控制：

| 策略 | 分片参数 | 分片梯度 | 分片优化器 | 显存节省 | 通信量 | 对应 ZeRO |
|------|---------|---------|-----------|---------|--------|----------|
| `NO_SHARD` | ✗ | ✗ | ✗ | 0 | 最低 | DDP |
| `SHARD_GRAD_OP` | ✗ | ✓ | ✓ | 中 | 中 | ZeRO-2 |
| `FULL_SHARD` | ✓ | ✓ | ✓ | 最大 | 高 | ZeRO-3 |
| `HYBRID_SHARD` | 节点内✗，跨节点✓ | ✓ | ✓ | 中高 | 中 | ZeRO-3++ |

```
NO_SHARD:
  → 等价于 DDP。每个 GPU 有完整副本，仅梯度 all-reduce。

SHARD_GRAD_OP:
  → 前向: 每 GPU 使用自己的完整参数（无通信）
  → 反向: 梯度计算后分片 + all-reduce，每个 GPU 只保留自己那部分梯度
  → 优化器 step: 每个 GPU 更新自己那部分参数，然后 all-gather 拼回完整参数

FULL_SHARD:
  → 前向: 每层计算前 all-gather 该层参数的完整副本
  → 反向: 每层计算前再次 all-gather 参数，计算梯度后分片 + reduce-scatter
  → 参数在任何时刻都不完整存在于单个 GPU 上

HYBRID_SHARD:
  → 节点内使用 NO_SHARD（NVLink 高速互联，通信免费）
  → 跨节点使用 FULL_SHARD（IB/RoCE 带宽有限，分片节省通信量）
  → 适合多节点训练
```

### 1.3 FSDP 的通信模式

FSDP 对每个 transformer layer 执行以下通信操作：

```
┌──────────────────────────────────────────────────────────┐
│  Forward:                                                │
│  ① All-gather: 收集该层完整参数（各 GPU 贡献自己的分片）     │
│  ② Compute: 用完整参数做前向计算                           │
│  ③ Discard: 释放非本地的参数分片                           │
│                                                          │
│  Backward:                                               │
│  ④ All-gather: 再次收集该层完整参数                          │
│  ⑤ Compute gradients: 用完整参数计算梯度                    │
│  ⑥ Reduce-scatter: 各 GPU 只保留自己负责分片的梯度           │
│  ⑦ Discard: 释放非本地的梯度分片                           │
│                                                          │
│  Optimizer Step:                                         │
│  ⑧ 每个 GPU 独立更新自己那部分参数（无需通信，因为都有梯度）   │
└──────────────────────────────────────────────────────────┘
```

**关键洞察**: 每个 layer 单独进行 all-gather / reduce-scatter，这使得：
- 通信可以与计算**流水线**重叠（下一层 all-gather 时，当前层在计算）
- GPU 显存峰值只取决于**最大的一层**（而非整个模型）

### 1.4 FSDP vs DeepSpeed ZeRO-3 对比

| 维度 | FSDP | DeepSpeed ZeRO-3 |
|------|------|-------------------|
| 维护方 | PyTorch 官方 | Microsoft |
| 集成度 | `torch.distributed.fsdp` 原生 | 需要安装 `deepspeed` 包 |
| HF Trainer 支持 | 原生 `--fsdp full_shard` | 原生 `--deepspeed config.json` |
| CPU Offload | PyTorch ≥ 2.0 支持 | 成熟稳定 |
| 混合精度 | 通过 `torch.cuda.amp` | 内置 `bf16`/`fp16` 字段 |
| 配置复杂度 | 低（几个参数） | 高（几十个参数） |
| 性能 | 与 ZeRO-3 持平 | 与 FSDP 持平 |
| 生态 | PyTorch 生态无缝 | DeepSpeed 特有功能（如 compression） |

**简单结论**: 如果你只需要分片训练且不想引入额外依赖，用 FSDP。如果需要 DeepSpeed 的高级特性（如 communication compression、ZeRO-Infinity 等），用 DeepSpeed。

---

## 二、FSDP 参数详解

在 HF Trainer 中使用 FSDP 需要配置两层参数：
1. `TrainingArguments` 的 `fsdp` 参数（选策略）
2. `TrainingArguments` 的 `fsdp_config` 参数（微调行为）

### 2.1 `fsdp` — 分片策略

```python
TrainingArguments(
    fsdp="full_shard",          # ← 主策略
    fsdp_config={...},          # ← 微调参数
)
```

可选值：

| 值 | 对应 ShardingStrategy | 适用场景 |
|----|----------------------|---------|
| `""` (空字符串) | 不使用 FSDP | 单 GPU 或 DDP |
| `"full_shard"` | FULL_SHARD | **推荐**，单节点多 GPU |
| `"shard_grad_op"` | SHARD_GRAD_OP | 模型勉强能放入单 GPU 时 |
| `"hybrid_shard"` | HYBRID_SHARD | 多节点训练 |
| `"offload"` | HYBRID_SHARD + CPU offload | 显存极端紧张 |
| `"auto_wrap"` | FULL_SHARD + 自动 wrap | 让 HF 自动决定 wrap 策略 |

---

### 2.2 `fsdp_config` 参数逐条详解

#### `fsdp_transformer_layer_cls_to_wrap`

```python
"fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]
```

**含义**: 指定需要对哪些模块类进行 FSDP 包装（wrapping）。

**这是最重要的参数**。FSDP 需要在某一层级将模型 "切分" 成多个 FSDP unit。对 Transformer 模型，最自然的切分点是**每个 decoder/encoder layer**。

**原理**:
```
Model                              FSDP Units (以 Qwen3DecoderLayer 为界)
├── Embedding          → [FSDP Unit 1: Embedding]        (单独一个 FSDP 包装)
├── Layer 0            → [FSDP Unit 2: Qwen3DecoderLayer] (完整的 layer)
├── Layer 1            → [FSDP Unit 3: Qwen3DecoderLayer]
├── ...                → ...
├── Layer N            → [FSDP Unit N+2: Qwen3DecoderLayer]
└── LM Head            → [FSDP Unit N+3: LM Head]          (单独一个 FSDP 包装)
```

每个 FSDP unit 内部独立进行 all-gather / reduce-scatter。这样：
- Embedding 层只在首尾通信一次
- 每个 layer 在需要时才 all-gather 参数
- 不需要的 layer 的参数已被释放，不占显存

**如何找到正确的类名**:

```python
# 方法 1: 打印模型结构
from transformers import AutoModel
model = AutoModel.from_pretrained("qwen3-0.6b")
print(model)
# 找到 decoder layer 的类名，例如: Qwen3DecoderLayer

# 方法 2: 遍历 named_modules
for name, module in model.named_modules():
    print(type(module).__name__)
# 从中找到 Transformer 的 decoder/encoder layer 类

# 方法 3: 查 HuggingFace 文档或源码
# Qwen2/Qwen3 系列: "Qwen2DecoderLayer" / "Qwen3DecoderLayer"
# Llama 系列: "LlamaDecoderLayer"
# Mistral 系列: "MistralDecoderLayer"
# Gemma 系列: "GemmaDecoderLayer"
```

**如果设错或留空的后果**:
- 留空 `[]`: FSDP 会把整个模型当作一个 unit。此时等于没有分层释放参数，显存峰值 = 整个模型大小。**显存节省极少**。
- 设错类名: FSDP 找不到匹配的模块，等同于留空。

---

#### `fsdp_backward_prefetch`

```python
"fsdp_backward_prefetch": "backward_pre"
```

**含义**: 反向传播时的参数预取策略。

**可选值**:

| 值 | 行为 | 效果 |
|----|------|------|
| `"backward_pre"` | 在当前 layer backward 结束前，提前触发下一 layer 的 all-gather | **推荐**，通信与计算重叠最好 |
| `"backward_post"` | 在当前 layer backward 结束后才触发下一 layer 的 all-gather | 无重叠，通信和计算串行 |
| `None` | 不预取 | 与 post 类似，但不使用 CUDA stream |

**原理图解**:
```
backward_pre (推荐):
  Layer 0 backward ──────────────────┐
  Layer 1 all-gather ────────────┐   │
  Layer 1 backward ──────────┐   │   │
  Layer 2 all-gather ────┐   │   │   │
  ...                    │   │   │   │
  time ─────────────────────────────────→
  → 通信被计算 "隐藏"，几乎无额外延迟

backward_post:
  Layer 0 backward ──────┐
  Layer 1 all-gather     ├──┐
  Layer 1 backward       │  ├──┐
  Layer 2 all-gather     │  │  ├── ...
  time ─────────────────────────────────→
  → 通信和计算交替进行，有明显空闲等待
```

---

#### `fsdp_forward_prefetch`

```python
"fsdp_forward_prefetch": False
```

**含义**: 前向传播时是否预取下一层的参数。

**可选值**:
- `False`: 当前层前向计算完成后，才 all-gather 下一层参数
- `True`: 在当前层前向计算期间，提前 all-gather 下一层参数（需要额外的 GPU 显存存两份参数）

**建议**: 大多数情况设为 `False`。前向计算的通信需求远小于反向（反向还需要 reduce-scatter 梯度），前向预取的收益不大，但会额外占用显存。

---

#### `fsdp_use_orig_params`

```python
"fsdp_use_orig_params": False
```

**含义**: 是否保留原始的参数名称和结构。

**`False` (默认)**:
- FSDP 会将参数 flatten（展平）到一个大 tensor 中再分片
- 参数失去原始名称，model.state_dict() 返回扁平化的参数
- **优点**: 更高效，单次 all-gather 代替多次小通信
- **缺点**: 无法直接用于 checkpoint 保存/加载（需要特殊处理）

**`True`**:
- 保留原始参数结构，每个参数独立分片和通信
- **优点**: 兼容常规的 `model.state_dict()` 和 `load_state_dict()`
- **缺点**: 大量小参数导致通信效率降低（all-gather 需要做多次）

**建议**: 使用 HF Trainer 时设为 `False`（Trainer 内部已处理 checkpoint 问题）。手动训练循环时设为 `True` 更方便。

---

#### `fsdp_cpu_ram_efficient_loading`

```python
"fsdp_cpu_ram_efficient_loading": True
```

**含义**: 是否使用 CPU 内存高效的加载方式。

**`True`**:
- 只有 rank 0 加载完整模型到 CPU 内存
- 其他 rank 在 rank 0 加载完成后才从 rank 0 广播自己的分片
- **优点**: 节省 CPU 内存（N 个进程不各自加载完整模型）
- **缺点**: 加载时间稍长（需要广播）

**`False`**:
- 所有 rank 各自从磁盘加载完整模型到 CPU，再各自只保留自己的分片
- **优点**: 加载更快（并行）
- **缺点**: CPU 内存占用高（N 个进程各有完整副本）

**建议**: 模型小于 10B 且 CPU 内存充足时设为 `False`（更快）。大模型或多进程时设为 `True`。

---

#### `fsdp_sync_module_states`

```python
"fsdp_sync_module_states": True
```

**含义**: 是否在 FSDP wrapper 创建后同步所有 rank 的参数状态。

**`True`**:
- FSDP 初始化时，各 rank 的随机初始化参数可能不一致（如果不用 `from_pretrained`）
- 开启后会自动从 rank 0 广播参数给其他 rank，确保所有 rank 起点一致

**`False`**: 仅在确信所有 rank 从同一个 checkpoint 加载了完全相同的参数时使用。

**建议**: 始终设为 `True`，除非你明确知道所有 rank 参数已经一致。

---

#### `fsdp_state_dict_type` (高级)

```python
"fsdp_state_dict_type": "FULL_STATE_DICT"  # 或 "SHARDED_STATE_DICT" 或 "LOCAL_STATE_DICT"
```

**含义**: `state_dict()` 返回的格式。

| 值 | 内容 | 用途 |
|----|------|------|
| `"FULL_STATE_DICT"` | 每个 rank 返回完整模型 | 保存 checkpoint 用 |
| `"SHARDED_STATE_DICT"` | 每个 rank 返回自己那部分 | 分布式 checkpoint |
| `"LOCAL_STATE_DICT"` | 只返回本地分片（不通信） | 调试用 |

HF Trainer 内部已处理，通常不需要手动设置。

---

#### `fsdp_activation_checkpointing` (高级)

```python
"fsdp_activation_checkpointing": True
```

等价于 `model.gradient_checkpointing_enable()`。但 FSDP 需要知道 activation checkpointing 的存在以正确排序通信操作。如果同时使用 FSDP 和 gradient checkpointing，建议在 fsdp_config 中设为 `True`，而非仅调用 model 的方法。

---

## 三、代码使用详解

### 3.1 方式一: HF Trainer + FSDP（推荐，我们的 `main_fsdp.py`）

这是最简单的方式，所有 FSDP 细节由 HF Trainer 内部处理：

```python
from transformers import TrainingArguments, Trainer

# ① 确定 transformer layer 类名
# 对于 Qwen3 系列: "Qwen3DecoderLayer"
# 对于 Llama3 系列: "LlamaDecoderLayer"
# 对于 Qwen2 系列: "Qwen2DecoderLayer"

fsdp_config = {
    "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
    "fsdp_backward_prefetch": "backward_pre",
    "fsdp_forward_prefetch": False,
    "fsdp_use_orig_params": False,
    "fsdp_cpu_ram_efficient_loading": True,
    "fsdp_sync_module_states": True,
}

training_args = TrainingArguments(
    # ... 常规参数 (lr, epochs, batch_size 等) ...
    fsdp="full_shard",          # ← 核心: 选择 FULL_SHARD 策略
    fsdp_config=fsdp_config,    # ← 核心: 微调参数
    bf16=True,                  # ← FSDP 模式下也要指定混合精度
)

trainer = Trainer(
    model=model,                # 不需要手动 .to(device), FSDP 会处理
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# 保存: HF Trainer 内部会自动 gather 分片参数再保存
model.save_pretrained("./output")
tokenizer.save_pretrained("./output")
```

**启动命令**:
```bash
# 单节点 8 GPU
torchrun --nproc_per_node=8 main_fsdp.py

# 多节点 2×8 GPU
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_endpoint=master:29500 main_fsdp.py
```

---

### 3.2 方式二: 原生 PyTorch FSDP（手动训练循环）

当需要完全控制训练过程时使用这种方式：

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# ── ① 初始化分布式环境 ──
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# ── ② 加载模型 ──
model = AutoModelForCausalLM.from_pretrained(
    "qwen3-0.6b", torch_dtype=torch.bfloat16
)

# ── ③ 定义 FSDP auto-wrap policy ──
# 这告诉 FSDP 在 Qwen3DecoderLayer 的边界进行分片
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Qwen3DecoderLayer},
)

# ── ④ 定义混合精度 ──
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,   # 参数存储和计算精度
    reduce_dtype=torch.float32,   # 梯度 reduce 精度（通常保持 fp32）
    buffer_dtype=torch.bfloat16,  # buffer (如 attention mask) 精度
)

# ── ⑤ 包装模型 ──
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ← ZeRO-3 等价
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # ← 通信计算重叠
    mixed_precision=mixed_precision_policy,
    device_id=torch.cuda.current_device(),
    sync_module_states=True,       # ← 同步各 rank 的参数
    use_orig_params=False,         # ← 展平参数以提升效率
    cpu_offload=False,             # ← 设为 True 启用 CPU offload
)

# ── ⑥ 训练循环 ──
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(epochs):
    for batch in train_loader:
        # batch 需要先移到正确的 GPU
        batch = {k: v.cuda(local_rank) for k, v in batch.items()}

        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# ── ⑦ 保存完整 checkpoint ──
# FSDP 需要先 gather 所有分片参数才能保存完整模型
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    full_state_dict = model.state_dict()

if dist.get_rank() == 0:
    torch.save(full_state_dict, "checkpoint.pt")
```

---

### 3.3 如何找到 Transformer Layer 的类名

不同模型的 decoder/encoder layer 类名不同，这是 FSDP 配置中最容易出错的地方：

```python
# ── 通用方法: 在训练脚本中打印 ──
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "your-model-path",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 只在 CPU 上加载
)

# 找 decoder/encoder layer
for name, module in model.named_modules():
    cls_name = type(module).__name__
    if "Layer" in cls_name or "Block" in cls_name or "Decoder" in cls_name:
        print(f"{name}: {cls_name}")
```

常见模型的类名速查：

| 模型系列 | Layer 类名 |
|---------|-----------|
| Qwen2/Qwen2.5 | `Qwen2DecoderLayer` |
| Qwen3 | `Qwen3DecoderLayer` |
| Llama 3/3.1/3.2 | `LlamaDecoderLayer` |
| Mistral | `MistralDecoderLayer` |
| Gemma 2 | `Gemma2DecoderLayer` |
| DeepSeek | `DeepseekDecoderLayer` |
| Phi-3/Phi-4 | `Phi3DecoderLayer` / `Phi4DecoderLayer` |
| GPT-NeoX | `GPTNeoXLayer` |
| BLOOM | `BloomBlock` |

---

### 3.4 FSDP 的 4 种 ShardingStrategy 使用场景

```python
from torch.distributed.fsdp import ShardingStrategy

# 场景 1: 单 GPU 或显存充足 → 不开 FSDP (DDP 即可)
# torchrun --nproc_per_node=8 main.py   (不加 fsdp 参数)

# 场景 2: 模型刚好放得下单 GPU → SHARD_GRAD_OP (只省优化器+梯度)
TrainingArguments(fsdp="shard_grad_op", fsdp_config={...})
# 对应: ShardingStrategy.SHARD_GRAD_OP

# 场景 3: 单节点多 GPU，模型放不下单卡 → FULL_SHARD (省一切) 【最常用】
TrainingArguments(fsdp="full_shard", fsdp_config={...})
# 对应: ShardingStrategy.FULL_SHARD

# 场景 4: 多节点 → HYBRID_SHARD (节点内不切，跨节点切)
TrainingArguments(fsdp="hybrid_shard", fsdp_config={...})
# 需要额外配置: fsdp_config["fsdp_hybrid_shard_num_node_groups"] = ...

# 场景 5: 极端显存紧张 → FULL_SHARD + CPU offload
training_args = TrainingArguments(
    fsdp="full_shard",
    fsdp_config={
        "fsdp_offload_params": True,        # ← 参数卸载到 CPU
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
        ...
    },
)
```

---

### 3.5 FSDP 的 checkpoint 保存与加载

```python
# ── 保存 ──
# 方法 A: HF Trainer 自动处理（推荐）
trainer.train()
model.save_pretrained("./output")   # Trainer 内部已经 gather 了参数

# 方法 B: 手动训练时保存
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    state_dict = model.state_dict()

if dist.get_rank() == 0:
    torch.save(state_dict, "checkpoint.pt")

# ── 加载 ──
# 方法 A: 用 from_pretrained 重新加载完整模型（推荐）
model = AutoModelForCausalLM.from_pretrained("./output")
model = FSDP(model, ...)

# 方法 B: 加载分片 checkpoint（快速恢复训练）
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    model.load_state_dict(torch.load("sharded_checkpoint.pt"))
```

---

### 3.6 常见问题与排查

#### Q1: 训练时 GPU 利用率波动大（0% → 100% → 0%）

**原因**: 通信和计算没有重叠。

**排查**:
```python
# 确认
"fsdp_backward_prefetch": "backward_pre",  # 设为 backward_pre
"fsdp_forward_prefetch": False,           # 前向不预取
```

检查模型是否太小（计算时间太短，通信占比高）。对小于 100M 参数的模型，FSDP 通信开销可能大于收益。

#### Q2: OOM even with FSDP full_shard

**排查顺序**:
1. 确认 `fsdp_transformer_layer_cls_to_wrap` 是否正确设置
2. 减小 `per_device_train_batch_size` 到 1
3. 开启 gradient checkpointing: `model.gradient_checkpointing_enable()`
4. 启用 CPU offload: `fsdp_config["fsdp_offload_params"] = True`
5. 减小 `max_seq_length`

#### Q3: `fsdp_transformer_layer_cls_to_wrap` 设错有什么现象

**现象**: 显存占用与不开 FSDP 几乎一样。

**原因**: FSDP 找不到匹配的模块类，将整个模型作为单个 FSDP unit。等于没做分层分片。

**修复**: 用 3.3 节的方法找到正确的类名。

#### Q4: 保存的 checkpoint 不完整

**现象**: `model.save_pretrained()` 后保存的模型文件很小或加载报错。

**原因**: FSDP 下直接调用 `state_dict()` 只返回当前 rank 的分片。

**修复**: HF Trainer 已处理，手动训练时参考 3.5 节用 `FULL_STATE_DICT`。

#### Q5: FSDP vs DeepSpeed 如何选择

| 条件 | 推荐 |
|------|------|
| 纯 PyTorch 项目，不想装 DeepSpeed | FSDP |
| 需要 DeepSpeed 高级特性 (compression token, monitor 等) | DeepSpeed |
| 使用 HF Trainer | 两者都方便，FSDP 配置更简单 |
| 手动训练循环 | FSDP API 更直观 |
| 需要 TP (Tensor Parallelism) | DeepSpeed (FSDP 不支持原生 TP) |

---

## 四、完整示例：从零开始用 FSDP 训练 Qwen3-0.6B

我们的 `main_fsdp.py` 是完整可运行的示例，总结其关键步骤：

```
1. torchrun 启动 → 自动设置 LOCAL_RANK, WORLD_SIZE 等环境变量
2. AutoTokenizer.from_pretrained() → 加载 tokenizer
3. AutoModelForCausalLM.from_pretrained() → 加载模型（CPU 上，FSDP 会处理分发）
4. model.gradient_checkpointing_enable() → 节省显存
5. load_and_prepare_data() → 数据 tokenize + 切分
6. TrainingArguments(fsdp="full_shard", fsdp_config={...}) → 配置 FSDP
7. Trainer(model, args, ...) → HF Trainer 自动包装 FSDP
8. trainer.train() → 训练
9. model.save_pretrained() → 保存完整模型
```

启动:
```bash
torchrun --nproc_per_node=8 src/sft/main_fsdp.py --config config.yaml
```
