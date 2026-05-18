# DeepSpeed ZeRO-3 配置参数详解

> 对应文件: `src/sft/ds_zero3.json`

---

## 一、ZeRO 概览

DeepSpeed ZeRO (Zero Redundancy Optimizer) 有三个阶段：

| 阶段 | 分片内容 | 每 GPU 显存节省 | 通信开销 |
|------|---------|---------------|---------|
| ZeRO-1 | Optimizer states | 4× (8 GPU 下) | 低 |
| ZeRO-2 | Optimizer states + Gradients | 8× | 中 |
| ZeRO-3 | Optimizer states + Gradients + Parameters | N× (N=GPU 数) | 高 |

**ZeRO-3 的核心思想**: 模型参数本身也被分片存储在所有 GPU 上。每个 GPU 只持有 1/N 的参数，前向/反向传播时按需 all-gather 所需的参数片，用完立即释放。这使得可以训练比单 GPU 显存大 N 倍的模型。

---

## 二、参数逐条详解

### 2.1 基础配置

```json
"train_batch_size": "auto"
```

**含义**: 全局总 batch size（所有 GPU 合计）。设为 `"auto"` 时，DeepSpeed 会自动计算：
```
train_batch_size = per_device_batch_size × gradient_accumulation_steps × world_size
```
这些值由 HF TrainingArguments 提供，所以设为 `"auto"` 是最方便的做法。

**手动设置**: 如果不用 HF Trainer，需要显式写一个整数，如 `"train_batch_size": 256`。

---

```json
"train_micro_batch_size_per_gpu": "auto"
```

**含义**: 每个 GPU 上每次 forward 的 batch size。设为 `"auto"` 时自动从 HF TrainingArguments 的 `per_device_train_batch_size` 读取。

**注意**: 这与 `train_batch_size` 的关系是：
```
train_batch_size = train_micro_batch_size_per_gpu × gradient_accumulation_steps × world_size
```

---

```json
"gradient_accumulation_steps": "auto"
```

**含义**: 梯度累积步数。设为 `"auto"` 时自动从 HF TrainingArguments 读取。

**原理**: 如果显存放不下大 batch，可以通过多次 forward 累积梯度，再一次性更新参数。等价于用更大的 batch 训练。

---

### 2.2 ZeRO-3 优化器配置

这是整个配置的核心。

```json
"zero_optimization": {
    "stage": 3
}
```

**含义**: 启用 ZeRO Stage 3。意味着 optimizer states、gradients 和 model parameters 三者都会被分片。

---

```json
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
}
```

**含义**: 将 optimizer states 卸载到 CPU 内存。

- **`device: "cpu"`**: optimizer states（如 Adam 的 momentum 和 variance）存储在 CPU 上，不在 GPU 显存中
- **`pin_memory: true`**: 使用 CUDA pinned memory 加速 CPU↔GPU 数据传输。pinned memory 允许 GPU DMA 直接访问 CPU 内存，避免中间拷贝

**显存收益**: Adam optimizer states 通常是参数量的 2 倍（fp32 momentum + fp32 variance）。对于 0.6B 模型，参数量 ~0.6B × 2 bytes (bf16) = 1.2GB，optimizer states = 0.6B × 4 bytes × 2 = 4.8GB (fp32)。offload 后这 4.8GB 全部省下。

**代价**: 每次 optimizer step 需要从 CPU 读取数据，增加少量延迟（通常 < 10%）。

---

```json
"offload_param": {
    "device": "cpu",
    "pin_memory": true
}
```

**含义**: 将模型参数本身也卸载到 CPU 内存。

- **`device: "cpu"`**: 在不需要计算时，模型参数存储在 CPU 上
- **只在使用时才搬到 GPU**: forward/backward 需要某层参数时，从 CPU 搬到 GPU，用完后立即释放

**显存收益**: 极大。GPU 上几乎不常驻参数，只在计算瞬间占用。

**代价**: 每次 layer 的计算都需要 CPU→GPU 的数据搬运，增加通信开销。对于带宽较低的 PCIe 系统（如 PCIe 3.0 ×16 ≈ 16GB/s），可能成为瓶颈。对于 NVLink/NVSwitch 系统（如 A100/H100），影响较小。

**适用场景**: GPU 显存极度紧张时使用。优先 offload optimizer，只在仍不够时 offload param。

---

```json
"overlap_comm": true
```

**含义**: 将通信（all-gather、reduce-scatter）与计算重叠执行。

**原理**: 在 backward 过程中，当计算某一层的梯度时，可以同时进行下一层参数的 all-gather。这样通信时间被计算时间 "隐藏" 起来，不增加总时间。

**开启条件**: 需要 GPU 和网络硬件支持异步通信（基本上所有现代 GPU 都支持）。

**效果**: 典型情况下可将通信开销从训练的 20-30% 降低到 5-10%。

---

```json
"contiguous_gradients": true
```

**含义**: 将分散在内存中的梯度复制到一段连续内存中再发送。

**原理**: 网络的 all-reduce 操作在连续内存上效率更高。开启后 DeepSpeed 会在通信前把各个参数的梯度拷贝到一个连续的 buffer 中。

**代价**: 增加一次内存拷贝（通常很小，可以忽略）。

**建议**: 始终开启。关闭只在极少数会导致额外内存碎片问题时。

---

```json
"sub_group_size": 1e9
```

**含义**: 控制 ZeRO-3 参数分组的最大大小（以参数个数计）。

**原理**: ZeRO-3 将参数分成多个 subgroup，每个 subgroup 内的参数一起被 all-gather 和释放。较小值 = 更细粒度的分片，通信更频繁但更小。较大值 = 更粗粒度，单次通信更大但次数少。

**设为 `1e9`（10 亿）**: 对于大多数模型，这意味着所有参数在一个 subgroup 中（因为参数总数 << 1e9）。这是推荐的默认值，让 DeepSpeed 自动决定最优策略。

**调优**: 对于超大模型（>100B），可能需要调小，因为一次性 all-gather 所有参数会 OOM。但对于 0.6B-70B 的模型，默认值即可。

---

```json
"reduce_bucket_size": "auto"
```

**含义**: 梯度 reduce 操作的 bucket 大小（以字节计）。

**原理**: all-reduce 操作将梯度分成多个 bucket 分别通信。小 bucket = 延迟低但吞吐差，大 bucket = 吞吐高但延迟高。

**设为 `"auto"`**: DeepSpeed 会根据模型大小和网络拓扑自动选择最优值（通常几百 MB）。

---

```json
"stage3_prefetch_bucket_size": "auto"
```

**含义**: ZeRO-3 参数预取的 bucket 大小。

**原理**: ZeRO-3 在前向传播中会提前预取（prefetch）下一层需要的参数。这个参数控制每次预取多少数据。

**设为 `"auto"`**: DeepSpeed 自动选择。通常不需要手动调。

**手动设置场景**: 当使用 CPU offload 时，较大的 prefetch bucket 可以利用 PCIe 带宽更好，但会增加 GPU 显存峰值。

---

```json
"stage3_param_persistence_threshold": "auto"
```

**含义**: 小于此大小的参数张量将被 "持久化" 保留在 GPU 上，不参与分片。

**原理**: 对于非常小的参数张量（如 LayerNorm 的 bias，通常只有几千个元素），分片的通信开销可能大于计算收益。这些 "小" 参数不如直接在每个 GPU 上保留完整副本。

**设为 `"auto"`**: DeepSpeed 自动根据模型情况选择阈值（通常几万到几十万元素）。

**调优**: 如果显存充足，可以调大此值让更多小参数常驻 GPU，减少通信；如果显存紧张，调小或设为 0。

---

```json
"stage3_max_live_parameters": 1e9
```

**含义**: GPU 上同时 "活" 着的参数数量上限。

**原理**: ZeRO-3 的 prefetch 机制会提前将参数搬到 GPU。如果一次 prefetch 太多，可能导致 OOM。这个参数限制了同时存在于 GPU 上的最大参数数量。

**设为 `1e9`**: 几乎等于不限制。对于 0.6B 模型没问题。

**调优**: 如果训练时 OOM，可以尝试减小此值（如 `1e8` = 1 亿），限制一次最多 preload 的参数数量。

---

```json
"stage3_max_reuse_distance": 1e9
```

**含义**: 控制参数在 GPU 上缓存的距离（以参数个数计）。

**原理**: 如果一个参数在不久之后还要再次使用，ZeRO-3 会将其保留在 GPU 缓存中，而不是释放再重新加载。这个参数决定了 "不久" 的定义。

**设为 `1e9`**: 几乎所有参数都不会被释放（类似不限制）。对于训练阶段这通常是安全的。

**调优**: 如果显存紧张，可以减小（如 `1e6`），让 DeepSpeed 更积极地释放参数。

---

```json
"stage3_gather_16bit_weights_on_model_save": true
```

**含义**: 保存 checkpoint 时，将各 GPU 上的参数片段 gather 到一起，并以 bf16/fp16 格式保存（而非 fp32）。

**原理**: 训练时 optimizer 通常以 fp32 精度存储参数。但保存模型用于推理只需要 16bit。设为 `true` 后：
1. 自动 gather 所有分片参数
2. 转为 16bit 精度
3. 保存为完整的模型权重文件

**建议**: 始终设为 `true`，否则保存的 checkpoint 是分片的 fp32 权重，既占空间又不易加载。

---

### 2.3 混合精度

```json
"bf16": {
    "enabled": true
}
```

**含义**: 启用 BF16 (Brain Floating Point 16) 混合精度训练。

**原理**: 
- 前向和反向传播使用 bf16（速度快、省显存）
- 权重更新（optimizer step）使用 fp32（精度高）
- bf16 相比 fp16 的优势：更大的动态范围（与 fp32 相同的 8 位指数），不会梯度溢出

**硬件要求**: A100、H100、RTX 4090、L40S 等支持 bf16 的 GPU。V100 和 RTX 3090 不支持 bf16，需要改用 `"fp16"`。

---

### 2.4 梯度裁剪

```json
"gradient_clipping": 1.0
```

**含义**: 梯度裁剪的阈值（L2 范数）。

**原理**: 当梯度的 L2 范数超过 `1.0` 时，按比例缩放到 1.0：
```
if ||g|| > 1.0:
    g = g * 1.0 / ||g||
```

**作用**: 防止梯度爆炸导致训练不稳定或 loss NaN。

**调优**: 
- 1.0 是最常用的默认值
- 如果 loss 曲线波动大，可以减小（如 0.5）
- 如果训练过于缓慢，可以增大（如 2.0）
- 通过 loss 曲线的平滑度来判断是否需要调整

---

### 2.5 调试配置

```json
"wall_clock_breakdown": false
```

**含义**: 是否输出详细的耗时分解。

**开启 (`true`) 后的输出示例**:
```
[wall_clock_breakdown]
  forward:        120.5 ms (35%)
  backward:       180.2 ms (53%)
  all-gather:      25.3 ms  (7%)
  reduce-scatter:  12.1 ms  (3%)
  optimizer_step:   5.0 ms  (1%)
  total:          343.1 ms
```

**建议**: 调试性能瓶颈时设为 `true`，正式训练时设为 `false`（减少日志量）。

---

## 三、推荐配置策略

### 场景 1: 显存充足（如 8×A100 80GB 跑 7B 模型）

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "none" },
        "offload_param": { "device": "none" },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

去掉 CPU offload，全程 GPU 运行，速度最快。

### 场景 2: 显存紧张（如 8×24GB RTX 3090 跑 7B 模型）

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "cpu", "pin_memory": true },
        "offload_param": { "device": "cpu", "pin_memory": true },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 1e8,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

CPU offload 全部开启，限制 live parameters 防止 OOM。

### 场景 3: 我们的配置（8×V100 32GB 跑 Qwen3-0.6B）

当前 `ds_zero3.json` 的配置：
- Optimizer CPU offload：开（保守起见）
- Param CPU offload：关（0.6B 模型不需要，显存足够容纳参数）
- overlap_comm：开
- gather 16bit on save：开

对于 0.6B 模型这个配置比较保守，实际可以去掉所有 CPU offload 获得更快速度。

---

## 四、常见问题

### Q: 什么时候用 ZeRO-3 vs ZeRO-2 vs FSDP？

| 场景 | 推荐 |
|------|------|
| 模型 < 1B，GPU 显存充足 | ZeRO-2 或 DDP |
| 模型 1B-7B | ZeRO-3 (无 offload) |
| 模型 7B-70B | ZeRO-3 + optimizer offload |
| 模型 > 70B | ZeRO-3 + optimizer + param offload |
| 使用 PyTorch 原生方案 | FSDP full_shard（等价于 ZeRO-3） |

### Q: overlap_comm 什么时候不生效？

如果模型太小（< 100M 参数），单层计算时间极短（< 1ms），通信来不及与计算重叠。此时 overlap_comm 可能反而增加开销。

### Q: 为什么训练开始后 GPU 利用率很低？

可能原因：
1. CPU offload 导致 GPU 等待 CPU→GPU 数据传输（检查 PCIe 带宽）
2. `num_workers` 太少，数据加载成为瓶颈
3. 模型太小导致 GPU 计算太快，通信占比高

排查方法：开启 `wall_clock_breakdown: true` 查看各阶段耗时。
