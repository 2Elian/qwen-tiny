# 自定义评估指标、自定义 Loss、多 Loss 输出

> 对应代码：`src/sft/main.py`, `src/sft/common/`

本文档基于我们的 DeepMath-103K SFT 训练代码，演示如何在 HF Trainer 框架下实现自定义评估、自定义 loss 和多 loss 监控。

---

## 一、自定义评估指标

### 1.1 原理

HF Trainer 的 `compute_metrics` 参数接收一个函数，在每个 eval step 结束后调用：

```
trainer.evaluate() 被触发 (每个 eval_steps)
  → 对 eval dataset 做完整前向推理
  → 收集所有 predictions 和 labels
  → 调用 compute_metrics(pred)
  → 返回 dict[str, float]，自动记录到 logs
```

默认情况下，Trainer 只输出 `eval_loss`。`compute_metrics` 让你可以额外计算任意指标。

### 1.2 对数学推理数据计算自定义指标

数学推理任务特别适合计算以下指标：

| 指标 | 含义 | 实现难度 |
|------|------|---------|
| Exact Match (EM) | 模型输出的最终答案与标准答案是否完全一致 | 低 |
| Answer Accuracy | 从生成文本中提取答案，判断是否正确 | 中 |
| Token-level Accuracy | 预测 token 与标签 token 的匹配率 | 低 |
| Perplexity | 基于 loss 的困惑度 | 低 (直接算) |
| Format Compliance | 是否正确使用了 `<think>` 标签 | 低 |

### 1.3 实现代码

**步骤 1**: 在 `src/sft/common/` 下新增 `metrics.py`：

```python
# src/sft/common/metrics.py
"""Custom evaluation metrics for math reasoning SFT."""

import re
import numpy as np
from typing import Dict


def extract_final_answer(text: str) -> str:
    """Extract the content inside \\boxed{...} from generated text."""
    # Match \boxed{...} - supports nested braces
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last boxed answer
    return ""


def compute_math_metrics(tokenizer):
    """Factory: returns a compute_metrics function for math reasoning."""

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        eval_pred has:
          - predictions: shape (batch, seq_len, vocab_size) logits, OR
                         shape (batch, seq_len) token ids if prediction_loss_only=False
          - label_ids: shape (batch, seq_len), with -100 for masked positions

        For generation-based metrics we need to actually generate, not
        just compute from logits. This is the "light" version based on loss.
        """
        predictions, labels = eval_pred

        # ── Token-level accuracy ──
        # predictions are logits (batch, seq_len, vocab_size) → argmax
        if predictions.ndim == 3:
            pred_tokens = predictions.argmax(axis=-1)
        else:
            pred_tokens = predictions  # already token ids

        # Only count positions where labels != -100
        mask = labels != -100
        if mask.sum() > 0:
            token_acc = (pred_tokens[mask] == labels[mask]).mean()
        else:
            token_acc = 0.0

        # ── Perplexity (derived from eval_loss logged by Trainer) ──
        # Trainer already computes eval_loss. We can't compute perplexity here
        # because we don't have access to the loss. See section 2.3 for how
        # to add it.

        metrics = {
            "token_accuracy": float(token_acc),
        }

        return metrics

    return compute_metrics


def compute_generation_metrics(tokenizer, dataset, max_new_tokens=512):
    """
    Factory: returns a compute_metrics that does actual generation
    to extract answers and compare with ground truth.

    NOTE: This is slower because it generates text at each eval step.
    For large eval sets, consider doing this only at the end of each epoch
    via a custom Callback (see section 3.3).
    """
    gt_answers = [sample["final_answer"] for sample in dataset]

    def compute_metrics(eval_pred) -> Dict[str, float]:
        # This function doesn't get the generated text directly — it gets
        # logits. For generation-based eval, use a TrainerCallback instead.
        # See section 3.3 for the proper approach.
        return {}

    return compute_metrics
```

**步骤 2**: 在 `main.py` 中接入：

```python
# main.py 的 main() 函数中，创建 Trainer 前添加:

from common.metrics import compute_math_metrics

# 使用基于 token accuracy 的轻量指标
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[TrainingVisualCallback()],
    compute_metrics=compute_math_metrics(tokenizer),  # ← 新增
)
```

### 1.4 基于生成的评估（在 Epoch 结束时执行完整推理）

对于数学推理，真正的评估应该让模型**生成完整文本**，然后用规则提取 `\boxed{...}` 中的答案与标准答案 `final_answer` 比较。

这需要通过 `TrainerCallback` 而非 `compute_metrics` 实现，因为 `compute_metrics` 只收到 logits 而非生成的文本。

```python
# src/sft/common/generation_eval.py
"""Generation-based math evaluation callback."""

import re
import torch
from transformers import TrainerCallback


class MathGenerationEvalCallback(TrainerCallback):
    """
    At the end of each epoch, generate answers for a subset of the eval set
    and compute exact match accuracy vs the ground truth final_answer.
    """

    def __init__(self, tokenizer, val_dataset, max_samples=50, max_new_tokens=512):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model") or control.model
        if model is None:
            return

        model.eval()
        correct = 0
        total = 0

        # Take a subset for speed
        indices = range(min(self.max_samples, len(self.val_dataset)))
        for i in indices:
            sample = self.val_dataset[i]
            input_ids = torch.tensor([sample["input_ids"]]).cuda()
            attention_mask = torch.tensor([sample["attention_mask"]]).cuda()

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated part (after the prompt)
            prompt_len = input_ids.shape[1]
            generated = self.tokenizer.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            )

            # Extract answer from \boxed{...}
            pred_answer = self._extract_boxed(generated)
            gt_answer = sample.get("final_answer", "")

            if pred_answer and pred_answer.strip() == gt_answer.strip():
                correct += 1
            total += 1

        model.train()
        accuracy = correct / max(total, 1)

        # Log to Trainer's tracking (will appear in logs and plots)
        if state.log_history:
            state.log_history[-1]["eval_answer_accuracy"] = accuracy

        print(f"\n[Eval] Generation-based accuracy: {accuracy:.4f} ({correct}/{total})")
        return control

    @staticmethod
    def _extract_boxed(text: str) -> str:
        pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
        matches = re.findall(pattern, text)
        return matches[-1].strip() if matches else ""


# ── Usage in main.py ──
# from common.generation_eval import MathGenerationEvalCallback
#
# gen_eval_cb = MathGenerationEvalCallback(
#     tokenizer=tokenizer,
#     val_dataset=val_dataset,
#     max_samples=50,
# )
# trainer = Trainer(..., callbacks=[TrainingVisualCallback(), gen_eval_cb])
```

---

## 二、自定义 Loss 计算

### 2.1 三种修改 Loss 的方式

| 方式 | 侵入性 | 灵活性 | 适用场景 |
|------|--------|--------|---------|
| ① 修改 labels 掩码 | 低 | 中 | 只调整哪些 token 参与 loss |
| ② 子类化 Trainer | 中 | 高 | 需要完全自定义 loss 逻辑 |
| ③ 修改模型 forward | 高 | 最高 | 需要修改模型内部计算 |

### 2.2 方式 ①：通过 labels 掩码控制 Loss

这是最简单的方式。HF CausalLM 的 loss 计算逻辑是：

```python
# transformers 内部等价代码
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

**`labels[i] == -100` 的位置不参与 loss 计算**。

我们已经在 `common/data.py` 中使用了这一点：prompt 部分（system + user）都设为 -100，只有 assistant 部分参与 loss。

**扩展**: 如果你只想让 `\boxed{...}` 答案部分参与 loss：

```python
# 在 _preprocess_batch 中, tokenize 之后
# 找到 \boxed{ 的位置, 将其之前的内容全部设为 -100
def mask_before_boxed(input_ids, labels, tokenizer):
    """Only compute loss on the answer part (after and including \boxed)."""
    boxed_start = tokenizer.encode("\\boxed{", add_special_tokens=False)
    # Find the first position where boxed pattern starts
    for i in range(len(input_ids) - len(boxed_start)):
        if input_ids[i:i+len(boxed_start)] == boxed_start:
            labels[:i] = [-100] * i  # Mask everything before \boxed
            return labels
    return labels  # No boxed found, keep original
```

### 2.3 方式 ②：子类化 Trainer 自定义 Loss

当需要完全自定义 loss 逻辑时（如加权 loss、多任务 loss），继承 `Trainer` 并重写 `compute_loss`：

```python
# src/sft/common/custom_trainer.py
"""Custom Trainer with flexible loss computation."""

import torch
import torch.nn.functional as F
from transformers import Trainer


class MultiLossTrainer(Trainer):
    """
    Custom Trainer that supports:
    - Multiple loss components with individual logging
    - Weighted loss combination
    - Custom loss functions

    Override compute_loss() to implement your custom logic.
    """

    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional: weights for different loss components
        self.loss_weights = loss_weights or {"lm": 1.0}
        # Storage for last computed loss components (for logging)
        self._last_loss_components = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override the default loss computation.

        This is called at every training step.
        Returns: loss tensor (scalar) or (loss, outputs) if return_outputs=True
        """
        # ── 1. Standard language modeling loss ──
        labels = inputs.get("labels")
        outputs = model(**inputs)
        lm_loss = outputs.loss  # Standard cross-entropy from the model

        # ── 2. Additional loss components (examples) ──

        # 2a. Contrastive loss: push apart different solutions of same question
        # contrastive_loss = self._contrastive_loss(outputs.logits, labels)

        # 2b. Length penalty: penalize overly long generations
        # length_loss = self._length_penalty(outputs.logits, labels)

        # 2c. Answer accuracy reward (for RL-like training)
        # answer_loss = self._answer_consistency_loss(outputs.logits, labels)

        # ── 3. Combine losses ──
        total_loss = lm_loss  # Start with LM loss

        # Add extra losses with weights
        # total_loss = total_loss + 0.1 * contrastive_loss
        # total_loss = total_loss + 0.01 * length_loss

        # Store for logging
        self._last_loss_components = {
            "lm_loss": lm_loss.detach().item(),
            # "contrastive_loss": contrastive_loss.detach().item(),
            # "length_loss": length_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, start_time=None):
        """
        Intercept logging to inject custom loss components.
        Called by Trainer._maybe_log_save_evaluate().
        """
        # Inject loss components into logs before they are recorded
        if self._last_loss_components:
            for key, value in self._last_loss_components.items():
                logs[key] = value

        super().log(logs, start_time)

    # ── Private helpers for custom losses ──

    def _contrastive_loss(self, logits, labels):
        """Example: contrastive loss between different solutions."""
        # Placeholder - implement based on your needs
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    def _length_penalty(self, logits, labels):
        """Example: penalize long sequences to encourage conciseness."""
        if labels is None:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        mask = (labels != -100).float()
        lengths = mask.sum(dim=1).float()
        # Penalize sequences longer than a target length
        target_len = 500
        penalty = F.relu(lengths - target_len).mean() * 1e-4
        return penalty

    def _answer_consistency_loss(self, logits, labels):
        """
        Example: ensure the final answer token is consistent with
        the generated reasoning.
        """
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


# ── Usage in main.py ──
# from common.custom_trainer import MultiLossTrainer
#
# trainer = MultiLossTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=data_collator,
#     callbacks=[TrainingVisualCallback()],
#     loss_weights={"lm": 1.0, "contrastive": 0.1},
# )
```

### 2.4 方式 ③：修改模型 Forward（高级）

如果需要在模型内部计算额外损失（如知识蒸馏、隐藏层正则化等），在模型加载后 monkey-patch 或包装 forward：

```python
class LossWrappedModel(torch.nn.Module):
    """Wraps a CausalLM to add auxiliary losses."""

    def __init__(self, base_model, aux_loss_weight=0.1):
        super().__init__()
        self.base_model = base_model
        self.aux_loss_weight = aux_loss_weight

    def forward(self, **inputs):
        outputs = self.base_model(**inputs, output_hidden_states=True)

        # Standard LM loss
        lm_loss = outputs.loss

        # Auxiliary loss: hidden state regularization
        # Penalize large hidden state norms to encourage smooth representations
        hidden_states = outputs.hidden_states
        if hidden_states:
            last_hidden = hidden_states[-1]
            aux_loss = last_hidden.norm(dim=-1).mean() * 1e-6
        else:
            aux_loss = 0.0

        outputs.loss = lm_loss + self.aux_loss_weight * aux_loss
        return outputs

    # Delegate other methods to base_model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
```

---

## 三、多 Loss 在训练中实时输出

### 3.1 默认行为

HF Trainer 每个 `logging_steps` 输出 `train_loss`，每个 `eval_steps` 输出 `eval_loss`。所有依赖仓库（如 `report_to="wandb"` 或 `"tensorboard"`）都会自动记录。

```
{'loss': 1.234, 'learning_rate': 1.8e-05, 'epoch': 0.5, ...}
```

### 3.2 方式 ①：通过 MultiLossTrainer 注入

使用第二节写的 `MultiLossTrainer`，loss 分量会自动注入到 log 中：

```python
# 训练时的日志输出会自动包含:
# {
#     'loss': 1.234,      # total_loss (Trainer 默认显示的)
#     'lm_loss': 1.100,   # 语言模型 loss
#     'total_loss': 1.234,
#     'learning_rate': 1.8e-05,
#     'epoch': 0.5,
# }
```

**在 `TrainingVisualCallback` 中捕获**：

```python
# callbacks.py 的 on_log 方法中:
def on_log(self, args, state, control, logs=None, **kwargs):
    if logs is None:
        return
    step = state.global_step
    if "lm_loss" in logs:
        self.lm_losses.append((step, logs["lm_loss"]))
    if "total_loss" in logs:
        self.train_losses.append((step, logs["total_loss"]))
```

### 3.3 方式 ②：通过 Callback 独立计算和注入指标

```python
class MultiMetricCallback(TrainerCallback):
    """
    Computes and injects multiple custom metrics at each eval step.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after each evaluation. We can inject additional metrics.
        `metrics` is the dict that will be logged.
        """
        if metrics is None:
            return

        # Compute perplexity from eval_loss
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["eval_perplexity"] = float(torch.exp(torch.tensor(eval_loss)))

        # The Trainer stores predictions in self for the last eval batch
        # if compute_metrics is not set. Otherwise compute_metrics already
        # provides custom metrics.

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Inject perplexity at every log step (training).
        """
        if logs is None:
            return
        train_loss = logs.get("loss")
        if train_loss is not None:
            logs["train_perplexity"] = float(torch.exp(torch.tensor(train_loss)))
```

### 3.4 方式 ③：所有指标一览

将以上方案整合后，训练时的 log 输出可以包含以下全部指标：

```
训练步日志 (每 logging_steps):
  loss (total_loss)
  lm_loss                    ← MultiLossTrainer 注入
  #contrastive_loss          ← MultiLossTrainer 注入（可选）
  train_perplexity           ← MultiMetricCallback 注入
  learning_rate
  grad_norm
  epoch

评估步日志 (每 eval_steps):
  eval_loss
  eval_perplexity            ← MultiMetricCallback 注入
  eval_token_accuracy        ← compute_metrics 注入
  eval_answer_accuracy       ← MathGenerationEvalCallback 注入 (epoch 结束时)
  eval_runtime
  epoch
```

---

## 四、完整接入示例

更新后的 `main.py` 关键部分：

```python
# ── 导入自定义模块 ──
from common import (
    load_config,
    load_and_prepare_data,
    DeepMathDataCollator,
    TrainingVisualCallback,
)
from common.metrics import compute_math_metrics
from common.generation_eval import MathGenerationEvalCallback
from common.custom_trainer import MultiLossTrainer
from common.multi_metric_callback import MultiMetricCallback

# ── 创建回调 ──
viz_callback = TrainingVisualCallback()
gen_eval_cb = MathGenerationEvalCallback(
    tokenizer=tokenizer,
    val_dataset=val_dataset,
    max_samples=50,
)
multi_metric_cb = MultiMetricCallback(tokenizer=tokenizer)

# ── 创建 Trainer ──
trainer = MultiLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[viz_callback, gen_eval_cb, multi_metric_cb],
    compute_metrics=compute_math_metrics(tokenizer),
    loss_weights={"lm": 1.0},
)

trainer.train()
```

---

## 五、常见问题

### Q1: `compute_metrics` 中的 predictions 为什么是 logits 而不是 token IDs？

如果 `TrainingArguments` 中没设置 `prediction_loss_only=True`（默认 False），Trainer 会把整个 eval set 的 logits 全部收集起来传给 `compute_metrics`。对于大模型和大数据集，这可能导致 OOM。

**解决**: 将不需要 logits 的指标放在 Callback 中计算；或者设置 `metric_for_best_model="eval_loss"` 并利用 Trainer 内置的 loss 指标。

### Q2: 如何只对 `\boxed{...}` 中的答案计算 loss？

参考 2.2 节的方式 ①，在 tokenize 时将 `\boxed{` 之前的所有 label 设为 -100。

### Q3: 自定义 loss 后，`eval_loss` 还是原来的 LM loss 吗？

不是。`eval_loss` 使用的是 `trainer.compute_loss()` 返回的值。如果你在 `compute_loss` 中修改了 loss 组合，eval 时也会用同样的组合。如果希望 eval 只用原始 LM loss，需要子类化 `evaluate()` 方法。

### Q4: 如何同时输出到 TensorBoard？

将 `report_to` 设为 `"tensorboard"`（或 `"all"` 包括 wandb）：

```yaml
training:
  report_to: "tensorboard"
```

```bash
tensorboard --logdir ./output/deepmath-sft/tensorboard
```

Trainer 内部通过 `accelerator.log()` 将所有 `compute_metrics` 返回的 dict 自动推送到对应的 reporting backend。
