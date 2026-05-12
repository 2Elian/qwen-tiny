# Continue pre-training using qwen3-base and attention residual method.

## 1. 你为什么需要学习这个项目（以便继续进行 attnRes 的预训练）？

我相信使用持续预训练 addRess 练将提高您对 LLM 架构的理解，并使您能够学习如何使用 transformers 训练模型、改进transformers的模型架构代码。

## 2. attenRes详解

一切的根源：深度神经网络：

$$h_l = f_{l-1}(h_{l-1})$$

这是一个标准的深度神经网络，第l层的hidden state. f是一个非线性激活函数(可以是relu、可以是sigmoid等)

那么它有什么问题呢？kaiming为什么要提出残差结构呢？

假设现在我们要求Loss对h1的梯度：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_2}\frac{\partial h_2}{\partial h_1} = \dots = \frac{\partial L}{\partial h_l}\frac{\partial h_l}{\partial h_{l-1}}\frac{\partial h_{l-1}}{\partial h_{l-2}}\dots\frac{\partial h_2}{\partial h_{1}}$$

而$\frac{\partial h_l}{\partial h_{l-1}} = \frac{\partial f_{l-1}}{\partial h_{l-1}}$

所以L对h1的梯度可以表示为：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_l}\frac{\partial f_{l-1}}{\partial h_{l-1}}\frac{\partial f_{l-2}}{\partial h_{l-2}}\dots\frac{\partial f_1}{\partial h_{1}}$$

从上述公式可以看出来什么问题嘛？

thinking...

---

好啦，上面公式最大的问题是：如果某一层的梯度值非常小（比如<1），那么随着网络深度增加，这个梯度相乘就会越来越小，从而导致最后传递到h1的梯度几乎为0，也就是我们常常说的：梯度消失问题。

所以从这个角度来看，如何能避免梯度消失呢？从上述梯度公式的角度来看，似乎把每一个$\frac{\partial f_{l-1}}{\partial h_{l-1}}$加上一点点的值变成：$(\frac{\partial f_{l-1}}{\partial h_{l-1}} + ?)$就能避免梯度连乘导致的小值向后传播的问题。


我们看看kaiming的残差是怎么做的：

$$h_l = h_{l-1} + f_{l-1}(h_{l-1})$$

此时，对 $h_{l-1}$ 求导：

$$\frac{\partial h_l}{\partial h_{l-1}} = \frac{\partial (h_{l-1} + f_{l-1}(h_{l-1}))}{\partial h_{l-1}} = I + \frac{\partial f_{l-1}}{\partial h_{l-1}}$$

其中 **$I$ 是单位矩阵（Identity Matrix）**。这就是为什么大家管残差连接叫做**恒等映射（Identity Mapping）**——因为即使 $f_{l-1}$ 什么都没学到（梯度为 0），信息也能通过 $I$ 原封不动地传到下一层。

---

### 2.1 残差如何解决梯度消失：从链式法则看

现在把残差的梯度代回 Loss 对 $h_1$ 的链式求导：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_l} \left(I + \frac{\partial f_{l-1}}{\partial h_{l-1}}\right) \left(I + \frac{\partial f_{l-2}}{\partial h_{l-2}}\right) \dots \left(I + \frac{\partial f_1}{\partial h_1}\right)$$

展开这个连乘，每一项都是 $(I + \frac{\partial f}{\partial h})$，而不是原来的 $\frac{\partial f}{\partial h}$。关键区别：

| | Plain 网络 | Residual 网络 |
|---|---|---|
| 每层雅可比 | $\frac{\partial f}{\partial h}$ | $I + \frac{\partial f}{\partial h}$ |
| 梯度链 | 纯连乘 → 指数衰减 | 恒等路径 + 变换路径 |
| 最坏情况 | $\prod \lambda_i \approx 0$ | 至少有 $I$ 项保底 |

**直观理解：**

- **Plain 网络**：梯度像传话游戏，每传一层就打一次折扣。如果每层梯度范数 < 1（这对 Sigmoid/Tanh 几乎必然发生），传 100 层后梯度就是 $0.9^{100} \approx 0$。
- **Residual 网络**：梯度有**两条路径**——一条是 $I$ 组成的"高速公路"（梯度直接穿过），另一条是经过各层变换的"普通公路"。展开后：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_l} \cdot \left[ I + \sum_i \frac{\partial f_i}{\partial h_i} + \sum_{i<j} \frac{\partial f_i}{\partial h_i}\frac{\partial f_j}{\partial h_j} + \dots \right]$$

第一项 $I$ 保证**梯度至少能原样传到浅层**，后面各项是各层变换的贡献。即使深层变换的梯度很小，$I$ 也不会消失。

---

### 2.2 实验验证：不同激活函数下的梯度流动

我们用 10/30/50/100 层 MLP 做了对照实验（代码见 `attenRes/deep_residual_experiment.py`），在初始化后测量每层输入梯度范数：

**Plain 网络（无残差）—— 梯度消失随深度加剧：**

| 深度 | ReLU | GELU | Tanh | SiLU |
|------|------|------|------|------|
| 10 | ratio=14.3 | ratio=13.7 | ratio=11.3 | ratio=30.3 |
| 30 | ratio=11.8 | ratio=12.7 | ratio=4.2 | **ratio=1954** |
| 50 | ratio=11.6 | ratio=79.2 | ratio=1.2 | **ratio=1.6×10⁷** |
| 100 | ratio=6.9 | **ratio=1.1×10⁹** | ratio=0.1 | **ratio=4.1×10¹⁰** |

> ratio = 最后一层梯度 / 第一层梯度。ratio < 0.01 即严重梯度消失。

**关键发现：**
- **SiLU 最严重**：100 层时前向值衰减到 $10^{-12}$，浅层几乎收不到梯度信号
- **Tanh 最鲁棒**：S 型函数梯度饱和是已知问题，但 Tanh 在无残差时表现最好
- **ReLU 的"假健康"**：梯度比看起来还行（6~14），但这是因为部分神经元死亡后梯度集中在少数活神经元上，不代表有效学习（训练损失 13 vs 残差网络 6）

**Residual 网络（有残差）—— 梯度流动健康：**

| 深度 | ReLU | GELU | Tanh | SiLU |
|------|------|------|------|------|
| 10 | ratio=5.8 | ratio=6.4 | ratio=5.9 | ratio=7.0 |
| 30 | ratio=1.6 | ratio=5.8 | ratio=5.1 | ratio=7.3 |
| 50 | ratio=0.4 | ratio=4.9 | ratio=4.6 | ratio=7.3 |
| 100 | ratio=0.04 | ratio=2.3 | ratio=4.2 | ratio=7.2 |

**关键发现：**
- 所有激活函数的梯度比稳定在健康范围，不会出现指数级衰减
- **Tanh + Residual 是黄金组合**：100 层时 ratio=4.2，训练损失从 ~14 降到 ~6
- **ReLU 在极深网络（100 层）中 ratio=0.04**：虽比 Plain 好很多，但 ReLU 的非负输出导致前向均值逐层漂移，需要配合 LayerNorm/BatchNorm 使用
- **SiLU 的救赎**：Plain 网络 ratio 高达 $10^{10}$，加残差后稳定在 ~7

---

### 2.3 残差学习的本质

```
Plain 网络:   h_l = f(h_{l-1})           → 梯度 ∏ ∂f/∂h   → 指数衰减
Residual 网络: h_l = h_{l-1} + f(h_{l-1}) → 梯度 ∏ (I+∂f/∂h) → 始终有 I 保底
```

残差的本质不是在"改进"网络，而是在**改变梯度传播的数学结构**——把连乘变成了加性展开，让深层网络从"几乎不可能训练"变成了"几乎和浅层一样容易训练"。这也就是何恺明在 ResNet 论文中说的：**"We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping."**

---

### 2.4 从残差连接到 Attention Residuals：动机与做法

把残差递推式展开：

$$h_l = h_{l-1} + f_{l-1}(h_{l-1}) = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$$

可以看到**每一层都接收到所有先前层输出的等权和**。残差定义了信息如何沿着深度聚合，这种深度维度的聚合仍然由**固定的单位权重**支配，即没有任何机制可以选择性地强调或抑制个别层的贡献。

#### 2.4.1 标准残差的三个局限

**1. 无法选择性访问**

第l层只能访问 $h_{l-1}$ 这一个单一的压缩状态。不同层类型（例如 self-attention vs. MLP）接收到完全相同的聚合状态，尽管它们可能受益于不同的层输出加权。打个比方：attention 层可能想多看几眼前面某个 MLP 的输出，但残差给它的只是一个"大锅饭"。

**2. 信息不可逆丢失**

展开递推 $h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$，所有先前层的贡献被**等权相加**。一旦信息被聚合进了 $h_{l-1}$，深层就无法从聚合体中单独恢复出某一层的原始输出。这在实践中有直接后果：论文中提到：实验表明，相当比例的层可以被剪枝而损失很小，说明深层并没有充分利用浅层的信息。

**3. 隐藏状态膨胀**

在现代 LLM 中，PreNorm（先归一化再做变换）已成为主导范式。但在 PreNorm + 残差的组合下，隐藏状态的模长随深度以 **$O(L)$ 增长**，逐渐稀释每一层的相对贡献。浅层信息被"淹没"在不断膨胀的累积和中，无法被深层选择性地检索（这句话的意思是说，如果中间的第k层的前向值非常大，那么第l层接收的大部分信息都来自于第k层，其他层的信息都被稀释掉了）。

这三者共同指向一个需求：**让每一层能够选择性地、以数据依赖的方式从所有先前层中聚合信息。**

#### 2.4.2 AttnRes 怎么做

将标准残差的等权求和：

$$h_l = \sum_{i=0}^{l-1} v_i \quad \text{（其中 } v_0 = h_1,\; v_i = f_i(h_i) \text{）}$$

替换为**可学习的加权求和**：

$$h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i, \quad \sum_i \alpha_{i \to l} = 1$$

其中 $\alpha_{i \to l}$ 是 softmax 注意力权重，由每层一个**可学习的伪查询向量** $w_l \in \mathbb{R}^d$ 计算：

$$\alpha_{i \to l} = \frac{\exp\left(w_l^\top \cdot \text{RMSNorm}(v_i)\right)}{\sum_{j=0}^{l-1} \exp\left(w_l^\top \cdot \text{RMSNorm}(v_j)\right)}$$

每一层只需要**一个额外的 d 维向量** $w_l$ 作为"查询"，去注意所有先前层的输出 $v_i$。RMSNorm 防止大模长的层输出主导注意力权重。

**关键：** $w_l$ 是参数而非输入依赖的——它代表"作为一个 attention 层（或 MLP 层），我应该多关注哪些前面的层"。这个设计选择意味着查询与当前层的计算解耦，使得注意力权重可以提前计算或批量处理。

**标准残差是 AttnRes 的特例：** 当所有 $\alpha_{i \to l}$ 退化为均匀分布时，AttnRes 退化为标准残差。更一般地说，标准残差执行的是**深度维度的线性注意力**（linear attention over depth），而 AttnRes 将其推广为**深度维度的 softmax 注意力**.

#### 2.4.3 Full AttnRes 与 Block AttnRes

**Full AttnRes** 让每一层直接注意所有先前层的输出：

```
第 l 层的输入 = softmax 注意力(查询=w_l, 键/值=[h1, f1(h1), f2(h2), ..., f_{l-1}(h_{l-1})])
```

优势：最细粒度的信息访问。开销：$O(L^2 d)$ 计算，$O(Ld)$ 显存（但在标准训练中这些激活值本来就要为反向传播保留，所以**没有额外显存开销**）。

**问题：** 在大规模训练中，激活重计算（activation recomputation）和流水线并行（pipeline parallelism）是标配。Full AttnRes 要求显式保存并跨流水线阶段传输所有 $L$ 个层的输出，通信开销为 $O(Ld)$。

**Block AttnRes** 将 $L$ 层划分为 $N$ 个 block，每个 block 包含 $S = L/N$ 层：

- **Block 内部：** 使用标准残差，$S$ 层的输出被累加为一个单一的 block 表示 $b_n = \sum_{j \in B_n} f_j(h_j)$
- **Block 之间：** 对 $N$ 个 block 级别表示 + 当前 block 内的部分和（partial block）应用 full attention

```
第 l 层的输入 = softmax_注意力(查询=w_l, 键/值=[b0, b1, ..., b_{n-1}, partial_block])
```

优势：显存和通信从 $O(Ld)$ 降至 $O(Nd)$。$N \approx 8$ 在实践中几乎恢复了 Full AttnRes 的全部收益。

**AttnRes 的完整算法伪代码（对应 `modeling_attnres.py`）：**

```python
def block_attn_res(blocks, partial_block, proj, norm, recency_bias):
    """
    blocks:      已完成 block 的表示 [b0, b1, ..., b_{n-1}]  每项 [B, T, D]
    partial_block: 当前 block 内部分累加和 (b_n^i)            [B, T, D]
    proj:        伪查询 w_l  (nn.Linear(D, 1, bias=False))
    norm:        RMSNorm
    recency_bias: 标量偏置，加在 partial_block 的 logit 上
    """
    V = torch.stack(blocks + [partial_block])          # [N+1, B, T, D]
    K = norm(V)                                        # 对 keys 做 RMSNorm
    query = proj.weight.view(-1)                       # (D,)
    logits = torch.einsum('d, n b t d -> n b t', query, K)
    logits[-1] = logits[-1] + recency_bias             # 近因偏置
    weights = logits.softmax(dim=0)                    # [N+1, B, T]
    h = torch.einsum('n b t, n b t d -> b t d', weights, V)
    return h
```

**近因偏置（Recency Bias）的设计：** 在 softmax 前给 partial_block（当前 block 的未完成部分）加一个大的正偏置，使初始化时 $\alpha_{\text{partial}} \approx 1$，即 `block_attn_res(...) ≈ partial_block`。这意味着**训练开始时，AttnRes 在数学上等价于标准残差**。随着训练进行，$w_l$ 和偏置共同适配，网络逐渐学会跨 block 的注意力。

**AttnRes vs 标准残差**

```
标准残差:   h_l = h1 + f1 + f2 + ... + f_{l-1}       (等权累加)
AttnRes:    h_l = α0·h1 + α1·f1 + α2·f2 + ...         (可学习加权)
                 └── softmax(w_l^T · RMSNorm(v_i)) ──┘
```


## 3. 如何修改模型

```python
"""
1. 为什么要定义 config_class？
    为了告诉 Hugging Face 加载器，“我是谁”。
    在 transformers 的底层，有一个全局的注册表（Registry）。当你调用 AutoModel.from_pretrained("你的模型路径") 时，HF 会先去读取路径下的 config.json 文件，找到里面的 "model_type" 字段，然后去注册表里找：“哪个类的 config_class 对应这个 model_type？” 找到了对应的 Config，会解析model的配置参数，然后再找对应的 Model，把config的参数注入到model里面。
"""
class Qwen3AttnResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = Qwen3AttnResConfig  # 👈 绑定关系
# 这行代码相当于贴了个标签：“我是一个特殊的 Qwen3 模型，请用 Qwen3AttnResConfig 这个规则来解析我的参数，不要用原生的 Qwen3Config。”
"""
2. PreTrainedConfig 是干嘛的？Qwen3Config 是干嘛的？
    PreTrainedConfig (基类)
        它是 HF 所有模型配置的老祖宗。它的核心职责是序列化和反序列化。
            保存时：把 Python 对象变成 config.json 文件。（序列化）
            加载时：读取 config.json，变成 Python 对象。（反序列化）
            它还处理一些通用逻辑，比如 kwargs 的容错、设备放置等。
class Qwen3Config(PreTrainedConfig):
    model_type = "qwen3"  # 👈 注册到 HF 全局表里的名字 你可以把 PreTrainedConfig 想象成一张空白表格有字段信息但没有value，Qwen3Config 就是把 Qwen3 的默认参数印在了这张表格上。
"""
class Qwen3AttnResConfig(Qwen3Config):
    model_type = "qwen3_attnres" # 1. 必须换名字！否则会和原版 Qwen3 冲突

    def __init__(self, 
                 attnres_num_blocks: int = 8,          # 2. 新参数：把模型分成几个 Block？
                 attnres_recency_bias_init: float = 3.0,# 3. 新参数：近期偏置的初始值
                 attnres_mode: str = "block",           # 4. 新参数：AttnRes 的模式（分组还是全层）
                 attnres_gate_type: str = "bias",       # 5. 新参数：门控类型（偏置还是 sigmoid）
                 **kwargs):                             # 6. 接收原版 Qwen3 的所有参数
"""
Qwen3AttnResModel (骨干模型的 forward)：是**“大脑”**，负责把文字变成高维向量，进行复杂的注意力计算。
Qwen3AttnResForCausalLM (最终模型的 forward)：是**“嘴巴”**，负责把大脑想出来的高维向量，翻译成具体的下一个汉字/单词，并算算自己猜得准不准（Loss）。
一、 两个 forward 的核心区别
维度	Qwen3AttnResModel.forward (大脑)	Qwen3AttnResForCausalLM.forward (嘴巴)
输入	只有基础特征（input_ids, attention_mask 等）	基础特征 + labels（正确答案，用于算损失）
核心动作	1. 词嵌入
2. 跑你的魔改 AttnRes 层
3. 算辅助熵 (entropy_accum)	1. 调用大脑 (self.model(...))
2. 投射到词表 (self.lm_head)
3. 算交叉熵损失
输出对象	BaseModelOutputWithPast
(只包含 last_hidden_state 隐藏向量)	CausalLMOutputWithPast
(包含 logits 词表概率 和 loss 标量损失)
能否直接对话	❌ 不能。吐出来的是矩阵，人看不懂	✅ 能。结合 GenerationMixin 可以直接打字
二、 @merge_with_config_defaults 是什么？
作用：自动填充配置文件里的默认参数。
痛点： 现代大模型的 forward 函数参数多达二三十个。很多参数其实用户不需要传，直接用 config.json 里的默认值就行。以前的做法是在函数开头写一堆 if xxx is None: xxx = self.config.xxx，代码极其臃肿。

它的魔法：

@merge_with_config_defaults
def forward(self, input_ids, attention_mask=None, use_cache=None, ...):
    pass
当你调用 model.forward(input_ids=[1,2,3]) 时（没传 use_cache），这个装饰器会在执行前偷偷去看一眼 self.config.use_cache，如果配置里是 True，它就会自动把 True 塞进 use_cache 参数里。实现了“缺什么，自动从 config 里补什么”。

三、 @capture_outputs 是什么？
作用：调试神器和中间特征提取器。
痛点： 深度学习最难的是解释模型内部到底发生了什么。如果你想提取第 10 层的注意力权重或者隐藏状态，以前你得去改 Qwen3AttnResDecoderLayer 的代码，把中间变量 return 出来，非常侵入式。

它的魔法：
加了这玩意后，你可以用极其优雅的方式在外部抓取数据：


from transformers import CaptureOutput

# 开启抓捕模式，指定要抓第 0 层和第 5 层的输出
with CaptureOutput(module=model.model, module_names=["layers.0", "layers.5"]) as captured:
    model(input_ids=...)

# 循环结束后，直接拿数据
layer_0_output = captured["layers.0"]
这个装饰器在底层劫持了 forward 的返回值，检查当前是否处于 with CaptureOutput 上下文中，如果是，就把中间结果存一份，不影响正常的模型运行。对你研究 AttnRes 机制非常有用！
四、 @auto_docstring 是什么？
作用：解放双手的文档生成器。
痛点： 看你代码里那一大段 """ Run the Qwen3AttnRes decoder stack. Args: input_ids: Token... """，这其实是手动写的。但 HF 框架要求所有模型的文档格式必须高度统一，手动写很容易漏掉类型提示或者格式错乱。

它的魔法：
其实你可以把函数里那一大段手动写的文档全部删掉，只留一个空的 """ """ 或者干脆不写。@auto_docstring 会在 Python 解释器加载这个类的时候，自动读取函数签名（input_ids: torch.LongTensor | None = None）以及你在父类里写的注释，动态拼装成一份完美的、符合 HF 规范的 Docstring。
你看到的那个完美的文档，大概率有一部分（或全部）是这个装饰器在运行时自动注入的。

总结图景
当你执行 model.generate("你好") 时，实际的调用链是这样的：


用户调用 generate()
      │
      ▼ (GenerationMixin 控制循环)
Qwen3AttnResForCausalLM.forward (带 labels=None)
      │
      ├─ @auto_docstring (帮你生成帮助文档)
      ├─ @merge_with_config_defaults (偷偷补齐 use_cache=True 等参数)
      │
      ▼ 调用大脑
Qwen3AttnResModel.forward
      │
      ├─ @capture_outputs (暗中观察，看你是否需要提取中间层)
      │
      ▼ 核心计算
跑 32 层 Qwen3AttnResDecoderLayer (维护 blocks 和 partial_block)
      │
      ▼ 返回大脑结果 (隐藏向量 + 熵)
回到 Qwen3AttnResForCausalLM.forward
      │
      ▼ 过 lm_head 变成词表概率
返回 logits 给 generate() -> 选出下一个字 "！"
"""
```

## 4. 训练参数详解

## 5. 训练过程记录

| Model | Chinese Held-out PPL | C-Eval Acc | CMMLU Acc |
|-------|----------------------|------------|-----------|
| Baseline (Standard Residual) | 41.79 | 0.2314 | 0.2437 |
| Full Attention Residuals | 60.08 | 0.2402 | 0.2437 |
| Block Attention Residuals | 38.95 | 0.2838 | 0.2437 |
