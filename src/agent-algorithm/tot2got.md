# Tree of Thoughts & Graph of Thoughts 深度解析

> 从算法原理到代码实现，每一个细节都讲清楚。

---

## 目录

1. [Chain-of-Thought：一切的起点](#一chain-of-thought-一切的起点)
2. [Tree of Thoughts (ToT)](#二tree-of-thoughts-tot)
   - 2.1 核心思路
   - 2.2 四个组件详解
   - 2.3 BFS 搜索详解
   - 2.4 DFS 搜索详解
   - 2.5 Game of 24 完整流程
   - 2.6 三个任务实验结果
3. [Graph of Thoughts (GoT)](#三graph-of-thoughts-got)
   - 3.1 核心思路
   - 3.2 四大模块架构
   - 3.3 三种思维变换操作
   - 3.4 评分与排名机制
   - 3.5 Volume of Thought 指标
   - 3.6 四种任务实验结果
4. [ToT vs GoT 对比](#四tot-vs-got-对比)
5. [完整代码演示](#五完整代码演示)
6. [与 DeepEye / Heuristic Learning 的关联](#六与-deepeye--heuristic-learning-的关联)

---

## 一、Chain-of-Thought：一切的起点

在讲 ToT 之前，必须先理解 CoT（Chain-of-Thought）。

### 1.1 标准 Prompting（IO）

```
输入: "4, 5, 6, 10 这四个数字，如何通过加减乘除得到 24？"
输出: "(5 * (10 - 4)) - 6 = 24"    ← 直接输出答案
```

LLM 在内部"一次性"完成推理。对于复杂问题，这经常出错——因为模型没有机会检验中间结果。

### 1.2 Chain-of-Thought Prompting

```
输入: "4, 5, 6, 10 用加减乘除得到 24，一步一步思考。"
输出:
  10 - 4 = 6
  6 * 5 = 30
  30 - 6 = 24                    ← 逐步输出中间"想法"(thought)
```

CoT 让 LLM 把思考过程写出来，每一步是一个 thought。这显著提升了数学推理任务的表现。

常见的CoT方式是在提示词中写：请一步一步推理。

### 1.3 CoT 的两个根本缺陷

CoT 的问题在于它的推理过程是一根**链条**：

```
thought1 → thought2 → thought3 → thought4 → ... → 最终答案
```

1. **局部性（Local）**：每一步只能接着前一步往下走，不能尝试"另一条路"
2. **全局性（Global）**：一旦走入歧途，无法回溯（backtrack），只能一条路走到黑

---

## 二、Tree of Thoughts (ToT)

**论文**：[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

**作者**：Shunyu Yao et al. (Princeton University & Google DeepMind)

**发表**：NeurIPS 2023

**代码**：https://github.com/princeton-nlp/tree-of-thought-llm

### 2.1 核心思路

ToT 把推理过程从**链**扩展为**树**：

```
                        level 0 (根: 初始状态)
                         |
            +------------+-----------+------------+
            |            |           |            |
         thought1    thought2    thought3    thought4    ← level 1
            |           /  \         |
        +---+---+     t5    t6    thought7            ← level 2
        |       |
       t8      t9                                       ← level 3
```

每个节点是一个 "thought"（思考单元），从父节点出发可以**生成多个候选子节点**，然后**评估每个候选**，**只保留最有希望的方向**继续探索。这本质上是把经典 AI 的**树搜索算法**（BFS / DFS）引入到 LLM 推理中。

ToT 不改变模型权重，纯靠 prompt 实现。

### 2.2 四个组件详解

#### 组件 1：Thought Decomposition（思维分解）

**问题**：在一个推理任务中，"一个 thought" 是什么？

答案取决于任务——thought 必须满足两个条件：

| 条件 | 含义 |
|------|------|
| 足够**小** | LLM 能生成多样化的候选（整本书太大，一个 token 太小） |
| 足够**大** | LLM 能评估它的好坏（一个 token 太小，无法判断） |

**三个任务中的 thought 定义**：

| 任务 | thought 是什么 | 标准步数 |
|------|---------------|---------|
| **Game of 24** | 一行算式，如 `10 - 4 = 6 (left: 5 6 6)` | 3 步 |
| **Creative Writing** | 一段写作计划段落 | 4-5 段 |
| **Mini Crosswords** | 一行中填的几个单词 | 5-10 步 |

#### 组件 2：Thought Generator — `G(pθ, s, k)`

给定当前状态 `s = [x, z1, z2, ..., zi]`，生成 **k 个候选的下一步想法**。

两种生成策略：

**策略 A：Sample (独立采样)**

```
for j in range(1, k+1):
    z^(j) ~ pθ_CoT( z_{i+1} | x, z_1, ..., z_i )
```

每次独立地从 CoT 分布采样一个 thought。用于**创意写作**——想法空间大，需要多样性。

```python
# 对应 prompt
prompt = f"""
前文: {s}
请为下一段落写出一个写作计划。发挥创意。
"""
# 重复调用 k 次，得到 k 个不同候选
```

**策略 B：Propose (顺序提议)**

```
[z^(1), z^(2), ..., z^(k)] ~ pθ_propose( z_{i+1}^{(1..k)} | s )
```

一次性生成 k 个不同候选。用于 **Game of 24**、**Crosswords**——想法空间有限，可以枚举。

```python
# 对应 prompt
prompt = f"""
当前数字: {numbers}
已有步骤: {steps}
请给出 5 种不同的下一步算式。
"""
# 一次调用，LLM 返回 5 个不同候选
```

#### 组件 3：State Evaluator — `V(pθ, s)`

评估一个状态（部分解）有多好，决定是否继续在这个方向上探索。

两种评估策略：

**策略 A：Value（独立评分）**

为每个候选独立打分。LLM 返回一个分类或数值：

```
sure / likely / impossible
或
1-10 分
或
0.0 - 1.0 的概率
```

对每个候选**采样多次**然后取平均，增加评估鲁棒性。

```python
# 实际 prompt (Game of 24)
prompt = f"""
当前数字: {numbers}
当前步骤: {steps}
下一步: {candidate}

请评估这个下一步是否有可能最终得到 24：
- sure (很有希望)
- likely (可能可行)
- impossible (不可能)

只回答 sure / likely / impossible。
"""
# 对每个候选调用 3 次，取众数作为最终评估
# sure = 1.0, likely = 0.5, impossible = 0.0
```

**策略 B：Vote（比较投票）**

把多个候选放在一起，让 LLM 投票选出最好的。

```python
# 实际 prompt (Creative Writing)
prompt = f"""
请分析以下几个写作计划，然后投票选出最有前途的一个：

候选 A: {plan_a}
候选 B: {plan_b}
候选 C: {plan_c}

请给出你的分析和投票结果。
"""
```

#### 组件 4：Search Algorithm（搜索算法）

ToT 支持两类搜索算法：

---

### 2.3 BFS 搜索详解

**BFS（广度优先搜索）**：每层探索 `b` 个最有希望的状态，逐层推进。

```
算法: ToT + BFS

输入: 初始问题 x, 最大深度 T, 束宽 b, 每节点生成数 k
输出: 解决方案 或 失败

1. level = 0
2. states = [x]                         # 当前层的状态集合
3. WHILE level < T AND 未找到解决方案:
   a. candidates = []                   # 所有候选 (状态, 评分)
   b. FOR EACH state IN states:
        # 对每个当前状态，生成 k 个下一步
        new_thoughts = G(pθ, state, k)
        FOR EACH thought IN new_thoughts:
            # 对每个候选，采样 n 次评估取平均
            score = mean(V(pθ, state + thought) for _ in range(n_evaluate))
            candidates.append((state + thought, score))
   c. # 修剪：只保留 top-b 个候选
      states = top_b(candidates, key=score)[:b]
   d. level += 1
4. 返回 states 中评分最高的完整解
```

**伪代码**：

```python
def tot_bfs(problem, max_depth=3, beam_width=5, generate_k=5, evaluate_n=3):
    states = [([problem], 1.0)]  # (path, score)

    for depth in range(max_depth):
        all_candidates = []

        for path, _ in states:
            # Step 1: Generate k candidate next thoughts
            new_thoughts = llm_propose(path, k=generate_k)

            for thought in new_thoughts:
                new_path = path + [thought]
                # Step 2: Evaluate n times, take average
                scores = [llm_value(new_path) for _ in range(evaluate_n)]
                avg_score = sum(scores) / len(scores)
                all_candidates.append((new_path, avg_score))

        # Step 3: Prune — keep top beam_width
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        states = all_candidates[:beam_width]

        # Check for solution
        for path, score in states:
            if is_solution(path):
                return path, score

    # Return best partial solution
    return states[0]

# Game of 24: beam_width=5, max_depth=3 (因为最多3步)
```

**关键参数**：

| 参数 | 含义 | Game of 24 值 |
|------|------|--------------|
| `b` (beam_width) | 每层保留多少状态 | 5 |
| `k` (n_generate) | 每个状态生成几个候选 | 5 |
| `n` (n_evaluate) | 每个候选评估几次 | 3 |
| `T` (max_depth) | 最大搜索深度 | 3 |

**为什么选择 BFS？**

BFS 适合**层数固定、每步都有明确进展**的任务：
- Game of 24：恰好 3 步，每步消耗 1 个数字
- Creative Writing：恰好 4-5 段，每段写完不能反悔

**参数影响**（来自论文消融实验）：

| b (束宽) | Game of 24 成功率 |
|----------|-------------------|
| 1 | ~20% (退化为贪心搜索) |
| 3 | ~45% |
| 5 | **74%** |
| 7 | ~72% (边际递减) |

---

### 2.4 DFS 搜索详解

**DFS（深度优先搜索）**：先深入探索最有希望的分支，遇到死胡同就回溯。

```
算法: ToT + DFS

输入: 当前状态 s, 深度 t, 最大深度 T
输出: 解决方案 或 None

1. IF t > T: RETURN None (超过深度限制)
2. IF is_solution(s): RETURN s
3. candidates = G(pθ, s, k)         # 生成 k 个候选
4. 对每个候选评分: scores = V(pθ, s+candidate)
5. 按评分从高到低排序 candidates
6. FOR EACH candidate IN sorted_candidates:
   a. IF score(candidate) == impossible: SKIP  # 剪枝
   b. result = DFS(s + candidate, t+1, T)     # 递归深入
   c. IF result != None: RETURN result         # 找到解就返回
7. RETURN None  (所有分支都失败，回溯)
```

**伪代码**：

```python
def tot_dfs(state, depth=0, max_depth=10, generate_k=5):
    if depth > max_depth:
        return None  # 超深度，回溯

    if is_solution(state):
        return state  # 找到解！

    # 生成候选下一步
    candidates = llm_propose(state, k=generate_k)

    # 评估 + 排序
    scored = []
    for c in candidates:
        score = llm_value(state + [c])
        if score > 0:  # 只保留有可能的
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    # 从最好的开始深搜
    for candidate, score in scored:
        new_state = state + [candidate]
        result = tot_dfs(new_state, depth + 1, max_depth)
        if result is not None:
            return result  # 找到解，不必继续

    return None  # 所有路都走不通，回溯
```

**为什么 Crosswords 用 DFS？**

填字游戏的特点是：
- **深度不固定**（5-10 步不等）
- **需要回溯**（某行填错后面全错，必须回头重填）
- **每个节点分支不多**（每个位置可填的单词有限）

BFS 对这种情况浪费严重（保留大量中间层状态），DFS 更自然——找到一条可行路线就走到底，走不通就退回来。

**DFS 的剪枝策略**：

```
if state_eval(state) == "impossible":
    # 跳过这个分支，不递归
    return None
```

论文中 DFS + 剪枝 vs 无剪枝：

| 策略 | Mini Crosswords 解决率 |
|------|----------------------|
| 无剪枝 DFS | 1/20 (5%) |
| 有剪枝 DFS | 4/20 (20%) |
| 有剪枝 DFS + Oracle 评估 | 7/20 (35%) |

---

### 2.5 Game of 24 完整流程

以 `[4, 5, 6, 10] → 24` 为例，走一遍完整的 ToT+BFS 流程。

**Step 0: 初始状态**

```
状态: (4, 5, 6, 10)
```

**Step 1: 第 1 层 — 生成 + 评估 + 修剪**

LLM 对初始状态生成 5 个候选下一步 (`k=5`, `propose` 模式)：

```
候选 1: 10 - 4 = 6 → (5, 6, 6)      [评估: sure=3/3 → 1.0]
候选 2: 4 + 5 = 9 → (6, 9, 10)      [评估: likely=2/3 → 0.67]
候选 3: 6 - 5 = 1 → (1, 4, 10)      [评估: impossible=2/3 → 0.0]
候选 4: 10 / 5 = 2 → (2, 4, 6)      [评估: likely=2/3 → 0.67]
候选 5: 4 * 5 = 20 → (6, 10, 20)    [评估: likely=2/3 → 0.67]
```

每个候选评估 3 次，取平均：

```
评分: 1.0, 0.67, 0.0, 0.67, 0.67
```

保留 top-5（束宽 b=5，实际全保留）：

```
入选: 候选1(1.0), 候选2(0.67), 候选4(0.67), 候选5(0.67)
淘汰: 候选3(0.0)  ← "impossible" 被剪掉
```

**Step 2: 第 2 层 — 从最好的状态继续**

以 `(5, 6, 6)` 为当前状态：

```
候选 1: 5 * 6 = 30 → (6, 30)        [评估: likely=3/3 → 0.67]
候选 2: 6 + 6 = 12 → (5, 12)        [评估: likely=2/3 → 0.67]
候选 3: 6 - 5 = 1 → (1, 6)          [评估: impossible=3/3 → 0.0]
...

对每个 level-1 存活的状态（4个）各生成 5 个候选 = 20 个候选
评估后保留 top-5
```

**Step 3: 第 3 层 — 找到解**

从 `(6, 30)` 出发：

```
候选: 30 - 6 = 24 → ()              [评估: sure=3/3]
```

`剩 0 个数字 = 找到解`：

```
最终答案: (5 * (10 - 4)) - 6 = 24
步骤:
  10 - 4 = 6
  5 * 6 = 30
  30 - 6 = 24
```

---

### 2.6 三个任务实验结果

| 任务 | IO Prompt | CoT | CoT-SC | **ToT** |
|------|-----------|-----|--------|---------|
| Game of 24 (GPT-4) | 7.3% | 4.0% | 9.0% | **74%** |
| Creative Writing (coherence) | 6.19 | 6.93 | — | **7.56** |
| Mini Crosswords (word-level) | 14% | 15.6% | — | **60%** |
| Mini Crosswords (game-level) | 0% | 1% | — | **20%** (35% w/ oracle) |

**核心发现**：

- **评估器的质量是关键瓶颈**：用 oracle 评估器代替 LLM 评估，Crosswords 从 20% → 35%，说明 LLM 的评估能力还不够好
- **剪枝不可或缺**：没有剪枝，Crosswords 只有 5% 解决率
- **回溯同样重要**：去掉 DFS 的回溯能力，性能大幅下降
- **束宽 b 有最优值**：b=5 最优，太小丢失好路径，太大增加噪音

---

## 三、Graph of Thoughts (GoT)

**论文**：[Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)
**作者**：Maciej Besta et al. (ETH Zurich)
**发表**：AAAI 2024
**代码**：https://github.com/spcl/graph-of-thoughts

### 3.1 核心思路

GoT 把 ToT 的**树**进一步推广为**任意有向图**：

```
CoT (链)                ToT (树)                 GoT (图)
───────────────────     ─────────────────────    ─────────────────────
a → b → c → d           a → b → c → d           a → b → c → d
                         ↘ e → f → g             ↘ e → f → g
                                                   ↓    ↓    ↓
                                                 合并  交叉  反馈
                                                   ↗    ↘
                                                  h ← i ← j
```

关键突破：一个 thought 可以**有多个前驱**（树中一个节点只有一个父节点）。这意味着：

1. **Aggregation（聚合）**：多条独立推理路径可以合并为一个综合 thought
2. **Refinement（精炼）**：一个 thought 可以自我迭代改进（自环）
3. **Feedback（反馈）**：后面的 thought 可以重新影响前面的 thought（循环）

### 3.2 四大模块架构

GoT 的工程架构由四个模块组成：

```
┌───────────────────────────────────────────────────┐
│                   Controller (控制器)              │
│  ┌─────────────────────┐ ┌─────────────────────┐  │
│  │   GoO (操作图)       │ │   GRS (推理状态图)   │  │
│  │   静态：定义可用的    │ │   动态：正在进行的    │  │
│  │   思维变换与依赖关系   │ │   所有思维的状态/分数 │  │
│  └─────────┬───────────┘ └──────────┬──────────┘  │
│            │                        │              │
│  ┌─────────▼────────────────────────▼──────────┐  │
│  │            Prompter (提示器)                  │  │
│  │   把图结构编码进 prompt 送给 LLM              │  │
│  └─────────────────────┬───────────────────────┘  │
│                        │                           │
│  ┌─────────────────────▼───────────────────────┐  │
│  │            Parser (解析器)                    │  │
│  │   从 LLM 响应中抽取信息，构造 thought state   │  │
│  └─────────────────────┬───────────────────────┘  │
│                        │                           │
│  ┌─────────────────────▼───────────────────────┐  │
│  │      Scoring & Validation (评分验证)          │  │
│  │   验证正确性 + 打分 + 排序 top-h              │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

#### 3.2.1 GoO — Graph of Operations（操作图）

**静态结构**，由开发者预先定义。它规定了：

- 有哪些类型的思维变换（operation）
- 每个操作的前置条件（需要哪些输入 thought）
- 每个操作的后置结果（产出的 thought 类型）
- 操作之间的执行顺序

```python
# 示例：排序任务的 GoO
class SortingGoO:
    operations = [
        {
            "id": "generate_init",
            "type": "generate",
            "predecessors": [],          # 无输入，从 scratch 生成
            "successors": ["score_init"], # 下一步是评分
            "description": "生成 k 个初始排序方案"
        },
        {
            "id": "score_init",
            "type": "score",
            "predecessors": ["generate_init"],  # 依赖初始生成
            "successors": ["select_top"],
            "score_function": sorting_error_score
        },
        {
            "id": "select_top",
            "type": "keep_top_h",
            "predecessors": ["score_init"],
            "successors": ["aggregate"],
            "top_h": 3
        },
        {
            "id": "aggregate",
            "type": "aggregate",        # 聚合操作——GoT 独有！
            "predecessors": ["select_top"],  # 合并 top-3 的最佳部分
            "successors": ["final_score"],
            "merge_strategy": "llm_merge"
        },
        {
            "id": "final_score",
            "type": "score",
            "predecessors": ["aggregate"],
            "successors": []
        }
    ]
```

#### 3.2.2 GRS — Graph Reasoning State（图推理状态）

**动态结构**，运行时维护。包含：

- 所有已创建的 thought（顶点）
- 每个 thought 的状态（pending / scored / finalized / pruned）
- 每个 thought 的分数
- thoughts 之间的有向边（谁生成了谁）
- 当前执行步数

```python
# GRS 运行时状态
class GraphReasoningState:
    thoughts: dict[str, Thought]  # thought_id → Thought
    edges: list[Edge]             # (source_id, target_id)
    current_step: int

class Thought:
    id: str
    content: str         # LLM 生成的文本
    score: float | None  # 评分（如果已评估）
    status: str          # pending | scored | pruned | final
    predecessors: list[str]  # 前驱 thought IDs
```

#### 3.2.3 Prompter（提示器）

把图结构信息编码进 prompt。与 ToT 的固定 prompt 不同，GoT 的 prompt 是**动态构建**的——根据当前节点在 GoO 中的位置、前驱 thought 的内容、任务类型动态组装。

```python
def build_prompt(operation, predecessors, task_spec):
    """为特定操作动态构建 prompt"""

    if operation.type == "generate":
        return f"""
任务: {task_spec.description}
请生成 {operation.k} 个候选方案。
"""

    elif operation.type == "aggregate":
        # GoT 独有 —— 合并多个前驱 thought
        context = "\n---\n".join(
            f"方案 {i+1}:\n{p.content}" for i, p in enumerate(predecessors)
        )
        return f"""
任务: {task_spec.description}

以下是 {len(predecessors)} 个候选方案：
{context}

请分析以上所有方案的优缺点，然后将它们的优势合并为一个更好的综合方案。
输出格式: 综合方案 + 合并理由
"""

    elif operation.type == "refine":
        return f"""
请审视并改进以下方案：
{predecessors[0].content}

找出其中的不足并给出改进版本。
"""
```

#### 3.2.4 Scoring & Validation（评分验证）

GoT 的评分系统比 ToT 更丰富：

**E(v, G, pθ) — 评分函数**：给 thought `v` 打分，可参考整个图 G。

```python
# 不同任务的评分函数

# 排序任务：计算错误排序数
def sorting_score(thought):
    sequence = extract_sequence(thought.content)
    error = 0
    for i in range(len(sequence) - 1):
        if sequence[i] > sequence[i+1]:
            error += 1
    return -error  # 负数，越接近 0 越好

# 集合交集任务：计算多余/缺少的元素
def set_intersection_score(thought, ground_truth_a, ground_truth_b):
    c = extract_set(thought.content)
    true_intersection = set(ground_truth_a) & set(ground_truth_b)
    extra_elements = len(c - true_intersection)    # 多出来的
    missing_elements = len(true_intersection - c)   # 漏掉的
    return -(extra_elements + missing_elements)

# 关键字计数任务
def keyword_counting_score(thought, ground_truth):
    counts = extract_counts(thought.content)
    errors = [abs(counts.get(k, 0) - v) for k, v in ground_truth.items()]
    return -sum(errors)
```

**R(G, pθ, h) — 排名函数**：返回图中分数最高的 h 个 thought。

### 3.3 三种思维变换操作

这是 GoT 最核心的创新。不同于 ToT 只能做 "生成"，GoT 定义了三种基本操作：

#### 操作 1：Generation（生成）

```
一个 thought → 多个新 thought

图操作：
  V⁺ = {v1⁺, v2⁺, ..., vk⁺}
  E⁺ = {(v, v1⁺), (v, v2⁺), ..., (v, vk⁺)}

     v (父)
    /|\
   v1 v2 v3 (k 个子)
```

从单个父 thought 生成 k 个候选新 thought。与 ToT 的 "propose / sample" 完全相同。

#### 操作 2：Aggregation（聚合）— GoT 独有

```
多个 thought → 合并为一个综合 thought

图操作：
  V⁺ = {v⁺}  (1 个新顶点)
  E⁺ = {(v1, v⁺), (v2, v⁺), ..., (vk, v⁺)}  (k 条入边)

  v1  v2  v3
   \   |   /
    \  |  /
      v⁺  (合并)
```

**这是 GoT 超越 ToT 的核心能力。** 多条独立推理路径同时探索，然后聚合成一个综合答案——这在树结构中不可能（树中一个节点只有一个父节点）。

```python
# 排序任务的聚合操作
def aggregate_sorting_plans(plans: list[str], llm) -> str:
    """
    3 个独立排序方案 → 1 个综合最优方案

    例如：
    方案1: [3, 1, 4, 2, 8, 5, 7, 6]
    方案2: [1, 3, 2, 4, 5, 8, 6, 7]
    方案3: [1, 2, 3, 4, 5, 6, 8, 7]

    LLM 分析后输出：
    综合: [1, 2, 3, 4, 5, 6, 7, 8]
    理由: 前半段参考方案3，后半段调整方案2第7位
    """
    prompt = f"""
以下是 {len(plans)} 个候选排序方案：

"""
    for i, plan in enumerate(plans):
        prompt += f"方案 {i+1}: {plan}\n"

    prompt += """
请分析所有方案，找出每个方案中最正确的部分，
然后将它们合并为一个最佳综合排序。
"""
    return llm(prompt)
```

#### 操作 3：Refinement（精炼）

```
一个 thought → 改进后的同一 thought（自我迭代）

图操作：
  V⁺ = {}  (不新建顶点)
  E⁺ = {(v, v)}  (自环)

  ┌──┐
  │ v │←─┐
  └──┘  │  改一次不够，再改
     └──┘
```

```python
def refine_thought(thought: str, llm, max_iterations=3) -> str:
    """反复改进同一个 thought，直到收敛"""
    current = thought
    for i in range(max_iterations):
        prompt = f"""
请审视以下方案，找出不足并改进：

{current}

改进后的版本：
"""
        improved = llm(prompt)
        if improved == current:
            break  # 收敛，不再改
        current = improved
    return current
```

### 3.4 延迟-体积权衡（Latency-Volume Tradeoff）

论文提出了一个量化指标来比较不同推理拓扑的效率：

**Volume of a Thought**：从根节点出发，能通过有向路径到达这个 thought 的中间 thought 总数。

```
CoT:   ● → ● → ● → ●    volume 随深度线性增长
       1    2    3    4

ToT:        ●
           / \
          ●   ●         每个分支的 volume 受限于路径长度
         /     \
        ●       ●       volume 增长受限于 log_k(N)

GoT:       ●──→●──→●
           │    │    │
           ●←──┼──→●   聚合使 volume 超过路径长度
           │    │
           ●←───┘       volume 可以接近总节点数 N
```

| 方法 | 延迟（生成顺序） | 最大 Volume |
|------|-----------------|-------------|
| CoT | N（串行） | N |
| CoT-SC | ~N/k | N/k |
| ToT | logₖ(N) | logₖ(N) |
| **GoT** | **logₖ(N)** ✓ | **N** ✓ |

GoT 是唯一同时实现低延迟和最大信息累积的方法——**聚合（Aggregation）**使得多条路径的信息集中到单个 thought。

### 3.5 四种任务实验结果

| 任务 | CoT | ToT | GoT | GoT 优势 |
|------|-----|-----|-----|---------|
| **排序** (128 个数字) | 基准 | — | **-62% 错误** | vs ToT: 62% 更少错误 |
| **集合操作** | 基准 | — | **34.6 正确率** | 精确计算集合交集/并集 |
| **关键词计数** | 基准 | — | **46.7 正确率** | 多个文档并行统计再合并 |
| **文档合并** | 基准 | — | **1.43 BR score** | 多文档去重合并 |

**排序任务详细分析**：

ToT 在排序上效果差（树结构不适合排序），因为排序天然可以分治（divide and conquer）：
1. 把序列分成多个子序列 → 分别排序（并行 generation）
2. 把排好的子序列合并（aggregation）
3. 对合并结果微调（refinement）

GoT 完美匹配这个模式：`split → parallel_sort → aggregate → refine`

**成本对比**：

| 任务 | ToT 成本 | GoT 成本 | 节省 |
|------|---------|---------|------|
| 排序 128 | 100% | **69%** | **-31%** |

GoT 比 ToT 节省约 31% 的 API 调用次数——因为不需要为每个非叶子节点生成大量候选，聚合操作天然合并信息。

---

## 四、ToT vs GoT 对比

| 维度 | ToT | GoT |
|------|-----|-----|
| **图结构** | 树（单父节点） | 任意有向图（多父节点、自环） |
| **核心操作** | 仅 Generation | Generation + Aggregation + Refinement |
| **评估方式** | LLM 评分 / 投票 | 任务特定评分函数 + LLM 评分 |
| **搜索策略** | BFS / DFS | 由 GoO 静态定义执行图 |
| **反馈循环** | 不支持 | 支持（Refinement 自环） |
| **适合任务** | 探索类问题（Game of 24, Crosswords） | 可分解类问题（排序、集合、合并） |
| **成本** | 高（大量候选评估） | 中（聚合减少冗余） |
| **工程复杂度** | 简单（BFS/DFS 即用） | 中等（需预定义 GoO） |
| **表达能力** | 中 | **最强**：几乎任意推理拓扑 |

---

## 五、完整代码演示

以下代码基于 `G:\source-code\LearningBeyondGradients\tot_got_demo\` 目录，使用 OpenAI 兼容 API，纯 Python 实现 ToT 和 GoT 算法。

### 5.1 toT.py — Tree of Thoughts 完整实现

```python
"""
Tree of Thoughts (ToT) — 完整实现
支持 BFS 和 DFS 两种搜索策略

论文: Tree of Thoughts: Deliberate Problem Solving with Large Language Models
作者: Shunyu Yao et al., NeurIPS 2023
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openai import AsyncOpenAI
from dotenv import load_dotenv

# ── 配置 ────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


async def llm(prompt: str, temperature: float = 0.7) -> str:
    """调用 LLM"""
    resp = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class ThoughtNode:
    """树中的一个思维节点"""
    content: str                          # thought 文本内容
    children: list["ThoughtNode"] = field(default_factory=list)
    score: float = 0.0                    # 评分 (0.0 - 1.0)
    depth: int = 0
    parent: "ThoughtNode | None" = None

    def path(self) -> list[str]:
        """从根到当前节点的路径"""
        if self.parent is None:
            return [self.content]
        return self.parent.path() + [self.content]


@dataclass
class ToTConfig:
    """ToT 配置参数"""
    max_depth: int = 3           # T: 最大搜索深度
    beam_width: int = 5          # b: 束宽（BFS 每层保留的状态数）
    n_generate: int = 5          # k: 每个状态生成几个候选
    n_evaluate: int = 3          # n: 每个候选评估几次
    search_mode: str = "bfs"     # "bfs" | "dfs"
    temperature: float = 0.7


# ── 任务定义 ──────────────────────────────────────────────

class Game24Task:
    """
    Game of 24: 给定 4 个数字，用加减乘除得到 24。

    状态表示: "3 5 6 10"  (剩余可用数字)
    thought: "5 * 6 = 30 (left: 3 10 30)"  (一步算式)
    """

    @staticmethod
    def is_solution(path: list[str]) -> bool:
        """检查是否已经得到 24"""
        if not path:
            return False
        # 最后一步: "... (left: 24)" 或 "... = 24"
        last = path[-1]
        return re.search(r"(?:left:\s*24\b|=\s*24\b)", last) is not None

    @staticmethod
    def is_terminal(path: list[str]) -> bool:
        """检查是否无法继续（所有数字用完但未得到24）"""
        if not path:
            return False
        last = path[-1]
        # 剩 1 个数字且不是 24
        m = re.search(r"left:\s*([\d\s]+)\)", last)
        if m:
            nums = m.group(1).strip().split()
            return len(nums) <= 1 and not Game24Task.is_solution(path)
        return False

    @staticmethod
    def extract_numbers(text: str) -> list[int]:
        """从状态文本中提取剩余数字"""
        nums = re.findall(r"\b\d+\b", text)
        return [int(n) for n in nums]


class Game24ToT:
    """Game of 24 的 ToT Prompt"""

    @staticmethod
    def propose_prompt(numbers: list[int], path: list[str]) -> str:
        """生成 k 个候选下一步"""
        steps_text = "\n".join(f"  {s}" for s in path) if path else "  (starting)"
        nums_text = " ".join(str(n) for n in numbers)
        return f"""You are solving Game of 24. Use the numbers {nums_text} to reach 24 using + - * /.
Each step uses exactly TWO numbers and replaces them with the result.

Current numbers: {nums_text}
Previous steps:
{steps_text}

Propose {5} different possible next steps. Each step must:
- Use TWO of the remaining numbers
- Apply exactly one operation (+ - * /)
- Produce a valid integer result (no fractions)
- Show the result in the format: A op B = C (left: remaining numbers)

Output exactly 5 lines, each in the format:
  [step] (left: [remaining numbers])
"""
    @staticmethod
    def value_prompt(path: list[str], candidate: str) -> str:
        """评估一个候选是否有可能达到 24"""
        steps = "\n".join(f"  {s}" for s in path)
        return f"""You are evaluating a partial solution for Game of 24.

Previous steps:
{steps}

Proposed next step:
  {candidate}

Evaluate: does this step look promising for eventually reaching exactly 24?
Answer with EXACTLY ONE of these three words:
  sure     — very likely to succeed
  likely   — might work
  impossible — cannot lead to 24

Your answer (one word only):"""


# ── ToT 核心算法 ──────────────────────────────────────────

class TreeOfThoughts:
    """Tree of Thoughts 搜索器"""

    def __init__(self, config: ToTConfig, task: Game24Task, prompts: Game24ToT):
        self.cfg = config
        self.task = task
        self.prompts = prompts

    # ── BFS ───────────────────────────────────────────────

    async def bfs(self, initial_numbers: list[int]) -> ThoughtNode | None:
        """BFS 搜索：逐层探索，每层保留 beam_width 个最好状态"""

        root = ThoughtNode(
            content=f"Start: {' '.join(str(n) for n in initial_numbers)}",
            score=1.0,
            depth=0,
        )

        frontier: list[ThoughtNode] = [root]

        for depth in range(self.cfg.max_depth):
            print(f"\n{'='*50}")
            print(f"BFS Depth {depth + 1}: {len(frontier)} states in frontier")
            print(f"{'='*50}")

            all_candidates: list[ThoughtNode] = []

            for node in frontier:
                numbers = self.task.extract_numbers(node.content)
                path = node.path()

                # ---- Step 1: GENERATE ----
                prompt = self.prompts.propose_prompt(numbers, path)
                response = await llm(prompt, temperature=self.cfg.temperature)
                candidates_raw = self._parse_proposals(response)

                print(f"  State [{numbers}]: generated {len(candidates_raw)} candidates")

                # ---- Step 2: EVALUATE ----
                for candidate_text in candidates_raw:
                    scores = []
                    for _ in range(self.cfg.n_evaluate):
                        eval_prompt = self.prompts.value_prompt(path, candidate_text)
                        eval_resp = await llm(eval_prompt, temperature=0.3)
                        scores.append(self._parse_value(eval_resp))

                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0:
                        child = ThoughtNode(
                            content=candidate_text,
                            score=avg_score,
                            depth=depth + 1,
                            parent=node,
                        )
                        all_candidates.append(child)
                        print(f"    → {candidate_text[:50]}... score={avg_score:.2f}")

            if not all_candidates:
                print("  No candidates survived evaluation. Stopping.")
                break

            # ---- Step 3: PRUNE ----
            all_candidates.sort(key=lambda n: n.score, reverse=True)
            frontier = all_candidates[:self.cfg.beam_width]

            # ---- Check for solutions ----
            for node in frontier:
                if self.task.is_solution(node.path()):
                    print(f"\n  ✅ SOLUTION FOUND at depth {node.depth}!")
                    return node

            print(f"  Pruned to top {len(frontier)} states")

        # 没找到解，返回最高分节点
        return frontier[0] if frontier else None

    # ── DFS ───────────────────────────────────────────────

    async def dfs(self, initial_numbers: list[int]) -> ThoughtNode | None:
        """DFS 搜索：深入探索最有希望的分支，失败则回溯"""

        root = ThoughtNode(
            content=f"Start: {' '.join(str(n) for n in initial_numbers)}",
            score=1.0,
            depth=0,
        )

        result = await self._dfs_recursive(root)
        return result

    async def _dfs_recursive(self, node: ThoughtNode) -> ThoughtNode | None:
        """DFS 递归搜索"""
        indent = "  " * node.depth
        path = node.path()

        # 检查是否已经是解
        if self.task.is_solution(path):
            print(f"{indent}✅ FOUND at depth {node.depth}")
            return node

        # 深度限制
        if node.depth >= self.cfg.max_depth:
            return None

        # 死胡同检测
        if self.task.is_terminal(path):
            return None

        numbers = self.task.extract_numbers(node.content)
        if len(numbers) <= 1:
            return None  # 数字用完但未得 24

        # 生成候选
        prompt = self.prompts.propose_prompt(numbers, path)
        response = await llm(prompt, temperature=self.cfg.temperature)
        candidates_raw = self._parse_proposals(response)

        # 评估 + 排序
        scored_candidates = []
        for candidate_text in candidates_raw:
            scores = []
            for _ in range(self.cfg.n_evaluate):
                eval_prompt = self.prompts.value_prompt(path, candidate_text)
                eval_resp = await llm(eval_prompt, temperature=0.3)
                scores.append(self._parse_value(eval_resp))
            avg_score = sum(scores) / len(scores)

            if avg_score >= 0.5:  # ignore "impossible"
                child = ThoughtNode(
                    content=candidate_text,
                    score=avg_score,
                    depth=node.depth + 1,
                    parent=node,
                )
                scored_candidates.append(child)

        # 从最好的开始深搜
        scored_candidates.sort(key=lambda n: n.score, reverse=True)
        for child in scored_candidates:
            print(f"{indent}Depth {node.depth}: trying {child.content[:50]}... "
                  f"(score={child.score:.2f})")
            result = await self._dfs_recursive(child)
            if result is not None:
                return result  # 找到解
            print(f"{indent}  ← backtracking...")

        return None  # 所有分支都失败

    # ── 辅助函数 ──────────────────────────────────────────

    @staticmethod
    def _parse_proposals(response: str) -> list[str]:
        """解析 LLM 返回的候选列表"""
        lines = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去除编号前缀: "1. " or "1) "
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if "left:" in cleaned.lower():
                lines.append(cleaned)
        return lines[:5]  # 最多 5 个

    @staticmethod
    def _parse_value(response: str) -> float:
        """解析 LLM 返回的评估值"""
        text = response.strip().lower()
        if "sure" in text:
            return 1.0
        elif "likely" in text:
            return 0.5
        elif "impossible" in text:
            return 0.0
        # 尝试解析数字
        nums = re.findall(r"(\d+\.?\d*)", text)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        return 0.0


# ── Demo ──────────────────────────────────────────────────

async def demo_tot():
    """演示 ToT 在 Game of 24 上的完整流程"""

    print("=" * 60)
    print("Tree of Thoughts (ToT) — Game of 24 Demo")
    print("=" * 60)

    puzzle = [4, 5, 6, 10]
    print(f"\nPuzzle: {puzzle} → 24")

    # BFS 搜索
    config_bfs = ToTConfig(
        max_depth=3,
        beam_width=5,
        n_generate=5,
        n_evaluate=3,
        search_mode="bfs",
    )

    tot = TreeOfThoughts(config_bfs, Game24Task(), Game24ToT())

    print("\n── Running BFS ──")
    result = await tot.bfs(puzzle)

    if result:
        print(f"\n{'='*60}")
        print("BFS Result (path from root):")
        print(f"{'='*60}")
        for i, step in enumerate(result.path()):
            print(f"  Step {i}: {step}")
    else:
        print("\nNo solution found with BFS.")

    # DFS 搜索（同一个 puzzle）
    config_dfs = ToTConfig(
        max_depth=3,
        n_generate=5,
        n_evaluate=3,
        search_mode="dfs",
    )

    tot_dfs = TreeOfThoughts(config_dfs, Game24Task(), Game24ToT())

    print("\n── Running DFS ──")
    result = await tot_dfs.dfs(puzzle)

    if result:
        print(f"\n{'='*60}")
        print("DFS Result (path from root):")
        print(f"{'='*60}")
        for i, step in enumerate(result.path()):
            print(f"  Step {i}: {step}")


if __name__ == "__main__":
    asyncio.run(demo_tot())
```

### 5.2 goT.py — Graph of Thoughts 完整实现

```python
"""
Graph of Thoughts (GoT) — 完整实现
支持 Generation、Aggregation、Refinement 三种思维变换

论文: Graph of Thoughts: Solving Elaborate Problems with Large Language Models
作者: Maciej Besta et al., AAAI 2024
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

# ── 配置 ────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


async def llm(prompt: str, temperature: float = 0.7) -> str:
    resp = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class Thought:
    """图中的一个思维顶点"""
    id: str
    content: str
    score: float | None = None
    status: str = "pending"  # pending | scored | pruned | final
    predecessors: list[str] = field(default_factory=list)  # 前驱 Thought IDs
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphReasoningState:
    """GRS: 运行时图推理状态"""
    thoughts: dict[str, Thought] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (src, dst)
    step_counter: int = 0
    topology: str = "custom"

    def add_thought(self, thought: Thought):
        self.thoughts[thought.id] = thought

    def add_edge(self, src: str, dst: str):
        self.edges.append((src, dst))
        if dst in self.thoughts:
            self.thoughts[dst].predecessors.append(src)

    def get_predecessors(self, thought_id: str) -> list[Thought]:
        preds = self.thoughts[thought_id].predecessors
        return [self.thoughts[p] for p in preds if p in self.thoughts]

    def top_h(self, h: int) -> list[Thought]:
        """返回分数最高的 h 个 thought"""
        scored = [t for t in self.thoughts.values() if t.score is not None]
        scored.sort(key=lambda t: t.score, reverse=True)
        return scored[:h]


class GraphOfOperations:
    """
    GoO: 静态操作图

    定义任务可用的操作序列，每个操作有：
    - type: "generate" | "aggregate" | "refine"
    - predecessors: 依赖的前驱操作 IDs
    - params: 操作参数 (k, prompts 等)
    """

    def __init__(self):
        self.operations: dict[str, dict] = {}

    def add_operation(self, op_id: str, op_type: str, predecessors: list[str],
                      **params):
        self.operations[op_id] = {
            "id": op_id,
            "type": op_type,
            "predecessors": predecessors,
            "params": params,
        }

    def execution_order(self) -> list[str]:
        """拓扑排序，返回执行顺序"""
        indegree = {op_id: 0 for op_id in self.operations}
        for op in self.operations.values():
            for pred in op["predecessors"]:
                if pred in indegree:
                    indegree[op["id"]] += 1

        queue = [op_id for op_id, d in indegree.items() if d == 0]
        order = []
        while queue:
            op_id = queue.pop(0)
            order.append(op_id)
            for other_id, other in self.operations.items():
                if op_id in other["predecessors"]:
                    indegree[other_id] -= 1
                    if indegree[other_id] == 0:
                        queue.append(other_id)
        return order


# ── GoT 核心引擎 ──────────────────────────────────────────

class GraphOfThoughts:
    """
    GoT 引擎：根据 GoO 定义的操作图执行推理，
    动态维护 GRS，每一步调用 LLM 完成思维变换。
    """

    def __init__(self, goo: GraphOfOperations):
        self.goo = goo
        self.grs = GraphReasoningState()

    async def run(self, task_input: str) -> None:
        """
        按 GoO 的拓扑顺序执行所有操作。
        每个操作从 GRS 中取前驱 thought，执行变换，结果写回 GRS。
        """
        order = self.goo.execution_order()
        print(f"\nGoO Execution Order: {' → '.join(order)}\n")

        # 把初始输入作为第一个 thought
        init_id = "thought_0"
        self.grs.add_thought(Thought(
            id=init_id, content=f"Task: {task_input}", score=1.0, status="scored"
        ))

        # 记录每个 operation 产出的 thought IDs（供后续操作引用的 key）
        # 简化：用 op_id 作为 map key
        op_outputs: dict[str, list[str]] = {"_input": [init_id]}

        for op_id in order:
            op = self.goo.operations[op_id]
            op_type = op["type"]
            params = op["params"]

            print(f"{'─'*50}")
            print(f"Executing: {op_id} (type={op_type})")

            if op_type == "generate":
                thought_ids = await self._execute_generate(op, op_outputs, params)
            elif op_type == "aggregate":
                thought_ids = await self._execute_aggregate(op, op_outputs, params)
            elif op_type == "refine":
                thought_ids = await self._execute_refine(op, op_outputs, params)
            elif op_type == "score":
                thought_ids = await self._execute_score(op, op_outputs, params)
            elif op_type == "keep_top":
                thought_ids = self._execute_keep_top(op, op_outputs, params)
            else:
                raise ValueError(f"Unknown operation type: {op_type}")

            op_outputs[op_id] = thought_ids
            print(f"  → Produced {max(0, len(thought_ids))} thoughts")

        print(f"\n{'='*60}")
        print(f"Final GRS: {len(self.grs.thoughts)} thoughts, "
              f"{len(self.grs.edges)} edges")

    # ── 操作实现 ────────────────────────────────────────

    async def _execute_generate(
        self, op: dict, op_outputs: dict, params: dict
    ) -> list[str]:
        """
        从 k 个前驱 thought 各生成 m 个新候选。

        prompt_template 中可以使用:
          {predecessor_content}  — 前驱 thought 的文本
          {k}                    — 要生成的候选数
        """
        k = params.get("k", 3)
        prompt_template = params["prompt_template"]
        pred_op_ids = op["predecessors"]
        new_ids = []

        for pred_op_id in pred_op_ids:
            # 找到该操作产出的 thought
            pred_thought_ids = op_outputs.get(pred_op_id, [])
            for pred_tid in pred_thought_ids:
                pred = self.grs.thoughts.get(pred_tid)
                if not pred:
                    continue

                prompt = prompt_template.format(
                    predecessor_content=pred.content,
                    k=k,
                )
                response = await llm(prompt, temperature=params.get("temperature", 0.7))
                candidates = self._parse_candidates(response)

                for cand_text in candidates:
                    new_id = f"thought_{self.grs.step_counter}"
                    self.grs.step_counter += 1
                    new_thought = Thought(
                        id=new_id,
                        content=cand_text,
                        predecessors=[pred_tid],
                    )
                    self.grs.add_thought(new_thought)
                    self.grs.add_edge(pred_tid, new_id)
                    new_ids.append(new_id)

        return new_ids

    async def _execute_aggregate(
        self, op: dict, op_outputs: dict, params: dict
    ) -> list[str]:
        """
        GoT 独有操作：合并多个前驱 thought 为一个综合 thought。

        所有前驱操作产出的 thought 被合并。
        """
        prompt_template = params["prompt_template"]

        # 收集所有前驱操作的所有 thought
        all_pred_thoughts = []
        for pred_op_id in op["predecessors"]:
            for tid in op_outputs.get(pred_op_id, []):
                t = self.grs.thoughts.get(tid)
                if t:
                    all_pred_thoughts.append(t)

        if not all_pred_thoughts:
            return []

        # 构建 prompt：所有候选方案的文本
        candidates_text = "\n---\n".join(
            f"Candidate {i+1}:\n{t.content}"
            for i, t in enumerate(all_pred_thoughts)
        )

        prompt = prompt_template.format(
            candidates=candidates_text,
            n=len(all_pred_thoughts),
        )

        response = await llm(prompt, temperature=params.get("temperature", 0.3))
        aggregated_content = response.strip()

        new_id = f"thought_{self.grs.step_counter}"
        self.grs.step_counter += 1
        aggregated = Thought(
            id=new_id,
            content=aggregated_content,
            predecessors=[t.id for t in all_pred_thoughts],
        )
        self.grs.add_thought(aggregated)
        for t in all_pred_thoughts:
            self.grs.add_edge(t.id, new_id)

        return [new_id]

    async def _execute_refine(
        self, op: dict, op_outputs: dict, params: dict
    ) -> list[str]:
        """
        精炼：迭代改进 thought，直到收敛或达到最大迭代次数。
        """
        prompt_template = params["prompt_template"]
        max_iterations = params.get("max_iterations", 3)

        pred_op_ids = op["predecessors"]
        refined_ids = []

        for pred_op_id in pred_op_ids:
            for tid in op_outputs.get(pred_op_id, []):
                t = self.grs.thoughts.get(tid)
                if not t:
                    continue

                current = t.content
                prev = None
                for _ in range(max_iterations):
                    prompt = prompt_template.format(content=current)
                    response = await llm(prompt, temperature=params.get(
                        "temperature", 0.5
                    ))
                    new_version = response.strip()
                    if new_version == prev:
                        break  # 收敛
                    prev = current
                    current = new_version

                new_id = f"thought_{self.grs.step_counter}"
                self.grs.step_counter += 1
                refined = Thought(
                    id=new_id,
                    content=current,
                    predecessors=[tid],
                )
                self.grs.add_thought(refined)
                self.grs.add_edge(tid, new_id)
                refined_ids.append(new_id)

        return refined_ids

    async def _execute_score(
        self, op: dict, op_outputs: dict, params: dict
    ) -> list[str]:
        """
        评分：对指定操作产出的所有 thought 打分。
        支持两种评分方式：
          1. "llm_judge": LLM 打分
          2. "rule_based": 用评分函数打分
        """
        score_mode = params.get("score_mode", "llm_judge")
        prompt_template = params.get("prompt_template", "")

        for pred_op_id in op["predecessors"]:
            for tid in op_outputs.get(pred_op_id, []):
                t = self.grs.thoughts.get(tid)
                if not t:
                    continue

                if score_mode == "llm_judge":
                    prompt = prompt_template.format(thought_content=t.content)
                    response = await llm(prompt, temperature=0.2)
                    score = self._parse_score(response)
                elif score_mode == "rule_based":
                    score_fn = params.get("score_function")
                    if score_fn:
                        score = score_fn(t.content)
                    else:
                        score = 0.5
                else:
                    score = 0.5

                t.score = score
                t.status = "scored"
                print(f"    Scored {tid}: {score:.2f}")

        # Score operations don't produce new thoughts
        return []

    def _execute_keep_top(
        self, op: dict, op_outputs: dict, params: dict
    ) -> list[str]:
        """
        保留 top-h 个最高分 thought，其余标记为 pruned。
        """
        h = params.get("h", 3)
        all_ids = []
        for pred_op_id in op["predecessors"]:
            all_ids.extend(op_outputs.get(pred_op_id, []))

        scored = []
        for tid in all_ids:
            t = self.grs.thoughts.get(tid)
            if t and t.score is not None:
                scored.append((tid, t.score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 保留 top-h，其余剪枝
        keep = scored[:h]
        prune = scored[h:]

        for tid, _ in prune:
            self.grs.thoughts[tid].status = "pruned"

        print(f"    Keeping {len(keep)}, pruning {len(prune)}")
        return [tid for tid, _ in keep]

    # ── 辅助函数 ──────────────────────────────────────────

    @staticmethod
    def _parse_candidates(response: str) -> list[str]:
        """解析 LLM 返回的多个候选"""
        lines = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if len(line) > 3:
                lines.append(line)
        return lines[:10]

    @staticmethod
    def _parse_score(response: str) -> float:
        """解析 LLM 返回的分数"""
        nums = re.findall(r"(\d+\.?\d*)", response)
        if nums:
            score = float(nums[0])
            return max(0.0, min(1.0, score / 10.0 if score > 1 else score))
        if "good" in response.lower() or "correct" in response.lower():
            return 0.8
        if "bad" in response.lower() or "wrong" in response.lower():
            return 0.2
        return 0.5


# ── Demo: Sorting Task ────────────────────────────────────

def sorting_error_score(text: str) -> float:
    """
    排序任务评分函数：
    计算排序错误程度，返回 0.0 (完全错误) - 1.0 (完全正确)
    """
    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) < 2:
        return 0.0

    sequence = [int(n) for n in nums]
    errors = 0
    for i in range(len(sequence) - 1):
        if sequence[i] > sequence[i + 1]:
            errors += 1

    max_errors = len(sequence) - 1
    if max_errors == 0:
        return 1.0
    error_rate = errors / max_errors
    return max(0.0, 1.0 - error_rate)


def build_sorting_goo() -> GraphOfOperations:
    """
    构建排序任务的 GoO：

    1. generate_initial  → 生成 k 个初始排序方案
    2. score_initial     → 用排序错误函数评分
    3. keep_best         → 保留 top-3
    4. aggregate         → 合并 3 个最佳方案
    5. refine_aggregated → 精炼合并结果
    6. final_score       → 最终评分
    """
    goo = GraphOfOperations()

    goo.add_operation(
        "generate_initial", "generate", [],
        k=5,
        prompt_template=(
            "Sort the following numbers in ascending order:\n"
            "{predecessor_content}\n\n"
            "Provide exactly {k} different sorted sequences, each on a new line. "
            "Output only the numbers separated by spaces."
        ),
        temperature=0.7,
    )

    goo.add_operation(
        "score_initial", "score", ["generate_initial"],
        score_mode="rule_based",
        score_function=sorting_error_score,
    )

    goo.add_operation(
        "keep_best", "keep_top", ["score_initial"],
        h=3,
    )

    goo.add_operation(
        "aggregate", "aggregate", ["keep_best"],
        prompt_template=(
            "You are given {n} candidate sorted sequences:\n\n"
            "{candidates}\n\n"
            "Analyze all candidates. Find the parts that each one got right, "
            "and produce ONE final correctly sorted sequence. "
            "Output only the numbers separated by spaces, no explanation."
        ),
        temperature=0.3,
    )

    goo.add_operation(
        "refine_aggregated", "refine", ["aggregate"],
        prompt_template=(
            "Review this sorted sequence for any errors:\n"
            "{content}\n\n"
            "If it's correct, output it unchanged. "
            "If there are errors, output the corrected sequence. "
            "Only output the numbers separated by spaces."
        ),
        max_iterations=3,
        temperature=0.3,
    )

    goo.add_operation(
        "final_score", "score", ["refine_aggregated"],
        score_mode="rule_based",
        score_function=sorting_error_score,
    )

    return goo


async def demo_got():
    """演示 GoT 在排序任务上的完整流程"""

    print("=" * 60)
    print("Graph of Thoughts (GoT) — Sorting Demo")
    print("=" * 60)

    # 生成待排序序列
    import random
    random.seed(42)
    unsorted = [random.randint(1, 100) for _ in range(16)]
    task_input = " ".join(str(n) for n in unsorted)
    print(f"\nUnsorted: {unsorted}")
    print(f"Expected: {sorted(unsorted)}")

    goo = build_sorting_goo()
    got = GraphOfThoughts(goo)
    await got.run(task_input)

    # 打印最终结果
    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")

    best = got.grs.top_h(1)
    if best:
        print(f"\nBest thought (score={best[0].score:.3f}):")
        print(f"  {best[0].content}")

    # 打印 GRS 摘要
    for tid, t in got.grs.thoughts.items():
        status_icon = {"scored": "●", "pruned": "✕", "pending": "○"}
        icon = status_icon.get(t.status, "?")
        score_str = f" score={t.score:.2f}" if t.score is not None else ""
        preds = f" ← [{', '.join(t.predecessors[:3])}]" if t.predecessors else ""
        print(f"  {icon} {tid}: {t.content[:60]}...{score_str}{preds}")


if __name__ == "__main__":
    asyncio.run(demo_got())
```

### 5.3 CoT, ToT, GoT 三者对比 Demo

```python
"""
对比 Demo: 同一个推理任务用 CoT、ToT、GoT 三种方式求解
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


async def llm(prompt: str, temperature: float = 0.7) -> str:
    resp = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


# ── 同一任务：多文档去重合并 ────────────────────────────────

documents = [
    "AI will transform healthcare by enabling early diagnosis through pattern recognition in medical imaging, personalized treatment plans, and automated administrative workflows.",
    "In education, AI-powered tutoring systems adapt to individual learning styles, provide instant feedback, and help teachers identify struggling students before they fall behind.",
    "Healthcare AI applications include medical image analysis, drug discovery acceleration, and virtual nursing assistants that monitor patients remotely.",
    "The future of education with AI involves personalized learning paths, automated grading systems, and intelligent content recommendation that matches student interests and abilities.",
    "Climate science benefits from AI through improved weather prediction models, carbon footprint optimization, and smart grid management for renewable energy distribution.",
]


async def run_cot(docs: list[str]) -> str:
    """Chain-of-Thought: 逐篇处理，线性链条"""
    print("\n─── CoT: Linear Chain ───")

    prompt = f"""You have {len(docs)} documents about AI applications.
For each document, extract the key application areas.
Then merge them, removing duplicates and contradictions.

Documents:
"""
    for i, doc in enumerate(docs):
        prompt += f"\nDoc {i+1}: {doc}\n"

    prompt += """
First, list the main topics of each document.
Then, merge them into a single, non-redundant summary.

Final merged summary:
"""
    return await llm(prompt)


async def run_tot(docs: list[str]) -> str:
    """Tree of Thoughts: 两两合并形成树"""
    print("\n─── ToT: Pairwise Merge Tree ───")

    # Level 0: 原始文档
    thoughts = docs.copy()

    # Level 1: 两两合并
    merged_level1 = []
    for i in range(0, len(thoughts), 2):
        pair = thoughts[i:i+2]
        prompt = f"""Merge the following texts, removing duplicates and contradictions:

Text A: {pair[0]}

Text B: {pair[1] if len(pair) > 1 else pair[0]}

Merged text:
"""
        result = await llm(prompt)
        merged_level1.append(result.strip())
        print(f"  Merged pair {i//2 + 1}: {len(merged_level1[-1])} chars")

    # Level 2: 最终合并
    final_prompt = f"""Merge the following partial summaries, removing all duplicates:

Summary 1: {merged_level1[0]}
Summary 2: {merged_level1[1] if len(merged_level1) > 1 else merged_level1[0]}
Summary 3: {merged_level1[2] if len(merged_level1) > 2 else ''}

Final comprehensive, non-redundant summary:
"""
    return await llm(final_prompt)


async def run_got(docs: list[str]) -> str:
    """Graph of Thoughts: 独立分析 → 聚合"""
    print("\n─── GoT: Independent Analysis + Aggregation ───")

    # Phase 1: 独立分析每篇文档（并行 generation）
    analyses = []
    for i, doc in enumerate(docs):
        prompt = f"""Extract the key applications of AI mentioned in this text.
List only concrete applications, one per line.

Text: {doc}

Applications:
"""
        result = await llm(prompt)
        analyses.append(result.strip())
        print(f"  Analyzed doc {i+1}: {len(result.splitlines())} applications")

    # Phase 2: 按领域聚合
    all_apps_text = "\n\n".join(
        f"Doc {i+1}:\n{a}" for i, a in enumerate(analyses)
    )
    aggregate_prompt = f"""Below are AI application lists extracted from multiple documents.
Group them by domain (e.g., Healthcare, Education, Climate, etc.).
Remove exact duplicates. Resolve contradictions by choosing the more specific version.

{all_apps_text}

Grouped summary (domain by domain):
"""
    aggregated = await llm(aggregate_prompt)

    # Phase 3: Refinement
    refine_prompt = f"""Review and improve this summary.
Remove any remaining redundancy. Ensure every claim is specific and concrete.

{aggregated}

Improved summary:
"""
    return await llm(refine_prompt)


async def main():
    print("=" * 60)
    print("CoT vs ToT vs GoT: Multi-Document Merge Task")
    print("=" * 60)

    result_cot = await run_cot(documents)
    print(f"\nCoT Result ({len(result_cot)} chars):")
    print(result_cot[:500])

    result_tot = await run_tot(documents)
    print(f"\nToT Result ({len(result_tot)} chars):")
    print(result_tot[:500])

    result_got = await run_got(documents)
    print(f"\nGoT Result ({len(result_got)} chars):")
    print(result_got[:500])

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"CoT: {len(result_cot):4d} chars")
    print(f"ToT: {len(result_tot):4d} chars")
    print(f"GoT: {len(result_got):4d} chars")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 六、与 DeepEye / Heuristic Learning 的关联

### 6.1 ToT 与 DeepEye

DeepEye 的 **SupervisorAgent + WorkflowAgent** 架构本质上是 ToT 的一种特化实现：

| ToT 概念 | DeepEye 对应 |
|----------|-------------|
| Thought Generation | `WorkflowAgent` 生成工作流 JSON |
| State Evaluation | `TrialRunner` 执行并返回 exit_code/stdout/stderr |
| BFS 剪枝 | Supervisor 从多个工作流草案中选最优的 |
| Backtracking | 工作流执行失败后重新规划 |

### 6.2 GoT 与 Heuristic Learning

GoT 的 **Aggregation** 操作与 HL 的 **Compress History** 完全对应：

| GoT 概念 | HL 对应 |
|----------|---------|
| Generation | `agent.write_strategy()` |
| Aggregation | `agent.compress_history()` （多个成功 trial → 一个 memory 模板）|
| Refinement | `agent.absorb_feedback()` （读错误日志 → 改进代码）|
| Scoring | `TrialResult.score` （0.0 或 1.0） |
| GoO (静态操作图) | `HeuristicSystem.run_analysis()` 的固定循环 |
| GRS (动态推理状态) | `trials.jsonl` + `memory/*.md` |

### 6.3 潜在改进方向

可以把 GoT 的图拓扑直接引入 HLDAA：

1. **并行探索多个分析方向**：不只是一个策略 → 生成 3-5 个不同分析脚本并行执行
2. **聚合多个分析结果**：把 EDA、SQL、可视化三个独立分析的结果合并为一个综合报告
3. **循环精炼**：分析结果不满意 → refine → 再执行 → 直到收敛

这些改进只需要在 `system.py` 中扩展 `run_analysis()` 方法即可，不需要改变底层架构。

---

## 参考文献

1. Yao et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023. arXiv: 2305.10601
2. Besta et al. (2024). *Graph of Thoughts: Solving Elaborate Problems with Large Language Models.* AAAI 2024. arXiv: 2308.09687
3. Besta et al. (2024). *Demystifying Chains, Trees, and Graphs of Thoughts.* arXiv: 2401.14295
