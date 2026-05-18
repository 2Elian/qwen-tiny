# 基于qwen3-0.6b 训一个nl2sql模型

## lora原理及其变体

## 代码讲解

初识transformers的tokenizer、peft、trainer、DataCollatorForSeq2Seq

## lora加在哪些层?

这里需要根据任务复杂度来做，比如nl2sql单个任务的话，q k 层一般是够了。

如果不够可以按照以下超参进行AB实验：
- qk
- qkv
- qkvo
- qkvogud(g是路由权重 u是升维权重 d是降维权重)

## 数据集讲解

## base模型实验

- 输入：
```text
Database Schema
###
CREATE TABLE donations
(
        donationid TEXT not null primary key,
        projectid TEXT, --
        is_teacher_acct TEXT, -- `is teacher acct` description: whether donor is also a teacher value description: • f: false, it means this donor is not a teacher • t: true, it means this donor is a teacher
        foreign key (projectid) references projects(projectid),
);

CREATE TABLE projects
(
        projectid TEXT not null primary key,
        teacher_acctid TEXT, -- description: teacher's unique identifier (teacher that created a project)
        school_city TEXT, -- examples: `Chicago`| `school city` description: city where the school is located
);

###
Question:
How many teachers have made some type of donation for projects in Chicago?

Hint:
in Chicago refers to school_city = 'Chicago'; teachers refers to is_teacher_acct = 't'
```

- 模型输出：
```text
SELECT COUNT(*) FROM donations WHERE school_city = 'Chicago' AND is_teacher_acct = 't';
```

我们的目标是：让模型有思考能力，产生think，然后产生```sql```闭合标签，并且准确率提升。

### base模型的精度

```text
============================================================
  Evaluation Results (500 samples)
============================================================
  Exact Match Accuracy:  7/500 = 1.40%
  Avg Token F1:          0.6059

  Component Matching (Jaccard):
    select_cols         : 0.3306
    where_conditions    : 0.4610
    join_conditions     : 0.5067
    group_by            : 0.9053
    order_by            : 0.8400
    aggregates          : 0.8247
    distinct            : 0.7940
    limit               : 0.8700
```

---

## lora模型实验

### 1. qkvogud

<p align="center">
  <img src="../src/lora/output/lora-nl2sql/qkvogud/plots/dashboard.png" width="700">
</p>

我们以第二个epoch的checkpoint来推理

#### 问题1：推理时没有 `<think>` 标签

不仅没有think标签，并且还会输出一段乱码，比如：นำข้อมูลมาใช้

它的token_ids是：124822, 47839, 80614, 91200, 124029, 124207, 19841

但是我看了qwen3-06b-base的词表，<think>是有自己id的。

- 原因：base模型尽管有<think>标签，但其在下游未经过对齐训练，所以<think>会非常的不稳定。

如果想从base模型开始训这个think标签，可以先冷启动一批次数据，然后去rl。


#### 问题2：\`\`\`sql\`\`\`` 出现复读机现象

**现象**：模型生成完 SQL 后不断重复同一个 SQL 语句：

```text
Assistant:  ForCanBeConverted
Step-by-Step Process for Answering the Question:
...
6. **Assemble the SQL Query**
```sql
SELECT COUNT(T2.donationid) FROM projects AS T1 ...
````
 zwłaszc

```sql
SELECT COUNT(T2.donationid) FROM projects AS T1 ... ;
``` zwłaszc
```sql
SELECT COUNT(T2.donationid) FROM projects AS T1 ... ;
``` zwłaszc
... (无限重复)
```

**根因分析：EOS token 不匹配**

这是问题的**核心根因**。训练让模型输出 `<|im_end|>`（ID **151645**）来结束对话，但推理代码只识别 `<|endoftext|>`（ID **151643**）作为结束信号。

关键 token 对照：

| Token | ID | 用途 |
|-------|-----|------|
| `<|endoftext|>` | **151643** | 预训练文档结束符，即 `tokenizer.eos_token` |
| `<|im_start|>` | 151644 | chat template 的消息开始符 |
| `<|im_end|>` | **151645** | chat template 的消息结束符 |

**为什么 base model 没有这个问题？**

Base model（预训练权重）主要学习以 `<|endoftext|>`（151643）结束文本。而 LoRA 微调后，模型通过 chat template 格式的数据，学到了 `<|im_end|>`（151645）作为对话结束符。但推理代码的 EOS 列表没跟上这个变化。

- 解决方案：
在训练处理数据的时候，在答案最后添加<endoftext> 这个token


#### 问题3：输出不产生最后的\`\`\`sql\`\`\`

调整温度参数，就会好一些。但还是不太稳定。(还是base模型的问题，换成下游模型就非常的稳定了)

#### 实验结果

```text
============================================================
  Evaluation Results (500 samples)
============================================================
  Exact Match Accuracy:  58/500 = 11.60%
  Avg Token F1:          0.8752

  Component Matching (Jaccard):
    select_cols         : 0.5329
    where_conditions    : 0.6707
    join_conditions     : 0.8082
    group_by            : 0.9107
    order_by            : 0.9020
    aggregates          : 0.8890
    distinct            : 0.7980
    limit               : 0.9600
============================================================
```

### 2. qk


```text
============================================================
  Evaluation Results (500 samples)
============================================================
  Exact Match Accuracy:  41/500 = 8.20%
  Avg Token F1:          0.8517

  Component Matching (Jaccard):
    select_cols         : 0.4910
    where_conditions    : 0.6095
    join_conditions     : 0.7757
    group_by            : 0.9047
    order_by            : 0.8940
    aggregates          : 0.8670
    distinct            : 0.7980
    limit               : 0.9600
==================================
```