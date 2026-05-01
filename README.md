# elian-qwen-tiny: LLM post-training

2elian/qwen-tiny基于qwen3-0.6B-Base出发，探索attentioon residuals(探索这个的目的是为了：全面了解transformers框架如何更改原始模型的结构，并进行持续训练，学完它您将掌握transformers的基本训练流程并对LLM的架构更加清晰）。然后转向指令微调和Agent训练

包括：
- 指令微调：训出LLM的基本chat能力，选取了一个特定领域的数据集+指令遵循数据集，目标是超越qwen3-0.6b的性能
- grpo探索
- agent-sft
- agent-rlvr + 冷启训练
- agent-opd
- skill-pod
- 在agent-sft的checkpoint上训embedding model和rerank model

## 文档

- [attention residual教程与日志](/docs/continue_pretrained_attnRes.md)

## 后续会探索的
- mtp训练
- mHC
- 稀疏注意力，比如：csa、hca
- search-r1训练
- generate reward model


## 训练框架：

- transformers：用于简单的task训练
- llamafactory-改进：基于llamafactory底层源码进行训练，而不是使用前端，最大限度的理解微调代码，便于后续小伙伴们在训推框架上作二次开发
- verl and slime

## 致谢
- 感谢[gouzigouzi](https://github.com/gouzigouzi/attention-residuals-reproduction)提供的attention-residuals代码