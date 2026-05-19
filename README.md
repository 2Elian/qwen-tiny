# elian-qwen-tiny: LLM post-training

## 为什么写了这个项目？

愿景是：让每一个学习LLM和Agent的本硕博同学少走弯路，qwen-tiny提供从LLM原理(致力于讲清楚每一个原理细节)到后训练(致力于教会同学们如何魔改模型，并掌握transformers、llamafactory、verl源码)再到agent训练的完整pipeline教程。

预计的教学形式为视频为主，文档为辅
<p align="center">
  <img src="docs/images/qwen-tiny-development-plan.png" width="700">
</p>

## 文档

- [llm的基本原理](/docs/llm_Basic%20principles.md)在这里您将详细的学习到transformer理论，每个部分揉碎了讲解，并附上transformers的model模块的学习。

- [continue pre-train: attention residual教程与日志 base qwen-0.6B-Base](/docs/continue_pretrained_attnRes.md)在这里您将学习到残差连接的本质与attention residual理论，并学习到如何使用transformers更改模型结构。
- [elf: 扩散语言模型实验](/docs/continue_pretrained_attnRes.md)穿插实验：ELF: 嵌入语言流, 预训练代码跑通，并看看如何进行数学的RL训练

- [lora 微调 base qwen-0.6B-instruct](/docs/qwen3_06b_lora_cot.md)在这里您将学习到如何使用transformers进行lora微调，模型采用qwen-0.6-base，数据采用nl2sql
- [全参指令微调 base qwen-0.6B-instruct](/docs/continue_pretrained_attnRes.md)在这里您将学习到如何使用transformers+并行策略(zero-3、fsdp、tp、pp等)进行小模型初步的微调训练，以及如何使用llamafactory后端进行高效训练。在这里您还将学习各个sft框架的优劣，包含：transformers、llamafactroy、ms-swift、xtuner，模型采用qwen-0.6-base，数据采用deepMath103k. 主要是做冷启动训练，使用15%的数据做冷启训练，为了下一阶段的RLVR训练做准备。评估方式为GPQA-Diamond和AIME-2024，对比Base模型。
- [rlvr-aime+冷启动 base qwen-0.6B-instruct](/docs/continue_pretrained_attnRes.md)在这里您将初步的学习到verl与slime源码，进行强化学习的训练，使用deepMath数据集，进行GRPO的训练。并使用GPQA-Diamond和AIME-2024作为测试集进行评估。训练方式为：冷启+RL和直接RL两种方式。
- [OPD训练](/docs/continue_pretrained_attnRes.md)在这里我们使用rl出来的数学模型+nl2sql模型，两个专家模型来在线蒸馏base模型，让base模型拥有二者能力。并且我们将会做AB实验，即将两个专家模型合并起来(两个数据源训一个模型)，对比蒸馏效果。
- [search-r1](/docs/continue_pretrained_attnRes.md)这是agent一切训练的开始, 基于verl
- [agent sft base qwen-0.6B-instruct](/docs/continue_pretrained_attnRes.md)在这里您将系统的学习到llafactroy后端源码，更重要的是您将系统的阅读agent几个比较重要的论文思想，并带您遨游claude code、openclaw、hermes的源码，让您看看当代agent架构是如何设计的，以及其最重要的memroy是如何做的。最最最重要的是您将学习到如何抽象出一个自己可复用的agent框架，用于您的业务系统，最后我将带您构建一个agent系统，并收集agent-sft训练数据，进行agent-sft(基于llamafactory魔改)
- [agent rl 包括: general/search/tool call base qwen-0.6B-instruct](/docs/continue_pretrained_attnRes.md)在这里您将系统的学习verl源码和Uni-Agent源码，进行更高难度的agentic-rl训练。
- [elian是如何做llm与agent相关实验发表A类论文的](/docs/continue_pretrained_attnRes.md)

## 后续会探索的
- mtp训练
- mHC
- 稀疏注意力，比如：csa、hca
- generate reward model


## 训练框架：

- transformers：用于简单的task训练
- llamafactory-改进：基于llamafactory底层源码进行训练，而不是使用前端，最大限度的理解微调代码，便于后续小伙伴们在训推框架上作二次开发
- verl and slime

## 致谢
- 感谢[gouzigouzi](https://github.com/gouzigouzi/attention-residuals-reproduction)提供的attention-residuals代码