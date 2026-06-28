# AI 教材深度优化项目

> 一本关于 LLM、Agent、World Model 互补与集成的技术教材，经过认知科学驱动的深度重构——不只是翻译或改写，而是对每个技术决策的底层逻辑进行逐章解剖。

## 项目概况

这本教材覆盖了从 Transformer 架构到具身智能的完整技术栈，共 15 章。每个章节按照一套严格的**优化标准手册**进行深度重构，核心方法论是：**还原历史约束、展开方案空间、标注证据强度、嵌入批判性视角**。

这不是一本"技术百科全书"，而是一套**分析框架**。读完本书后，当你看论文时，你将能够：

- 判断一个技术宣称背后有多少是证据支持、多少是研究者乐观投射
- 识别"幸存者偏差"——为什么当前主流方案赢了，不一定是它更好
- 用反事实推演评估"如果当年选了另一条路会怎样"

## 优化标准

优化过程遵循 [optimization_standard.md](revised_textbook_optimized/optimization_standard.md)，核心原则包括：

| 维度 | 说明 |
|------|------|
| **认知路径** | 先探索问题复杂性 → 再收敛到方案 → 最后给出带证据强度标注的结论 |
| **证据强度** | 每个核心命题标注：已证明 / 强实验证据 / 合理推断 / 开放问题 / 作者推测 |
| **方案空间** | 每个技术决策必须列出 2-3 个候选方案 + 反事实推演（"如果选了另一个会怎样"） |
| **认知偏差** | 每章主动审视研究者可能陷入的认知偏差（确认偏差、工具偏差、幸存者偏差等） |
| **教学逻辑** | 先直觉后形式化、错误驱动学习、连接已有知识 |

## 目录

| 章节 | 标题 | 优化状态 |
|------|------|----------|
| 第 1 章 | 从语言到世界——LLM、Agent、World Model 的互补与集成 | ✅ 已优化 |
| 第 2 章 | Transformer 的演化与极限（从 BERT/GPT 到 Mamba/MoE） | ✅ 已优化 |
| 第 3 章 | 表征学习——文本、视觉、世界动态的共享数学结构 | ✅ 已优化 |
| 第 4 章 | 预训练与 Scaling Laws（数据、计算、参数的最优配置） | ✅ 已优化 |
| 第 5 章 | 推理与思维链（CoT）——从 prompt 工程到长程推理的涌现 | ✅ 已优化 |
| 第 6 章 | 对齐与安全（RLHF、DPO、Constitutional AI、可解释性） | ✅ 已优化 |
| 第 7 章 | 多模态与工具化（视觉语言模型、Function Calling、Code Generation） | ✅ 已优化 |
| 第 8 章 | 从 ReAct 到 AutoGen——Agent 架构的范式演进 | ✅ 已优化 |
| 第 9 章 | 工具使用、记忆与规划（Toolformer、RAG、Long-term Memory） | ✅ 已优化 |
| 第 10 章 | 多智能体协作与社会模拟 | ✅ 已优化 |
| 第 11 章 | Agent 的评估与对齐（AgentBench、HarmBench、Agent Safety） | ✅ 已优化 |
| 第 12 章 | 基于模型的强化学习（MBRL）——从 Dreamer 到 MuZero | ✅ 已优化 |
| 第 13 章 | JEPA 与预测性表征学习：LeCun 的自主机器智能蓝图 | ✅ 已优化 |
| 第 14 章 | World Model × LLM × Agent 的融合 | ✅ 已优化 |
| 第 15 章 | 前沿与未来：具身智能、科学发现与 AI for Robotics | ✅ 已优化 |

## 仓库结构

```
study-materials/
├── README.md                          # 本文件
├── 文档/                              # 原始教材章节（优化前版本）
│   ├── Chapter_01.md ~ Chapter_15.md
├── revised_textbook_optimized/        # 深度优化后的章节
│   ├── Chapter_01_Optimized.md ~ Chapter_15_Optimized.md
│   ├── optimization_standard.md       # 优化标准手册（方法论核心）
│   ├── build_ch11.py                  # 第 11 章构建脚本
│   ├── build_p1.py                    # 第一部分构建脚本
│   └── ch09_optimization_artifact_20260628.md  # 第 9 章优化产物记录
└── .gitignore
```

## 技术覆盖范围

- **LLM 核心架构**：Transformer、BERT、GPT 系列、Mamba、MoE、Attention 机制
- **训练与对齐**：预训练、Scaling Laws、RLHF、DPO、Constitutional AI
- **Agent 系统**：ReAct、AutoGen、Toolformer、RAG、多智能体协作、Agent 评估与安全
- **世界模型**：MBRL、Dreamer、MuZero、JEPA、预测性表征学习
- **前沿议题**：具身智能、VLA、科学发现、AI for Robotics

## 时间窗口

本书内容有效窗口截至 2025 年 6 月。对于快速演进的领域（如 Agent 框架、推理模型），建议结合最新论文阅读。

## 许可

本仓库为个人学习与研究成果，保留所有权利。
