# 第9章 工具使用、记忆与规划（Toolformer、RAG、Long-term Memory）

> **本章动机（养老机器人视角）**
> 机器人需要记住"王奶奶对花生过敏"，并在几个月后的对话中准确检索——即使中间经历了数百次其他对话。
> RAG加长期记忆，就是机器人的"医疗档案系统"。
> Toolformer解决了另一个问题：如何让模型知道"什么时候该查资料、什么时候靠自己"。
>
> 但更深的问题是：**为什么语言模型需要外部工具？** 一个拥有1750亿参数的模型，已经"记住"了互联网上数万亿token的知识，为什么它在回答"2023年法国总统是谁"时会出错？答案不在于参数不够多，而在于**知识的时效性、精确性和可验证性**——这三个维度是参数化模型在原理上难以同时满足的。本章不是简单地介绍工具调用和RAG的工程实现，而是要回答：**在什么条件下，"不知道但知道去哪里查"比"记住一切"更优？**

---

## 核心思想

语言模型通过显式工具调用、外部检索与结构化记忆存储，将自身从静态参数化的知识库转变为可扩展、有状态、能与外部世界持续交互的智能系统——其本质不是让模型记住更多，而是让模型学会在何时、以何种方式、向哪些外部系统请求信息。

这一转变的深层驱动力是**认知科学中"扩展心智"（extended mind）假说在工程上的体现**：一个智能系统的能力不仅取决于其内部表征，更取决于它能否有效地利用外部资源。人类的计算能力不取决于能否心算乘法，而取决于能否找到并使用计算器。类似地，LLM的价值不在于它记住了多少知识，而在于它能否判断"这个问题的答案我应该自己回答还是去查"——这种**认知调节能力（metacognitive monitoring）**是工具使用的认知基础。

**证据强度标注**：本章核心命题"工具调用+RAG+长期记忆构成现代Agent的认知基础设施"是**[强实验证据]**——多个独立工业部署验证了这一架构的有效性。但"这是最优架构"是**[开放问题]**——端到端训练的模型是否能替代显式工具调用+检索+记忆的松耦合架构，目前尚无定论。


---

## 历史脉络：从封闭知识到开放系统——不是进步，而是被迫转向

### 问题本质：参数化知识的三个原理性缺陷

在GPT-3 [Brown et al., 2020] 之前，语言模型的知识完全封闭于训练语料。这并非设计者的疏忽，而是预训练范式的直接后果：模型的知识是其训练数据的统计压缩，一旦训练完成，知识就"冻结"在参数中。这种封闭性导致了三个原理性缺陷：

1. **时效性缺陷**：模型不知道训练截止日期之后发生的事件。一个在2021年6月训练完成的模型不知道2022年的俄乌冲突——这不是"信息缺失"，而是参数化表征的时间不变性导致的结构性盲区。**[已证明]**
2. **精确性缺陷**：模型对稀有事实的回忆精度低。研究表明，GPT-3对训练语料中出现少于5次的事实，回忆准确率低于30%。这是神经网络"频率驱动的学习"机制导致的——低频模式在梯度更新中被高频模式淹没。**[强实验证据]**
3. **可验证性缺陷**：模型无法提供知识的来源。即使它回答正确，你无法知道这个答案是从哪个文档中学到的、置信度多高、是否有矛盾证据。这是参数化表征的固有局限——知识被"揉碎"在数十亿参数中，丢失了来源元信息。**[已证明]**

### 历史约束：2020年的世界是什么样的？

要理解为什么工具调用和RAG在2020-2023年间兴起，必须还原当时的技术约束空间：

- **计算约束**：2020年，GPT-3的训练成本估计为1200万美元。频繁重新训练以更新知识是不现实的——每次更新都需要在数千亿token上重新训练。继续预训练（continual pre-training）虽然存在，但面临**灾难性遗忘**（catastrophic forgetting）——新知识的学习会干扰旧知识的保持。
- **数据约束**：2020年最大的预训练语料（如The Pile、C4）规模在数百GB到数TB之间。但知识更新的速度远超训练周期——每天都有新的新闻、科学发现、软件版本发布。任何静态语料都无法捕捉实时信息。
- **理论约束**：2020年，"参数化知识 vs 检索知识"的理论框架尚未建立。研究者直觉上知道"在prompt中拼接外部文本"可以提供上下文信息，但缺乏系统的理论指导来优化这一过程。in-context learning的机制理解（如Olsson et al.的"归纳头"假设）要到2022年才被提出。
- **社区约束**：2020年的NLP社区仍然以"训练更大模型"为主导范式。Scaling Laws（Kaplan et al., 2020）的发表强化了"大就是好"的社区信念，检索增强被视为"不走正道"的工程技巧。REALM [Guu et al., 2020] 在发表时并未获得广泛关注——直到2022年ChatGPT暴露了幻觉问题，社区才开始认真对待检索增强。

### 方案空间：在当时约束下，有哪些候选路径？

**路径A：持续预训练（Continual Pre-training）**
- 核心思想：定期在新数据上继续训练模型，让参数吸收新知识
- 在当时约束下的优势：不需要修改推理架构；新知识直接融入模型参数
- 在当时约束下的劣势：训练成本极高；灾难性遗忘导致旧知识退化；无法实现"分钟级"知识更新
- **反事实推演**：如果持续预训练的成本能降低到每小时数美元，我们可能不需要RAG——模型可以直接"记住"最新知识。但短期（1-2年）内，这种成本下降不太可能发生；长期（5-10年）来看，即使成本下降，灾难性遗忘仍然是参数化方法的原理性障碍。今天的证据（2025年中）表明，持续预训练在工业实践中被用于"季度级"知识更新，但"分钟级"更新仍然依赖RAG。**[强实验证据]**

**路径B：Prompt拼接（In-Context Injection）**
- 核心思想：在推理时将外部知识作为上下文拼接到prompt中
- 在当时约束下的优势：零训练成本；立即可用；知识更新只需更新外部文档
- 在当时约束下的劣势：模型不会自主判断何时需要外部信息；拼接什么、拼接多少需要人工规则决定
- **关键认知偏差**：早期实践者可能犯了**工具偏差**——把prompt拼接当作万能锤子。但Transformer的"Lost in the Middle"现象（Liu et al., 2023）表明，上下文中的信息并非被平等对待——中间位置的信息检索率显著低于首尾。这意味着"拼接更多"并不等于"记住更多"。

**路径C：预训练阶段检索融合（REALM, 2020）**
- 核心思想：在预训练阶段同时优化语言模型与检索器
- 在当时约束下的优势：检索行为被内化到模型参数中；检索器与生成器联合优化
- 在当时约束下的劣势：检索器与生成器紧耦合，知识库更新需要重新训练整个系统
- **反事实推演**：如果REALM的紧耦合方案被证明在性能上显著优于解耦方案，今天的RAG生态可能完全是另一个样子。但REALM的耦合设计在工程上造成了严重的维护负担，这促使社区选择了Lewis et al. (2020)的解耦RAG方案。**这个选择不是"性能最优"驱动的，而是"工程可维护性"驱动的。**

**路径D：自监督工具调用学习（Toolformer, 2023）**
- 核心思想：在继续预训练阶段，让模型在未标注语料中自动学习何时插入API调用
- 在当时约束下的优势：工具调用语法被内化到参数中；自监督筛选机制不需要人工标注
- 在当时约束下的劣势：训练计算开销巨大；只支持闭集工具；无法处理多步工具链组合
- **为什么这条路径最终被选中？** 因为Toolformer证明了一个关键假设：**模型可以在预训练阶段学会"何时需要外部信息"的认知调节能力**。**[强实验证据]**

### 选择逻辑：为什么是RAG + Toolformer + 长期记忆的三角组合？

在2023年的约束下，这三条路线各自解决了参数化模型的一个缺陷，但没有任何一条能单独解决全部：

| 路线 | 解决的缺陷 | 未解决的缺陷 |
|------|-----------|-------------|
| Toolformer | 可验证性（显式工具调用提供来源） | 时效性（工具集固定）；精确性（单步调用） |
| RAG | 时效性（向量数据库可实时更新）；精确性 | 可验证性（检索对模型透明） |
| 长期记忆 | 跨会话状态持久化 | 三层缺陷均未直接解决 |

**选择逻辑**：在2023年的约束下，这三条路线的组合在"知识更新频率×工具多样性×状态持续性"三个维度上提供了最优的trade-off。**这个选择并非不可避免**，而是在约束空间中的理性优化。

### 认知偏差警示：RAG的流行是否被过度归因于"技术优越性"？

RAG在2023-2025年的流行有一个常被忽视的**生态因素**：向量数据库产业的商业化推动了RAG的标准化。Pinecone、Milvus、Weaviate、Chroma等公司在2022-2024年间获得了大量风险投资。一个研究者选择RAG，不仅是因为它在技术上合适，还因为生态工具链的支持。这种**生态锁定效应**意味着：即使存在技术上更优的替代方案，它也可能因为缺乏商业生态支持而难以被采用。**[作者推测]**


---

## 9.1 三条路线的交汇：从独立演化到Agent认知循环

### 路线一：工具调用（Tool Use）

GPT-3的1750亿参数在问答任务上表现优异，但面对"2023年法国总统是谁"或"计算14523的平方根"这类问题时，要么生成训练数据中的过时信息，要么在算术上出错。前者是**时效性缺陷**，后者是**精确性缺陷**。

**错误驱动学习：prompt工程的失败**

```python
# 常见错误：期望通过few-shot prompt让模型学会工具调用
prompt = """
Q: 2+2等于几？
Thought: 我需要用计算器。
Action: calculator(2+2)
Observation: 4
Answer: 4

Q: 5*8等于几？
Thought: 我需要用计算器。
Action: calculator(5*8)
Observation: 40
Answer: 40

Q: 法国2023年的GDP是多少？
"""
# 错误分析：模型看到前两个示例后，倾向于机械模仿"Thought: 我需要用计算器"
# 即使第三个问题需要搜索而非计算器。这不是模型"不聪明"，而是
# in-context learning的本质——它学习的是prompt中的模式，而非"何时使用工具"的判断力。
# [注意] few-shot prompt教会模型的是"格式模仿"，不是"认知调节"。
# [关键局限] 没有预训练阶段的对齐，模型无法区分"该用工具"和"该自己回答"的时机。
```

Toolformer [Schick et al., 2023] 从根本上改变了这一状况：它不是在推理时通过prompt教模型使用工具，而是在继续预训练阶段，让模型在大量未标注文本中自动学习何时插入`<API>`调用、如何解析返回值。

**为什么自监督有效？** 直觉上理解：如果在一个句子中插入"QA(爱因斯坦出生年份)→1879"后，模型预测后续文本的困惑度下降了，说明这个API调用提供了**有用的信息**——模型"意识到"自己需要这个信息才能更好地预测下文。这类似于人类阅读时遇到不确定的事实，会主动去查阅资料——查阅后理解更顺畅。Toolformer的自监督机制在计算层面复现了这种认知调节过程。**[合理推断]**

### 路线二：检索增强（Retrieval-Augmented Generation）

与Toolformer的"主动调用"不同，RAG走的是"被动检索"路线。REALM [Guu et al., 2020] 是这一方向的开创性工作，它在预训练阶段同时优化语言模型与检索器。然而，REALM将检索器与生成器紧耦合，导致知识库更新时需要重新训练整个系统。

2020年底，Lewis et al. [2020] 提出的RAG框架将检索与生成解耦：检索器与生成器分别训练，在推理阶段通过向量检索动态获取上下文。这一架构使知识库可以独立更新。

**选择逻辑**：在2020年的约束下，RAG的解耦设计在"知识更新频率"维度上最优，但牺牲了"检索-生成协调性"。工业实践选择了前者，因为**知识更新频率在产品需求中是硬约束，而检索-生成协调性可以通过重排序等后处理手段部分弥补**。

### 路线三：长期记忆（Long-term Memory）

上述两条路线解决了"知识从哪里获取"的问题，但未解决"知识如何持久化与管理"的问题。2023年开始，研究者将认知心理学中的记忆层次模型映射到LLM Agent架构：感知记忆对应实时环境输入，工作记忆对应LLM的上下文窗口，长期记忆分为语义记忆（向量数据库）与情景记忆（事件时间序列）。Du [2026] 的综述指出当前Agent系统的瓶颈已从"模型能力"转移到"记忆管理机制"。

**认知科学类比的局限**：将人类记忆的三层模型映射到Agent架构是一个**[合理推断]**，但不是**[已证明]**的等价关系。人类的工作记忆容量约为4±1个组块，而LLM的上下文窗口可以容纳数万token——量级差异意味着两者的信息处理策略可能根本不同。如果人类工作记忆的容量限制是"特性而非缺陷"（迫使信息被压缩和抽象化），那么LLM的超大上下文窗口可能反而**阻碍了有效抽象**。**[开放问题]**

三条路线的交汇点出现在ReAct [Yao et al., 2022] 与后续的工具增强Agent框架中，形成了完整的Agent认知循环。


---

## 9.2 Toolformer：在预训练阶段学习工具调用

### 问题动机：为什么模型需要学会"知道自己不知道"？

在Toolformer之前，LLM使用工具的标准方法是**prompt工程**。但这种方法存在一个根本问题：**模型缺乏元认知能力（metacognition）**——它不知道自己什么时候"不知道"，自然也不知道什么时候该调用工具。

直觉上理解：一个学生如果在考试时不知道自己哪些题不会做，他就不会主动去翻书。Toolformer的思路是：在"学习阶段"（预训练）就让学生练习"遇到不会的题就翻书"的习惯，这样在"考试时"（推理时）这个习惯就自动生效了。

### 核心方法论：自监督API调用合成

Toolformer的核心假设是：**如果语言模型在预训练阶段见过大量`<API>`标签与返回值的交互模式，它就能在推理时自主生成格式正确的工具调用。** **[强实验证据]**

**数据构造**。Toolformer选取了计算器、问答系统（Q&A）、维基百科搜索、机器翻译、日历五个API。对于每一个预训练样本中的每一个位置，算法执行以下步骤：
1. 在该位置后插入一个候选API调用；
2. 执行该调用，获取返回值；
3. 将调用与返回值插入原文，形成增强文本；
4. 用一个辅助语言模型计算增强文本的加权交叉熵损失，并与原始文本对比；
5. 只有当损失下降超过阈值时，才保留该API调用作为训练数据。

**直觉解释**：如果一个API调用"有用"，那么插入它之后，后续文本应该"更好预测"（困惑度更低）。例如，在"爱因斯坦出生于1879年"后插入"Calculator(1879)"不会降低困惑度，因此被丢弃；而插入"QA(爱因斯坦出生年份)"并返回"1879"则会降低困惑度，被保留。最终，约5%-10%的预训练token位置被标注了API调用。

**训练目标**。Toolformer采用标准的自回归语言建模目标，但词汇表扩展了`<API>`、`</API>`等控制token。模型在训练阶段看到API调用位置之前的上下文，但API调用本身与返回值作为输入的一部分被掩码处理——这种设计确保了模型不会产生暴露偏差（exposure bias）。

**推理行为**。在推理时，Toolformer模型生成文本直到遇到`<API>`标签。此时解码暂停，提取API调用参数，执行外部工具，将返回值重新插入解码上下文，然后继续生成。这一流程与ReAct循环高度相似，区别在于Toolformer的工具调用语法已内化到模型参数中。

### 错误驱动学习：Toolformer的常见误解

**常见错误1：认为Toolformer教会了模型"理解"工具**

```python
# 错误理解：Toolformer让模型理解了计算器的工作原理
# 实际上：Toolformer让模型学会了"在特定模式下插入<API>标签"的统计规律
# 模型不知道"计算器"是什么，它只是学会了"当文本中出现数字运算模式时，
# 插入<API>calculator(...)标签通常会降低困惑度"
# [注意] 这是统计学习，不是符号理解
# [关键局限] 模型对工具的"理解"停留在模式匹配层面，缺乏因果推理
```

**常见错误2：认为Toolformer的API调用是"自主决策"**

```python
# 错误理解：模型在推理时"决定"是否调用工具
# 实际上：模型只是在生成下一个token时，根据上下文概率生成<API>标签
# 这个"决策"是基于训练时学到的统计模式，而非显式的成本-收益分析
# 当训练分布与推理分布不匹配时，模型可能在不该调用时调用，或在该调用时不调用
# [注意] Toolformer的"决策"是隐式的统计推理，不是显式的元认知
```

### 局限性分析

Toolformer的评估暴露了三个关键局限：

1. **单步调用**：模型无法处理需要多步工具链组合的任务。每一个`<API>`调用独立决策，没有跨调用的状态传递机制。**[已证明]**
2. **工具闭集**：训练时只见过五个API，无法泛化到未见过的新工具。后续工作ToolLLM [Qin et al., 2024] 通过16,000+真实API的微调解决了这一问题。**[强实验证据]**
3. **无状态性**：Toolformer的工具调用是瞬时的，不维护跨会话的长期记忆。**[已证明]**

### 反事实推演：如果Toolformer没有出现？

假设2023年Toolformer未被提出，工具调用完全依赖prompt工程：

- **短期（1-2年）**：GPT-4级别的模型通过更强的in-context learning能力，可以在few-shot prompt下实现接近Toolformer的工具调用精度。但每次推理都需要在prompt中提供工具调用示例，增加了token消耗和延迟。
- **长期（5-10年）**：如果模型规模持续增大，in-context learning的能力可能达到"不需要预训练阶段内化"的水平——模型在推理时就能从few-shot示例中学会新工具的使用。但这需要模型具备更强的**元认知能力**，而不仅仅是更大的参数规模。**[开放问题]**

这个反事实揭示了一个关键点：**Toolformer的"创新"不在于工具调用本身，而在于将工具使用的判断力从推理时（prompt层面）转移到训练时（参数层面）**。这个转移是否必要，取决于推理时prompt工程的成本和效果是否可接受。

### 认知偏差警示：预训练内化的"锁定效应"

Toolformer将工具调用语法内化到参数中，这带来一个隐含的风险：**工具集锁定**。模型在预训练阶段只见过五个API，它的工具调用行为被"锁定"在这五个API的模式上。即使后续通过prompt提供新工具的描述，模型也可能因为预训练分布的惯性而无法有效使用新工具。这是一种**路径依赖**——早期的设计决策（选择哪些API）会限制后续的灵活性。

这与第1章讨论的"统一架构 vs 松耦合集成"的trade-off一脉相承：Toolformer选择了"紧耦合"（工具调用内化到参数），牺牲了灵活性；而prompt工程选择了"松耦合"（工具调用在推理时定义），牺牲了精度。**[合理推断]**


---

## 9.3 RAG：检索增强生成（向量数据库、Chunk策略、重排序）

### 问题动机：为什么不能直接把所有文档拼到prompt里？

当知识库规模较小时（如10篇文档），直接将所有文档拼接到prompt中是最简单的方案。但当知识库增长到百万级文档时，这种"暴力拼接"策略在三个维度上崩溃：

1. **上下文长度**：百万文档×平均500 token = 5亿token，远超任何模型的上下文窗口。
2. **信息稀释**：即使模型能处理5亿token，绝大部分文档与当前查询无关，它们会"稀释"真正相关信息的注意力权重。这就是"Lost in the Middle"问题的规模放大版。
3. **计算成本**：Transformer的注意力计算复杂度是O(n²)，5亿token的推理成本在天文数字级别。

因此，RAG的核心问题是：**如何从大规模知识库中高效地检索最相关的少量文档，并将其注入生成过程？** 这是一个信息检索问题与生成问题的交叉。

### 核心架构：三组件协同

RAG系统的核心架构包含三个组件：文档编码器（将知识库转换为向量）、检索器（根据查询向量召回相关文档片段）、生成器（基于检索结果生成回答）。这一架构的工业落地依赖于向量数据库、文本分块（chunking）策略与重排序（reranking）的协同优化。

#### 向量数据库与Embedding模型

**先建立直觉**：想象一个巨大的图书馆，每本书都被压缩成一个"坐标点"放在高维空间中。当你提出一个查询时，查询也被转换为坐标点，系统找到离它最近的几个点——这就是向量检索的本质。关键问题是：这个"坐标系"是否足够好，能让语义相关的文档在空间中靠近？

早期RAG使用DPR（Dense Passage Retrieval）的BERT-base编码器，将文档与查询编码为768维向量。2023年后，sentence-transformers库中的`all-MiniLM-L6-v2`（384维，轻量）与`all-mpnet-base-v2`（768维，高精度）成为工业标准。对于中文场景，则常用`BAAI/bge-large-zh`或`BAAI/bge-m3`。

向量数据库的选择涉及精度与延迟的权衡：FAISS（Facebook AI Similarity Search）的IVF-PQ索引在百万级文档上提供毫秒级查询，Milvus支持分布式部署与混合检索（向量+关键词），而Chroma与Qdrant则更适合轻量级应用。

**隐性假设与潜在问题**：Embedding模型的一个隐性假设是——文档与查询应处于同一语义空间。如果文档是技术手册而查询是口语化问题，二者的向量分布可能不一致，导致检索失败。这被称为**语义鸿沟（semantic gap）**问题。解决方案包括使用领域特定数据微调embedding模型，或采用查询扩展（query expansion）技术——用LLM生成多个查询变体，分别检索后合并结果。

#### Chunk策略：RAG中最具工程敏感性的决策

**为什么需要分块？** Embedding模型的输入长度有限（如`all-MiniLM-L6-v2`最大256 token），而原始文档可能长达数千token。必须将文档切分为可嵌入的片段。但怎么切不是简单的技术问题——它直接影响检索精度。

**直觉理解**：想象你在一本烹饪书中查找"红烧肉用什么酱油"。如果烹饪书按整章切分（每章50页），你找到的"章节"可能包含红烧肉的完整做法，但也包含红烧鱼、红烧排骨的做法——信息被稀释了。如果按句子切分，你找到的"句子"可能只有"加入适量酱油"，但缺少上下文（是什么菜？多少量？）。Chunk策略就是在"太多上下文"和"太少上下文"之间找平衡。

常见策略包括：
- **固定长度滑动窗口**：以固定token数（如512）切分，重叠率20%-50%。实现简单，但会在句子边界处切断语义单元。
- **递归字符分割**：先按段落分割，若段落过长则按句子分割，若句子仍过长则按固定长度分割。LangChain的`RecursiveCharacterTextSplitter`实现了这一策略。
- **语义分割**：用模型检测主题转换点，在语义边界处切分。计算成本更高，但检索精度通常提升5%-15%。**[强实验证据]**
- **代理增强分割**：用LLM直接读取文档并决定最佳切分点。成本最高，适用于高质量要求的场景。

Chunk大小与embedding模型的最大输入长度直接相关。若使用`all-MiniLM-L6-v2`（最大256 token），chunk应控制在256 token以内；若使用支持8192 token的模型（如BGE-M3），则可以采用较大的chunk以保留更多上下文。

一个常被忽视的问题是：chunk的元数据（metadata）同样重要。在chunk向量中附加文档标题、章节名、时间戳等信息，能显著提升过滤与排序效果。

#### 重排序（Reranking）：从"语义相关"到"信息相关"

**问题动机**：向量检索基于余弦相似度或内积，其本质是对比查询向量与文档向量的全局相似性。这种全局度量可能召回语义相关但信息价值低的文档。例如，查询"Transformer的注意力机制复杂度"可能召回一篇介绍Transformer整体架构的综述，而非专门讨论复杂度优化的论文——前者"语义相关"但不包含"信息相关"的答案。

重排序器（如ColBERT、BGE-Reranker、Cross-Encoder）在检索阶段后执行精排：将查询与每个候选文档拼接，通过一个交叉注意力模型计算相关性分数。ColBERT采用延迟交互（late interaction）机制，分别编码查询与文档，然后在token级别计算最大相似度，兼顾效率与精度。实验表明，在Top-5检索结果上添加重排序，可将问答准确率提升8%-20%，而延迟增加通常不超过50ms。**[强实验证据]**

**双编码器 vs 交叉编码器的trade-off**：

| 维度 | 双编码器（Bi-Encoder） | 交叉编码器（Cross-Encoder） |
|------|----------------------|---------------------------|
| 计算方式 | 分别编码query和doc，计算点积 | 拼接query+doc，输入Transformer |
| 速度 | 快（可预计算文档向量） | 慢（每对query-doc需独立计算） |
| 精度 | 中等 | 高 |
| 适用阶段 | 初检（从百万文档召回Top-K） | 重排（从Top-K精排Top-N） |

**选择逻辑**：在2023年的约束下，"双编码器初检 + 交叉编码器重排"的两阶段架构在"延迟×精度"维度上提供了最优trade-off。这种两阶段架构与计算机体系结构中的"TLB + 页表"两级地址翻译思路一脉相承——快速但粗略的第一级过滤大部分候选，精确但缓慢的第二级精排最终结果。**[合理推断]**

### RAG与Toolformer的互补性

RAG将外部知识以隐式方式注入模型（模型看到的是拼接后的文本），适合处理事实性查询；Toolformer生成显式的API调用，适合处理计算、翻译、实时数据获取等任务。现代Agent系统（如AutoGen [Wu et al., 2023]）通常同时集成二者：先通过RAG检索领域知识，再通过工具调用获取实时信息，最终将多源信息整合到统一的推理链中。

### 批判性分析：RAG的实证盲区

1. **内在局限性**：RAG的检索质量受限于embedding模型的语义表达能力。如果embedding模型无法将查询与文档映射到相近的向量空间（如跨语言、跨领域检索），RAG就无法工作。这是**原理性局限**，不是更多数据能解决的。**[已证明]**

2. **实证盲区**：RAG的评估通常在标准问答数据集（如Natural Questions、TriviaQA）上进行，这些数据集的查询通常是"事实性问题"（如"XXX的创始人是谁"）。但实际应用中，用户的查询往往是**推理性问题**（如"为什么X比Y更适合场景Z"）或**多跳问题**（如"X公司的CEO之前在哪家公司工作"）。推理性查询的检索难度远高于事实性查询——因为推理需要的信息可能分散在多个不相关的文档中，而当前的chunk策略倾向于将每个chunk独立检索。**[开放问题]**

3. **替代路径**：结构化知识图谱（Knowledge Graph）在多跳推理上可能优于向量检索——因为图结构天然支持多跳遍历。但知识图谱的构建成本高，且无法处理非结构化文本。2025年的趋势是**混合检索**：向量检索 + 关键词检索（BM25） + 知识图谱，三者互补。**[合理推断]**

4. **认知偏差警示**：RAG研究者可能存在**确认偏差**——在标准基准上验证RAG的有效性后，倾向于在更多标准基准上测试，而非挑战RAG的边界条件（如推理性查询、对抗性查询）。这导致RAG的"成功率"被高估。


---

## 9.4 长期记忆：感知记忆、工作记忆、长期记忆的架构映射

### 问题动机：为什么Agent需要"记忆"？上下文窗口不是够了吗？

直觉上，一个拥有128k token上下文窗口的LLM Agent似乎不需要"记忆系统"——所有历史对话都可以塞进上下文。但这个直觉忽略了三个关键限制：

1. **容量限制**：即使128k token窗口，也无法容纳一本书或整个代码库。对于需要长期积累知识的Agent（如法律顾问、持续学习的个人助手），上下文窗口仍然太小。
2. **成本限制**：上下文长度与推理计算量呈二次或线性关系（取决于注意力实现），长上下文意味着高延迟与高费用。每次对话都将所有历史拼接进去是不现实的。
3. **检索限制**：Transformer的"Lost in the Middle"现象意味着——当上下文中有大量历史对话时，中间部分的信息检索率显著降低。Agent可能"忘记"几分钟前的关键信息。

因此，记忆系统的价值不是"扩大容量"，而是**提供有结构的、按需检索的信息访问机制**——类似于图书馆的索引系统，而非无限书架。

### 认知心理学框架：三层记忆的映射

认知心理学将人类记忆系统分为三层：感知记忆（sensory memory，毫秒级，容量极大但几乎不进入意识）、工作记忆（working memory，秒级，容量4±1个组块）、长期记忆（long-term memory，终身，容量几乎无限）。LLM Agent研究者将这一三层架构映射到计算系统，形成了今日Agent记忆设计的理论框架 [Du, 2026]。

**感知记忆（Sensory Memory）**。在Agent系统中，感知记忆对应于原始环境输入的短暂缓存。对于文本Agent，这是用户输入的原始token流；对于多模态Agent，这是图像特征、音频波形或传感器读数的原始表示。感知记忆的关键需求是低延迟与无损存储，通常直接流入工作记忆而不做复杂处理。与人类的视觉暂留（iconic memory）类似，Agent的感知记忆只保留最近数秒至数分钟的原始输入，过期后丢弃或压缩。

**工作记忆（Working Memory）**。工作记忆对应于LLM的上下文窗口。当前GPT-4支持128k token，Claude 3支持200k token，Gemini 1.5支持1M token。上下文窗口是Agent进行in-context learning、推理与决策的"思维工作台"。

然而，工作记忆存在三个硬性约束：
1. **容量约束**：即使1M token窗口也无法容纳一本书或整个代码库。
2. **衰减约束**：Transformer的注意力机制对所有位置一视同仁，但实践中，中间位置的信息容易被"Lost in the Middle"——模型对上下文中间部分的检索显著弱于首尾部分。**[强实验证据]**
3. **成本约束**：上下文长度与推理计算量呈二次或线性关系，长上下文意味着高延迟与高费用。

工作记忆的管理策略包括：
- **滑动窗口**：只保留最近K轮对话，实现简单但会丢弃早期关键信息。
- **摘要压缩**：用LLM将早期对话压缩为摘要，在保持关键信息的同时减少token消耗。
- **层次化索引**：将历史对话按主题聚类，按需检索加载——类似于书籍的目录系统。

**长期记忆（Long-term Memory）**。长期记忆是Agent的持久化知识存储，通常分为两类：
- **语义记忆（Semantic Memory）**：存储事实、概念与关系。在Agent中通常由向量数据库实现，通过RAG机制按需检索到工作记忆。
- **情景记忆（Episodic Memory）**：存储事件序列与时间线。例如，用户上周要求"用Python写一个快速排序"，昨天要求"优化这个排序的性能"——这些交互事件按时间顺序存储，支持时间线查询（"用户上周让我做了什么？"）。情景记忆通常用图数据库（如Neo4j）或时间序列数据库（如TimescaleDB）实现，节点为事件，边为时间关系或因果关系。

### 记忆架构的工程实现：分层缓存策略

一个完整的Agent记忆系统通常采用分层缓存策略，与计算机体系结构中的缓存层次（L1/L2/L3/RAM）完全一致：

| 层级 | 存储介质 | 访问延迟 | 容量 | 类比 |
|------|---------|---------|------|------|
| L1 | 上下文窗口（KV Cache） | 纳秒级 | ~100k token | CPU寄存器 |
| L2 | 内存/Redis（近期摘要） | 毫秒级 | ~10k token | L3缓存 |
| L3 | 向量数据库（语义记忆） | 毫秒至秒级 | 百万级文档 | 内存 |
| L4 | 图数据库/关系数据库（情景记忆） | 秒级 | 无上限 | 磁盘 |

当Agent需要回答用户问题时，系统首先查询L1（工作记忆中的当前上下文），若信息不足则检索L2（近期摘要），仍不足则检索L3（向量数据库），最后查询L4（结构化知识库）。这种分层检索遵循"越热的数据越先访问"原则，类似于计算机体系结构中的缓存逐级回退机制。

### 记忆的写入与遗忘：主动管理的重要性

人类记忆不是简单的"存储-读取"系统，而是经过编码、巩固、遗忘的主动过程。Agent记忆同样需要遗忘策略，否则记忆系统会：

1. **存储膨胀**：向量数据库中积累了大量过时知识（如旧软件版本的API文档），占用存储空间并污染检索结果。
2. **信号稀释**：无关事件（如日常闲聊）与关键事件（如用户偏好、重要决策）混在一起，检索时关键信息被稀释。

推荐的遗忘策略：
- **L1工作记忆**：无遗忘，由上下文窗口硬截断。
- **L2近期摘要**：基于token数量的滑动窗口，超出部分用LLM压缩为摘要。压缩提示应明确指示"保留关键事实、丢弃闲聊"。
- **L3语义记忆**：按"显著性分数"指数衰减。每次用户查询命中某chunk时，该chunk分数增加；长期未命中则衰减。衰减公式：`score_new = score_old * exp(-Δt / τ)`，其中τ为时间常数（如30天）。
- **L4情景记忆**：按时间衰减与重要性双重过滤。用户标记为"重要"的事件应永久保存；日常闲聊类事件在24小时后衰减至零并删除。

**一个常被忽视的问题**：遗忘策略的设计本身可能引入偏差。如果系统对"重要事件"的判断基于用户反馈，那么活跃提问多的用户的历史会被过度保留，而沉默用户的偏好可能被过早遗忘。这是一种**活跃度偏差**，可能导致Agent对不同用户的个性化程度不一致。**[开放问题]**

### 批判性分析：人类记忆类比的根本局限

将认知心理学的记忆模型映射到Agent架构是一个**启发式类比**，而非严格的理论等价。关键差异在于：

1. **信息表征形式**：人类记忆是**联想网络**（associative network）——每个记忆与其他记忆通过语义、时间和情感关系连接。Agent的语义记忆（向量数据库）是**几何空间**（points in high-dimensional space），情景记忆（图数据库）是**结构化图**。这三种表征形式的检索机制和容错能力根本不同。
2. **遗忘机制**：人类遗忘（如衰退、干扰）是**被动的、不可控的**；Agent遗忘是**主动设计的**（指数衰减、定期清理）。这意味着Agent可以选择性地"记住"或"忘记"，而人类不能。
3. **元记忆能力**：人类知道自己在记忆什么（metamemory），可以主动决定记忆策略。Agent缺乏这种元认知能力——它不知道自己存储了什么（除非有检索日志），自然也无法主动管理记忆。

这些差异意味着：**Agent记忆系统不是"人类记忆的计算模拟"，而是"受人类记忆启发但根本不同的信息管理系统"**。过度依赖人类记忆类比可能导致设计上的误导。**[合理推断]**


---

## 9.5 规划：从ReAct单步到层次化任务分解

### 问题动机：为什么ReAct不够用？

第8章详细介绍了ReAct [Yao et al., 2022] 框架：语言模型通过交替生成推理轨迹（Thought）与行动（Action），在推理与行动的交织中解决任务。ReAct的成功验证了"显式推理链"对工具使用与问题求解的价值，但其原始设计主要针对单步或少数步的简单任务（如HotpotQA的多跳问答）。

当任务复杂度上升——例如"为我规划一次日本七日游，包含航班预订、酒店选择、每日行程安排与预算控制"——ReAct的线性推理链面临组合爆炸。这类似于：一个人可以用纸笔规划一顿晚餐（买菜→做饭→摆盘），但用同样的方法规划一场千人婚礼时会崩溃——不是因为能力不够，而是因为**线性思维无法管理组合爆炸的任务依赖关系**。

### 从单步到多步：工具链的组合问题

ReAct的每个Action只调用一个工具，返回的Observation直接进入下一步推理。对于需要多个工具协同的任务（如先搜索航班、再查询酒店、再计算总价），ReAct依赖模型在每一步自主决定下一步工具。这种贪婪的单步决策缺乏全局规划视野，容易陷入局部最优。

**错误驱动学习：ReAct的局部最优陷阱**

```python
# 常见错误：用ReAct处理需要全局规划的多步任务
# 场景：规划日本七日游，预算10000元

Thought: 先搜索便宜的航班。
Action: search_flights("上海", "东京", "2025-07-01")
Observation: 最便宜航班￥3000，到达时间7月2日凌晨1点。

Thought: 航班很便宜，选这个。现在搜索酒店。
Action: search_hotels("东京", "2025-07-02", "2025-07-08")
Observation: 东京酒店均价￥800/晚，6晚￥4800。

Thought: 预算还剩￥2200用于餐饮和景点。
Action: plan_daily_itinerary("东京", 7, 2200)
Observation: 预算不足，东京7日游建议预算至少￥5000（不含交通住宿）。

# 错误分析：ReAct在第一步选择了最便宜的航班，但凌晨1点到达导致
# 第一天无法游览（浪费一天住宿和餐饮预算），同时高估了剩余预算。
# 如果有全局规划，应该选择白天到达的稍贵航班（￥3500），
# 但能节省一天的住宿和餐饮费用。
# [注意] ReAct的贪婪决策缺少"回溯"能力——一旦选择了某个航班，
# 后续所有决策都基于这个选择，即使发现它是次优的也无法回退。
# [关键局限] 线性推理链无法表达"如果A则B，否则C"的条件分支和回溯。
```

Kim et al. [2023] 提出的LLM Compiler将这一问题显式化：通过静态分析工具依赖图，将串行工具调用并行化。例如，航班搜索与酒店搜索之间无数据依赖，可以并行执行；而总价计算依赖于航班与酒店的结果，必须串行。这种"编译时优化"将ReAct的线性链转换为DAG（有向无环图）执行计划，显著降低了多步任务的延迟。

### 层次化任务分解（Hierarchical Task Decomposition）

当任务复杂度超过一定阈值时，人类会采用"分而治之"策略：先制定高层计划（"去日本→关东→东京→浅草寺"），再逐步细化每个子计划。LLM Agent的层次化规划遵循同一原则。

Wei et al. [2026] 提出的"Planner-Centric Framework"在ReAct之上引入了一个显式的规划器（Planner）模块：规划器接收高层目标，生成一个由子任务组成的计划树（plan tree），每个子任务分配给专门的执行器（Executor）。执行器可以是一个独立的LLM实例、一个工具调用模块或一个子Agent。规划器持续监控执行器的反馈，当检测到偏差（如航班预订失败）时，触发重规划（replanning）。

这种架构与机器人领域的HTN（Hierarchical Task Network）规划器一脉相承，但用LLM替代了手工编写的任务分解规则。

**认知科学类比**：层次化规划对应认知科学中的"执行功能"（executive function）——人类前额叶皮层负责制定高层计划、分解子目标、监控执行进度。当任务复杂时，人类会自动切换到层次化规划模式。LLM Agent的层次化架构在功能上复现了这一机制，但实现方式不同——人类的层次化规划是**并行分布式的**（多个脑区协同），而Agent的层次化规划是**串行层级式的**（Planner→Executor→反馈）。**[合理推断]**

### 反思与重规划（Reflexion & Replanning）

层次化规划的前提是规划器能够识别执行失败。ReAct原始框架中，模型一旦开始执行，只能沿着单一路径前进，没有显式的"反思"步骤。

Shinn et al. [2023] 的Reflexion框架在执行链的末端添加了一个评估器（Evaluator）：评估器用LLM判断最终结果是否正确，若错误则生成反思文本（"我在第二步错误地使用了加法而非乘法"），将反思注入工作记忆，然后重新执行。这种自我纠正机制使Agent在算术、编程等任务上的成功率提升15%-30%。**[强实验证据]**

在更复杂的层次化架构中，反思不仅发生在任务末端，还发生在每个子任务完成后：每个子Agent返回结果时，父级规划器先验证结果合理性，再决定是否接受、修正或重新分配。

**反思机制的局限**：Reflexion的自我纠正能力受限于LLM的**自我评估能力**——如果LLM无法判断自己的答案是否正确，反思就无法触发。这在需要外部验证的领域（如数学证明、代码测试）表现良好（因为可以运行测试用例验证），但在主观判断领域（如文本质量评估）效果有限。**[合理推断]**

### 规划的表示形式：自然语言 vs 符号 vs 程序

ReAct使用自然语言文本作为推理轨迹的表示，这带来了灵活性（任何任务都可以用自然语言描述）但也带来了不精确性（自然语言无法表达严格的约束关系，如"酒店必须在机场附近 AND 预算低于￥500/晚"）。

**方案空间**：

**路径A：自然语言推理（ReAct）**
- 优势：灵活、通用、可解释
- 劣势：不精确、无法表达严格约束、容易"幻觉"

**路径B：符号规划（PDDL）**
- 优势：形式化、可验证、可证明最优
- 劣势：需要领域建模（手工编写PDDL）；LLM将自然语言翻译为PDDL的精度有限

**路径C：程序表示（Python代码）**
- 优势：精确、可执行、有明确的输入输出类型
- 劣势：代码生成的错误率较高；调试成本高

**选择逻辑**：现代工业级Agent系统通常混合使用上述表示：高层用自然语言ReAct进行灵活推理，中层用DAG或代码表示工具依赖，低层用传统规划器处理严格约束。这种混合策略牺牲了形式化的纯粹性，换取了灵活性。**[合理推断]**

### 反事实推演：如果LLM的推理能力足够强，还需要层次化规划吗？

假设2027年的LLM能在单次推理中处理1M token、执行100步推理链而不出错：

- **短期（1-2年）**：线性ReAct可能在多数任务上表现接近层次化规划——因为模型能"在脑中"完成分解和规划，不需要外部的Planner模块。但层次化规划在**可调试性**上仍然有优势——你可以检查每个子任务的执行结果，而线性ReAct的推理链是一个不可分割的整体。
- **长期（5-10年）**：如果LLM的推理能力达到人类水平，层次化规划可能退化为"优化手段"而非"必需品"——类似于人类处理简单任务时不需要显式分解，但处理复杂任务时仍然会列清单。**[开放问题]**

这个反事实揭示了一个关键点：**层次化规划的价值不在于"LLM能力不够"，而在于"复杂任务的依赖关系需要显式管理"**。即使LLM的推理能力无限强，管理100个工具调用的依赖关系仍然需要某种结构化的规划机制。

### 认知偏差警示：规划的"过度工程化"陷阱

Agent系统设计者可能陷入**过度工程化偏差**——为简单任务设计复杂的层次化规划架构。如果一个任务只需要3步工具调用，引入Planner-Executor-Reflector的完整架构会增加延迟和token消耗，而不一定提升成功率。在工程实践中，选择规划策略的关键标准是**任务复杂度**：简单任务用线性ReAct，中等复杂度用DAG并行化，高复杂度才用层次化分解。**[作者推测]**


---

## 9.6 代码：带向量数据库的RAG系统与长期记忆存储实现

本节提供一个可运行的PyTorch实现，涵盖：(1) 基于FAISS的向量检索（若FAISS未安装则回退到纯PyTorch实现）；(2) 文档分块与Embedding；(3) 重排序；(4) 分层记忆存储（工作记忆+向量语义记忆+情景时间线记忆）。

**错误驱动学习：先看常见错误实现**

```python
# 常见错误1：将所有文档作为一个chunk存入向量数据库
def bad_rag_1(documents: List[str], query: str):
    """错误：不对文档分块，直接编码整个文档"""
    embedder = Embedder()
    vectors = embedder.encode(documents)  # 每个文档可能上万token
    # [问题] embedding模型有最大输入长度限制（如256 token），
    # 超长文本会被截断，丢失大量信息
    # [后果] 检索到的文档向量只编码了文档开头部分，后续内容完全被忽略
    ...

# 常见错误2：chunk之间无重叠
def bad_rag_2(text: str, chunk_size: int = 256):
    """错误：无重叠的固定切分"""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
    # [问题] 如果关键信息恰好跨越两个chunk的边界（如"爱因斯坦"在chunk1末尾，
    # "出生于1879年"在chunk2开头），两个chunk都无法提供完整信息
    # [后果] 检索精度大幅下降，尤其是关键信息分布在边界处时
    return chunks

# 常见错误3：不做重排序
def bad_rag_3(query: str, vector_store: VectorStore, top_k: int = 5):
    """错误：直接使用向量检索结果，不做重排序"""
    qvec = embedder.encode([query])[0]
    results = vector_store.search(qvec, top_k=top_k)
    # [问题] 向量检索基于全局相似度，可能召回"语义相关但不含答案"的文档
    # [后果] Top-1结果可能只是"主题相关"而非"包含答案"
    return results
```

**正确实现**：

```python
"""
RAG + Long-term Memory System
Requirements: torch, numpy, sentence-transformers (optional), faiss-cpu (optional)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import json
import time

# ============================================================
# 1. Vector Store (FAISS if available, else pure PyTorch)
# ============================================================

class VectorStore:
    """Simple vector store with cosine similarity search."""
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.vectors = []  # List[np.ndarray]
        self.texts = []    # List[str]
        self.metadata = [] # List[dict]
        self.use_faiss = use_faiss and self._faiss_available()
        self.index = None
        if self.use_faiss:
            import faiss
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

    def _faiss_available(self):
        try:
            import faiss
            return True
        except ImportError:
            return False

    def add(self, vectors: np.ndarray, texts: List[str], metadata: Optional[List[Dict]] = None):
        """Add vectors and corresponding texts."""
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        for i, v in enumerate(vectors):
            self.vectors.append(v)
            self.texts.append(texts[i])
            self.metadata.append(metadata[i] if metadata else {})
        if self.use_faiss:
            self.index.add(vectors.astype('float32'))

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Return [(text, score, metadata), ...] sorted by score desc."""
        if len(self.vectors) == 0:
            return []
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        if self.use_faiss:
            import faiss
            scores, indices = self.index.search(query_vec.astype('float32').reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.texts):
                    continue
                results.append((self.texts[idx], float(score), self.metadata[idx]))
            return results
        else:
            # Pure PyTorch cosine similarity
            q = torch.tensor(query_vec, dtype=torch.float32)
            vecs = torch.tensor(np.stack(self.vectors), dtype=torch.float32)
            sims = torch.matmul(vecs, q)  # Cosine after normalization
            top_vals, top_idx = torch.topk(sims, min(top_k, len(sims)))
            return [(self.texts[i], float(sims[i]), self.metadata[i]) for i in top_idx.tolist()]

# ============================================================
# 2. Chunking Strategies
# ============================================================

def fixed_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size sliding window chunking."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks

def recursive_chunk(text: str, chunk_sizes: List[int] = [512, 256, 128]) -> List[str]:
    """Recursive character-based chunking: paragraph -> sentence -> fixed."""
    import re
    # Level 1: split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        tokens = para.split()
        if len(tokens) <= chunk_sizes[0]:
            chunks.append(para)
            continue
        # Level 2: split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        buf = []
        buf_len = 0
        for sent in sentences:
            sent_len = len(sent.split())
            if buf_len + sent_len > chunk_sizes[1] and buf:
                chunks.append(" ".join(buf))
                buf = [sent]
                buf_len = sent_len
            else:
                buf.append(sent)
                buf_len += sent_len
        if buf:
            chunks.append(" ".join(buf))
    return chunks

# ============================================================
# 3. Simple Embedding Wrapper (using sentence-transformers if available)
# ============================================================

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            self.model = None
            self.dim = 384  # fallback dimension
            print("Warning: sentence-transformers not installed. Using random embeddings for demo.")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)
        # Fallback: deterministic random projection for demo
        rng = np.random.default_rng(42)
        base = rng.random((len(texts), self.dim)).astype(np.float32)
        for i, t in enumerate(texts):
            h = hash(t) % 10000
            rng2 = np.random.default_rng(h)
            base[i] += rng2.random(self.dim) * 0.1
        return base

# ============================================================
# 4. Cross-Encoder Reranker (simplified mock for demonstration)
# ============================================================

class Reranker:
    """Mock reranker. In production, use BGE-Reranker or cross-encoder."""
    def __init__(self):
        self._biases = {
            "cost": -0.5, "price": 0.3, "budget": 0.4,
            "error": -0.8, "bug": -0.6, "fail": -0.7
        }

    def score(self, query: str, candidate: str) -> float:
        # A toy keyword-based scorer for demonstration.
        qtok = set(query.lower().split())
        ctok = set(candidate.lower().split())
        overlap = len(qtok & ctok)
        base = overlap / (len(qtok) + 1e-3)
        for kw, bias in self._biases.items():
            if kw in candidate.lower():
                base += bias
        return float(torch.sigmoid(torch.tensor(base * 2.0)))

    def rerank(self, query: str, candidates: List[Tuple[str, float, Dict]], top_k: int = 3):
        scored = [(cand, self.score(query, cand), meta) for cand, _, meta in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# ============================================================
# 5. Hierarchical Memory System
# ============================================================

class HierarchicalMemory:
    """
    L1: Working memory (context window, limited slots)
    L2: Recent summary (compressed short-term history)
    L3: Semantic memory (vector store, facts)
    L4: Episodic memory (structured timeline, events with timestamps)
    """
    def __init__(self, embedder: Embedder, working_memory_limit: int = 10):
        self.embedder = embedder
        self.working_memory = deque(maxlen=working_memory_limit)  # L1
        self.recent_summary = ""  # L2
        self.semantic_memory = VectorStore(embedder.dim, use_faiss=False)  # L3
        self.episodic_memory = []  # L4: list of {"time": float, "event": str, "importance": float}

    def add_interaction(self, user_input: str, agent_response: str, importance: float = 1.0):
        # L1: append to working memory
        self.working_memory.append(("user", user_input))
        self.working_memory.append(("agent", agent_response))

        # L2: update summary (naive: just append; in production, use LLM to compress)
        self.recent_summary += f"User: {user_input}\nAgent: {agent_response}\n"
        if len(self.recent_summary.split()) > 500:
            self.recent_summary = "... " + " ".join(self.recent_summary.split()[-400:])

        # L3: store user input as semantic chunk
        vec = self.embedder.encode([user_input])
        self.semantic_memory.add(vec, [user_input], [{"type": "user_query", "importance": importance}])

        # L4: episodic event
        self.episodic_memory.append({
            "time": time.time(),
            "event": f"User asked: {user_input[:80]}...",
            "importance": importance
        })
        self._decay_episodic()

    def _decay_episodic(self, threshold: float = 0.1):
        """Exponential decay of episodic memory based on age and importance."""
        now = time.time()
        decayed = []
        for e in self.episodic_memory:
            age_hours = (now - e["time"]) / 3600.0
            score = e["importance"] * np.exp(-age_hours / 24.0)  # 24h half-life
            if score > threshold:
                e["score"] = score
                decayed.append(e)
        self.episodic_memory = decayed

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, List]:
        """Multi-tier retrieval: working -> semantic -> episodic."""
        results = {
            "working": list(self.working_memory),
            "semantic": [],
            "episodic": []
        }
        # L3 semantic retrieval
        qvec = self.embedder.encode([query])[0]
        semantic_hits = self.semantic_memory.search(qvec, top_k=top_k)
        reranker = Reranker()
        results["semantic"] = reranker.rerank(query, semantic_hits, top_k=top_k)

        # L4 episodic retrieval: keyword match + importance score
        qtok = set(query.lower().split())
        episodic_hits = []
        for e in self.episodic_memory:
            etok = set(e["event"].lower().split())
            overlap = len(qtok & etok)
            if overlap > 0:
                episodic_hits.append((e["event"], overlap * e.get("score", e["importance"]), e))
        episodic_hits.sort(key=lambda x: x[1], reverse=True)
        results["episodic"] = episodic_hits[:top_k]
        return results

    def build_prompt_context(self, query: str, max_tokens: int = 1500) -> str:
        """Build a prompt by filling from L1->L2->L3->L4 until max_tokens."""
        retrieved = self.retrieve(query)
        parts = []
        # L1 working memory
        if retrieved["working"]:
            parts.append("## Recent Conversation\n")
            for role, text in retrieved["working"]:
                parts.append(f"{role.capitalize()}: {text}\n")
        # L2 recent summary
        if self.recent_summary:
            parts.append(f"## Summary\n{self.recent_summary}\n")
        # L3 semantic memory
        if retrieved["semantic"]:
            parts.append("## Relevant Facts\n")
            for text, score, meta in retrieved["semantic"]:
                parts.append(f"- {text} (score={score:.3f})\n")
        # L4 episodic memory
        if retrieved["episodic"]:
            parts.append("## Past Events\n")
            for event, score, meta in retrieved["episodic"]:
                parts.append(f"- {event} (score={score:.3f})\n")
        context = "".join(parts)
        # Naive token truncation
        tokens = context.split()
        if len(tokens) > max_tokens:
            context = " ".join(tokens[:max_tokens])
        return context

# ============================================================
# 6. RAG System integrating retriever + generator (mock generator)
# ============================================================

class RAGSystem:
    def __init__(self, embedder: Embedder, reranker: Reranker):
        self.embedder = embedder
        self.reranker = reranker
        self.vector_store = VectorStore(embedder.dim, use_faiss=False)
        self.chunker = recursive_chunk

    def ingest_document(self, doc_id: str, text: str, chunk_size: int = 256):
        chunks = self.chunker(text)
        if not chunks:
            return
        vectors = self.embedder.encode(chunks)
        metadata = [{"doc_id": doc_id, "chunk_index": i, "strategy": "recursive"}
                    for i in range(len(chunks))]
        self.vector_store.add(vectors, chunks, metadata)

    def answer(self, query: str, top_k: int = 5) -> Dict:
        qvec = self.embedder.encode([query])[0]
        hits = self.vector_store.search(qvec, top_k=top_k * 2)  # over-fetch for reranking
        reranked = self.reranker.rerank(query, hits, top_k=top_k)
        # In production, feed retrieved context into LLM with prompt:
        # "Given the following documents, answer the query."
        context = "\n".join([f"[{i+1}] {text}" for i, (text, _, _) in enumerate(reranked)])
        return {
            "query": query,
            "retrieved_context": context,
            "sources": reranked,
            "answer": f"(Mock) Based on {len(reranked)} documents, answer would be generated here."
        }

# ============================================================
# 7. Demo / Test
# ============================================================

if __name__ == "__main__":
    embedder = Embedder()
    reranker = Reranker()

    # Demo RAG ingestion
    rag = RAGSystem(embedder, reranker)
    doc = """
    Transformer models have revolutionized NLP since 2017. The attention mechanism
    allows the model to weigh the importance of different input tokens. However,
    standard attention has O(n^2) complexity with respect to sequence length. FlashAttention
    reduces the memory footprint by tiling the attention computation, keeping the algorithmic
    complexity unchanged but significantly lowering HBM memory access.
    """
    rag.ingest_document("doc_001", doc)

    # Query the RAG system
    result = rag.answer("What is the complexity of attention?")
    print("=== RAG Result ===")
    print(result["retrieved_context"])
    print("Answer:", result["answer"])

    # Demo Hierarchical Memory
    mem = HierarchicalMemory(embedder, working_memory_limit=6)
    mem.add_interaction("What is FlashAttention?", "FlashAttention is a memory-efficient attention algorithm.", importance=1.0)
    mem.add_interaction("How does it reduce memory?", "It uses tiling to reduce HBM accesses.", importance=0.8)
    mem.add_interaction("What is the complexity?", "O(n^2) in sequence length, but with better memory constants.", importance=1.0)

    retrieved = mem.retrieve("attention complexity")
    print("\n=== Memory Retrieval ===")
    print("Semantic hits:", retrieved["semantic"])
    print("Episodic hits:", retrieved["episodic"])

    prompt_ctx = mem.build_prompt_context("attention complexity")
    print("\n=== Prompt Context ===")
    print(prompt_ctx[:800])
```

**代码设计决策解析**：

1. **向量存储的抽象**：`VectorStore`类封装了FAISS与纯PyTorch两种后端。FAISS的`IndexFlatIP`在归一化向量上等价于cosine similarity，但速度更快；当FAISS不可用时，纯PyTorch实现确保了可运行性。生产环境中应使用FAISS的IVF（Inverted File Index）或HNSW（Hierarchical Navigable Small World）索引以处理百万级以上的向量。

2. **分块策略的对比**：`fixed_chunk`实现简单但可能切断语义边界；`recursive_chunk`优先按段落、再按句子、最后按固定长度分割，在实际RAG系统中更为常用。实验数据表明，对于问答任务，递归分割在NQ数据集上的Top-5检索精度通常比固定分割高3-8个百分点。**[强实验证据]**

3. **分层记忆的检索顺序**：`retrieve()`方法按L1→L3→L4的顺序搜索，遵循"越热的数据越先访问"原则。`build_prompt_context()`将检索结果按此顺序拼接，然后用朴素的token计数截断。生产环境中应使用tiktoken或transformers的tokenizer进行精确截断。

4. **重排序的模拟**：`Reranker`类是一个简化示例，实际生产环境应使用`BAAI/bge-reranker-large`或`cross-encoder/ms-marco-MiniLM-L-6-v2`等预训练交叉编码器。交叉编码器将query与document拼接后输入Transformer，计算相关性分数，其精度远高于双编码器的点积相似度，但计算成本高一个数量级。


---

## 9.7 工程实践框

### Chunk大小选择：没有普适最优值

Chunk大小没有普适最优值，它取决于三个因素：embedding模型的最大输入长度、文档的平均信息密度、以及下游任务的粒度需求。

- 对于**事实性问答**（如"某某公司的CEO是谁"），128-256 token的chunk通常最优，因为这类问题的答案往往集中在单个句子中。
- 对于**总结性任务**（如"总结这篇论文的贡献"），512-1024 token的chunk能保留更多上下文。

一个实用的调试方法是：在开发集上运行检索，检查正确答案是否落在Top-3 chunk中，若频繁出现"正确答案被切分在相邻chunk边界"的情况，则增加overlap比例或改用语义分割。

**认知偏差根源**：工程师在选择chunk大小时容易陷入**锚定偏差**——默认使用LangChain等框架的预设值（如512 token），而不根据实际文档结构和任务需求调整。正确的做法是：先分析文档的信息密度分布（如每100 token包含多少个关键事实），再据此选择chunk大小。

### 检索失败模式与诊断

RAG系统在生产环境中最常见的失败模式不是生成错误，而是检索错误。具体表现包括：

1. **语义漂移**：用户用口语化查询（"怎么让模型跑得更快"）检索技术文档，embedding模型无法将口语映射到正式术语（"inference optimization"）。
   - 解决：查询扩展（query expansion）或领域微调embedding模型。
   - 认知偏差根源：**可用性偏差**——工程师测试时用的是与文档语言风格一致的查询，但真实用户的查询风格差异很大。

2. **信息碎片化**：答案分散在多个chunk中，单个chunk无法回答完整问题。
   - 解决：parent-child chunking（小块用于检索，大块用于生成），或多轮检索（iterative retrieval）。

3. **陈旧知识**：向量数据库中的文档已过时。
   - 解决：为每个chunk附加时间戳metadata，在检索时优先返回近期文档；或维护"失效检测"管道。

4. **假阳性高排名**：检索返回的chunk在语义上与查询相关，但不包含实际答案。
   - 解决：引入重排序器，或训练"答案存在性"分类器过滤检索结果。

### 记忆遗忘策略

长期记忆系统的容量不是无限的，必须设计遗忘机制。推荐的分层策略：
- **工作记忆（L1）**：无遗忘，由上下文窗口硬截断。
- **近期摘要（L2）**：基于token数量的滑动窗口，超出部分用LLM压缩为摘要。压缩提示应明确指示"保留关键事实、丢弃闲聊"。
- **语义记忆（L3）**：按"显著性分数"指数衰减。衰减公式：`score_new = score_old * exp(-Δt / τ)`，其中τ为时间常数（如30天）。分数低于阈值的chunk可归档到冷存储。
- **情景记忆（L4）**：按时间衰减与重要性双重过滤。用户标记为"重要"的事件应永久保存；日常闲聊类事件在24小时后衰减至零并删除。

### 硬件需求估算（2025年中参考价格）

RAG系统的硬件成本主要集中在三个环节：

1. **Embedding计算**：100万文档 × 512 token/文档，使用`all-MiniLM-L6-v2`（22M参数）在V100 GPU上约需2小时；在CPU上约需20小时。若使用更大的`BAAI/bge-large-en`（335M参数），GPU时间约10小时。
   - 2025年中参考：单张A100 80GB租赁价格约$2-4/小时（按需），100万文档的embedding计算成本约$4-8。

2. **向量存储**：100万条768维float32向量占用约`1e6 × 768 × 4 ≈ 3 GB`内存。FAISS的IVF索引额外增加约20%开销。若使用HNSW索引，内存开销可能翻倍。
   - 2025年中参考：32GB内存的云服务器约$0.5-1/小时。

3. **生成阶段**：检索本身只贡献毫秒级延迟；真正瓶颈是LLM生成。一个128k上下文、4k输出的请求在GPT-4级别模型上成本约为$3-5，延迟约30-60秒。优化方向：通过摘要与过滤将实际输入上下文压缩到10k token以内，可降低成本约80%。

### 超参选择速查表

| 组件 | 超参数 | 推荐值 | 调整信号 | 原理说明 |
|------|--------|--------|----------|---------|
| Chunking | chunk_size | 256-512 | 答案碎片化→增大；噪声混入→减小 | chunk需足够大以包含完整语义单元，但不能超出embedding模型的输入限制 |
| Chunking | overlap | 10%-20% | 边界切断→增大；冗余重复→减小 | 重叠确保跨边界信息至少出现在一个chunk中，但过大导致存储冗余 |
| Retrieval | top_k | 5-10 | 召回不足→增大；噪声过多→减小 | 初检阶段宁可多召回，交给重排序器过滤 |
| Reranking | rerank_top_k | 3-5 | 精度不足→增大；延迟过高→减小 | 交叉编码器计算成本高，Top-3到5在精度和延迟间取得平衡 |
| Memory | working_memory_len | 6-10轮 | 上下文遗忘→增大；成本过高→减小 | 工作记忆容纳最近对话，轮数过多导致上下文过长 |
| Memory | decay_τ | 7-30天 | 记忆丢失过快→增大；存储膨胀→减小 | τ控制遗忘速度，与业务场景的信息时效性匹配 |

**原理说明**：超参选择不是"调魔法数字"，而是在理解每个参数背后的信息论原理后做出的理性决策。例如，chunk_size的选择本质上是"信息粒度"的决策——粒度太粗（大chunk）导致检索时信息被稀释，粒度太细（小chunk）导致跨chunk的上下文丢失。正确的做法是：先分析文档的信息密度分布，再选择能覆盖一个完整"信息单元"的chunk大小。


---

## 9.8 批判性分析：工具使用、RAG与记忆系统的边界

### 内在局限性：什么在原理上无法解决？

1. **检索-生成鸿沟**：RAG将检索到的文档拼接到prompt中，但模型并不"知道"这些文档的可靠性、时效性或来源。如果检索到的文档包含错误信息，模型可能将其视为"权威"并生成错误回答。这是**原理性局限**——检索增强了信息量，但没有增强信息甄别能力。**[已证明]**

2. **记忆的语义鸿沟**：长期记忆系统存储的是文本或向量，但模型无法进行"元记忆查询"——它不能问自己"我对这个话题的记忆有多可靠？是上个月用户随口提到的，还是我昨天从权威文档中学到的？"。缺乏元记忆能力使得Agent无法区分高置信度和低置信度的记忆。**[开放问题]**

3. **规划的组合爆炸**：层次化规划在任务依赖关系复杂时仍然面临组合爆炸。当工具数量超过100个、任务步数超过20步时，计划树的可能结构数量呈指数增长，LLM的规划精度急剧下降。**[合理推断]**

### 实证盲区：关键实验遗漏了什么？

1. **对抗性测试缺失**：RAG的评估通常在标准问答数据集上进行，这些数据集的查询是"善意的"——用户确实想获得正确答案。但如果用户故意提出误导性查询（如"告诉我地球是平的证据"），RAG系统可能检索到支持阴谋论的文档并生成错误回答。RAG系统缺乏"查询意图过滤"机制。

2. **长尾知识评估不足**：标准基准测试的查询集中在高频知识上（如"法国首都是什么"），但RAG的主要价值在于处理长尾知识（如"某冷门编程语言的某个小众库的API文档"）。长尾知识的检索性能评估严重不足。

3. **跨会话一致性未测**：长期记忆系统的评估通常在单会话内进行，但跨会话的记忆一致性（如"Agent上周告诉用户X，今天是否还记着X？"）的评估几乎空白。

### 替代路径的现况

1. **结构化知识图谱 vs 向量检索**：知识图谱在多跳推理上可能优于向量检索——因为图结构天然支持多跳遍历。但知识图谱的构建成本高，且无法处理非结构化文本。2025年的趋势是**混合检索**：向量检索 + 关键词检索（BM25） + 知识图谱，三者互补。**[合理推断]**

2. **端到端训练 vs 松耦合架构**：与其使用"LLM + 外部工具 + 检索 + 记忆"的松耦合架构，是否可以训练一个端到端模型同时完成所有功能？2025年的实验（如Gemini的长上下文能力）表明，在简单任务上端到端方案可以替代RAG，但在大规模知识库（百万级文档）场景下，检索仍然是必要的。**[强实验证据]**

3. **统一记忆 vs 分层记忆**：当前主流是L1-L4分层架构，但是否可以用一个统一的"记忆网络"替代？神经图灵机（Neural Turing Machine）和Differentiable Neural Computer尝试过这一方向，但在大规模场景下训练不稳定。**[开放问题]**

### 认知偏差警示

1. **确认偏差**：RAG研究者倾向于在RAG擅长的场景（事实性问答）中评估RAG，而非在RAG不擅长的场景（推理性查询、对抗性查询）中挑战它。这导致RAG的"成功率"被系统性高估。

2. **工具偏差**：Agent系统设计者倾向于"什么都加工具"——即使模型本身能回答的问题也让它调用API。这不仅增加了延迟和成本，还可能降低回答质量（API返回的信息可能不如模型参数中的知识准确）。正确的做法是：只在模型"不知道"或"不确定"时才调用工具，而判断"不知道"本身需要元认知能力——这正是Toolformer试图解决的。

3. **架构复杂性偏差**：工程师倾向于设计过度复杂的记忆和规划架构（如五层记忆系统、三阶段规划器），因为"复杂看起来更专业"。但在多数实际应用中，简单的两层架构（上下文窗口 + 向量数据库）就能覆盖80%的需求。**[作者推测]**

---

## 9.9 与其他章节概念的连接

### 前置知识

- **第1章**：本章的"工具使用+RAG+记忆"三角组合是第1章"LLM+Agent+World Model"互补框架在信息获取维度的具体实现。第1章讨论的是"为什么需要互补"，本章讨论的是"如何实现互补"。
- **第8章**：ReAct框架是本章所有讨论的基础架构。本章的Toolformer是对ReAct工具调用机制的改进（从prompt层面到参数层面），长期记忆是对ReAct工作记忆的扩展（从单会话到跨会话），层次化规划是对ReAct线性推理的升级（从单步到多步）。

### 后续展开

- **第10章**：多Agent协作将本章的单Agent工具使用扩展到多Agent工具共享与协调。当多个Agent各自拥有不同的工具和记忆时，如何协调信息共享是一个新挑战。
- **第11章**：安全与对齐将讨论工具使用的安全约束——如果Agent调用了错误的工具或检索到恶意文档，如何防止有害后果？

---

## 9.10 小结

本章从Toolformer的自监督工具调用学习出发，经过RAG的检索增强架构，延伸至长期记忆的分层存储，最终在规划层次上讨论了ReAct的扩展与层次化任务分解。

贯穿始终的主线是：**语言模型的能力边界不仅取决于参数规模，更取决于它与外部系统的接口设计**——工具是能力的扩展，检索是知识的延伸，记忆是状态的持久化，而规划是这些组件的协调器。

Toolformer证明了模型可以在预训练阶段内化工具调用语法，但其单步、闭集、无状态的局限推动了后续研究 **[强实验证据]**。RAG将外部知识解耦到向量数据库，实现了知识的动态更新，但检索质量严重依赖于chunk策略、embedding模型与重排序器的协同优化 **[强实验证据]**。长期记忆系统通过认知心理学的三层映射（感知→工作→长期），为Agent提供了跨会话的持续性，但记忆的写入、检索与遗忘策略仍是活跃的研究方向 **[开放问题]**。规划模块从ReAct的单步推理扩展到层次化任务分解与DAG并行执行，反映了Agent系统从简单问答工具向复杂工作流自动化平台的演进 **[合理推断]**。

这些组件并非独立存在。一个工业级Agent系统的典型执行流程是：用户输入进入感知记忆与工作记忆；规划器分解任务并决定是否需要检索或工具调用；检索器从向量数据库召回相关知识；工具执行器调用外部API；所有观察结果被重新注入工作记忆；最终生成器综合上下文输出回答，并将本轮交互写入长期记忆。这一循环——感知、推理、行动、记忆——构成了现代Agent系统的认知架构基础。

**核心结论**（附证据强度）：
- 工具调用、RAG和长期记忆的组合是当前Agent系统的有效架构 **[强实验证据]**
- 这是否是"最优"架构仍是开放问题 **[开放问题]**
- 松耦合集成在工程上可行且可调试，但统一架构在特定场景可能更优 **[合理推断]**
- 认知心理学记忆模型对Agent设计的指导价值有限，因为两者在表征形式和机制上存在根本差异 **[合理推断]**

---

## 本章练习题

### 概念题
> **Q1**：Toolformer和RAG都是"让模型获取外部信息"的方案。从以下三个维度对比：
> (a) 知识更新时机（训练时 vs 推理时）
> (b) 对新知识的适应速度
> (c) 计算开销与工程复杂度
>
> **批判性追问**：如果Toolformer的训练成本降低到与RAG部署成本相当的水平，Toolformer是否会取代RAG？为什么？（提示：考虑知识的动态性、工具闭集 vs 开放知识库、以及认知调节能力的泛化性。）

### 推导题
> **Q2**：RAG检索用余弦相似度：sim(q,d) = q·d / (‖q‖·‖d‖)
> (a) 证明：对L2归一化的向量，余弦相似度等价于内积。
> (b) 这个等价性在工程上有什么好处？（提示：FAISS为什么优先支持内积搜索？）
> (c) **边界条件分析**：当向量维度d趋于无穷时，随机向量的余弦相似度会趋近于什么值？这对RAG的检索精度意味着什么？（提示：考虑高维空间的"维度诅咒"。）

### 反事实分析题
> **Q3**：假设2023年Toolformer未被提出，而OpenAI转而采用REALM风格的紧耦合检索（检索器与生成器联合训练）来增强GPT-4。
> (a) 这种设计在知识更新频率上会有什么问题？
> (b) 如果问题(a)被某种工程手段缓解（如模块化微调），紧耦合方案在哪些维度上可能优于当前的松耦合RAG？
> (c) 从生态角度分析：紧耦合方案是否可能催生出与当前向量数据库产业不同的生态？请给出你的推理。
> **证据强度要求**：你的回答中，每个论断需标注证据强度等级（[已证明]/[强实验证据]/[合理推断]/[开放问题]/[作者推测]）。

### 批判性思考题
> **Q4**：阅读以下论断并评估：
> "RAG系统通过外部检索解决了LLM的幻觉问题。"
> (a) 这个论断在什么条件下成立？
> (b) 在什么条件下不成立？（即RAG系统仍然会产生幻觉的场景）
> (c) 用证据强度框架评估：这个论断的证据强度是什么级别？为什么？
> (d) **认知偏差分析**：持这个论断的人可能陷入了什么认知偏差？

### 编程题
> **Q5**：用纯NumPy实现一个ToyRAG：支持文档添加、余弦相似度检索、Top-K返回，不依赖任何向量数据库。
> (a) 添加4条文档后，查询"养老机器人跌倒检测"能检索到最相关的文档。
> (b) **扩展要求**：添加一个简单的重排序器（基于关键词重叠度），对比重排序前后的Top-3结果差异。
> (c) **边界条件测试**：当查询向量为全零向量时，系统应如何处理？当文档库为空时呢？

### 边界条件分析题
> **Q6**：考虑一个使用长期记忆系统的养老机器人Agent。分析以下边界条件：
> (a) 当用户的记忆条目超过向量数据库的容量上限时，系统应如何处理？（考虑遗忘策略的设计）
> (b) 当用户A的记忆被误检索为用户B的上下文时（跨用户记忆泄露），会造成什么后果？如何防止？
> (c) 当用户明确说"忘掉我之前说的X"时，如何实现"定向遗忘"？（这在当前的向量数据库架构中是一个开放问题——为什么？）

---

## 核心概念速查

| 术语 | 定义（≤25字） |
|------|-------------|
| Toolformer | 通过自监督在预训练阶段内化工具调用时机的方法 |
| RAG | 检索增强生成：推理时动态检索外部知识库 |
| 向量数据库 | 专为高维向量近似最近邻检索优化的存储系统 |
| 记忆流 | 存储Agent所有感知和对话历史的时序记忆结构 |
| 记忆反思 | 将原始记忆压缩提炼为高层抽象见解的过程 |
| 长期记忆 | 跨对话持久化存储的记忆（区别于上下文窗口） |
| 混合检索 | 结合稀疏检索（BM25）和稠密检索（向量）的策略 |
| 元认知 | 对自身认知状态的监控和调节能力 |
| 语义鸿沟 | 文档与查询向量分布不一致导致的检索失败问题 |
| 层次化规划 | 将复杂任务分解为子任务树的规划方法 |

---

## 延伸阅读

[Guu et al., 2020] K Guu, K Lee, Z Tung, P Pasupat, M Chang. "Retrieval augmented language model pre-training." *Proceedings of the 37th International Conference on Machine Learning (ICML)*, 2020. 该论文提出了REALM，首次在预训练阶段联合优化语言模型与稠密检索器，是RAG方向的前置奠基工作。

[Lewis et al., 2020] P Lewis, E Perez, A Piktus, F Petroni, V Karpukhin, NM Goyal, et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." *NeurIPS*, 2020. 该论文提出了解耦式RAG架构，将检索器与生成器分别训练，成为工业RAG系统的标准范式。

[Schick et al., 2023] T Schick, J Dwivedi-Yu, R Dessì, R Raileanu, M Lomeli, E Hambro, et al. "Toolformer: Language models can teach themselves to use tools." *NeurIPS*, 2023. 该论文提出自监督API调用合成方法，让模型在预训练阶段内化工具调用语法。

[Yao et al., 2022] S Yao, J Zhao, D Yu, N Du, I Shafran, K Narasimhan, et al. "ReAct: Synergizing reasoning and acting in language models." arXiv:2210.03629, 2022. 该论文提出了推理与行动交替执行的ReAct框架（详见第8章）。

[Qin et al., 2024] Y Qin, S Liang, Y Ye, K Zhu, L Yan, Y Lu, et al. "ToolLLM: Facilitating large language models to master 16000+ real-world APIs." *ICLR*, 2024. 该论文将Toolformer的闭集工具扩展到16,000+真实API。

[Du, 2026] P Du. "Memory for autonomous LLM agents: Mechanisms, evaluation, and emerging frontiers." arXiv:2603.07670, 2026. 该综述系统性地梳理了LLM Agent记忆系统的架构设计、评估基准与新兴研究方向。

[Kim et al., 2023] S Kim, S Moon, R Tabrizi, N Lee, MW Mahoney, et al. "An LLM compiler for parallel function calling." arXiv:2312.04511, 2023. 该论文提出将LLM工具调用从线性链编译为DAG并行执行图。

[Shinn et al., 2023] N Shinn, F Cassano, E Berman, A Gopinath, K Narasimhan, S Yao. "Reflexion: Language agents with verbal reinforcement learning." *NeurIPS*, 2023. 该论文提出在执行链末端添加反思评估器，实现自我纠正。

[Liu et al., 2023] N Liu, S Lin, J Raiman, et al. "Lost in the Middle: How language models use long contexts." *TACL*, 2023. 该论文揭示了Transformer模型对上下文中间部分检索能力显著弱于首尾部分的现象。

[Wei et al., 2026] H Wei, et al. "A planner-centric framework for LLM-based agent systems." 2026. 该论文提出以显式规划器为核心的Agent框架，支持层次化任务分解与重规划。

