# 第9章 工具使用、记忆与规划（Toolformer、RAG、Long-term Memory）

**核心思想**：语言模型通过显式工具调用、外部检索与结构化记忆存储，将自身从静态参数化的知识库转变为可扩展、有状态、能与外部世界持续交互的智能系统——其本质不是让模型记住更多，而是让模型学会在何时、以何种方式、向哪些外部系统请求信息。

**历史脉络**：在GPT-3 [Brown et al., 2020] 之前，语言模型的知识完全封闭于训练语料。研究者尝试过两种扩展路径：一是在预训练阶段不断增大模型参数与数据量，但计算成本呈指数级上升且知识仍无法实时更新；二是在推理阶段通过prompt拼接外部文本（如"根据以下维基百科内容回答问题"），但这种方式无法教会模型自主判断何时需要外部信息。前置工作REALM [Guu et al., 2020] 率先在预训练阶段融合检索器，证明检索增强的预训练能提升知识密集型任务性能，但它仍局限于闭域问答，未开放到通用工具调用。替代方案的失败教训在于：简单地在prompt中告诉模型"你可以使用计算器"并不能真正习得工具调用模式，模型在预训练阶段从未见过`<API>`标签与返回值的交互结构。Toolformer [Schick et al., 2023] 的突破在于利用自监督方式，在未标注语料中自动插入API调用示例，让模型在预训练阶段内化工具使用语法。几乎同时，RAG（Retrieval-Augmented Generation）框架将外部检索从预训练阶段解耦到推理阶段，用向量数据库实现了知识的动态更新。这两条路径——预训练阶段的工具内化与推理阶段的动态检索——最终在第8章的ReAct [Yao et al., 2022] 架构中交汇，形成了今日Agent系统的标准范式：推理（Reasoning）、行动（Acting）、记忆（Memory）的三元循环。

---

## 9.1 从Toolformer到RAG到长期记忆系统

2020年至2023年间，大语言模型社区经历了从"封闭知识"到"开放工具"的范式转换。这一转换并非单点突破，而是三个相互独立又逐步融合的技术路线的交汇。

**路线一：工具调用（Tool Use）**。GPT-3的1750亿参数在问答任务上表现优异，但面对"2023年法国总统是谁"或"计算14523的平方根"这类问题时，要么生成训练数据中的过时信息，要么在算术上出错。早期研究者尝试通过few-shot prompting注入工具使用示例，例如提供"Question: 14523的平方根是多少？ Thought: 我需要使用计算器。 Action: calculator(14523)"这样的演示。然而，GPT-3在预训练阶段从未见过`Action:`与`calculator()`这类结构化语法，因此泛化能力极差——当问题表述稍有变化时，模型要么不调用工具，要么生成格式错误的调用。Toolformer [Schick et al., 2023] 从根本上改变了这一状况：它不是在推理时通过prompt教模型使用工具，而是在继续预训练（continual pre-training）阶段，让模型在大量未标注文本中自动学习何时插入`<API>`调用、如何解析返回值。具体而言，Toolformer对每一个预训练样本中的每一个候选位置，采样一个API调用（如计算器、翻译器、搜索引擎），执行该调用，将返回值插入原文，然后用一个语言模型打分器评估插入后的文本困惑度（perplexity）是否降低。只有降低困惑度的调用才保留为训练数据。这种自监督筛选机制使模型在零样本情况下就能生成格式正确的工具调用。

**路线二：检索增强（Retrieval-Augmented Generation）**。与Toolformer的"主动调用"不同，RAG走的是"被动检索"路线。REALM [Guu et al., 2020] 是这一方向的开创性工作，它在预训练阶段同时优化语言模型与检索器（基于BERT的稠密检索编码器），让模型在生成每个token时都能从外部文档库中检索相关知识。然而，REALM将检索器与生成器紧耦合，导致知识库更新时需要重新训练整个系统。2020年底，Lewis et al. [2020] 提出的RAG框架将检索与生成解耦：检索器（基于DPR的稠密向量检索）与生成器（BART或T5）分别训练，在推理阶段通过向量检索动态获取上下文。这一架构使知识库可以独立更新，无需重新训练生成模型。随着2022年向量数据库（如FAISS、Milvus、Pinecone）的成熟，RAG从研究原型迅速转化为工业标准。RAG与Toolformer的关键差异在于：RAG的检索行为对语言模型透明（模型只接收到拼接后的上下文），而Toolformer的工具调用由模型自主生成显式语法。

**路线三：长期记忆（Long-term Memory）**。上述两条路线解决了"知识从哪里获取"的问题，但未解决"知识如何持久化与管理"的问题。对话系统的上下文窗口有限（早期GPT-3为2048 token，GPT-4为32k token），多轮对话中较早的信息会被截断。简单的滑动窗口记忆会丢失早期关键信息。2023年开始，研究者将认知心理学中的记忆层次模型（sensory memory → working memory → long-term memory）映射到LLM Agent架构：感知记忆对应实时环境输入（传感器数据、用户消息），工作记忆对应LLM的上下文窗口（in-context learning的载体），长期记忆则分为语义记忆（向量数据库中的事实知识）与情景记忆（事件流的时间序列记录）。Du [2026] 的综述系统性地梳理了这一架构映射，指出当前Agent系统的瓶颈已从"模型能力"转移到"记忆管理机制"。

三条路线的交汇点出现在ReAct [Yao et al., 2022] 与后续的工具增强Agent框架中。ReAct将推理（Thought）与行动（Action）交替执行，而行动的输出（Observation）被重新放入工作记忆，驱动下一步推理。当行动扩展到工具调用（Toolformer路线）与检索查询（RAG路线）时，一个完整的Agent认知循环便形成了。

## 9.2 Toolformer：在预训练阶段学习工具调用

Toolformer的核心方法论可以概括为"自监督API调用合成"。其假设是：如果语言模型在预训练阶段见过大量`<API>`标签与返回值的交互模式，它就能在推理时自主生成格式正确的工具调用。这一假设挑战了传统的"prompt工程"范式——后者认为工具使用是推理时通过上下文学习（in-context learning）获得的技能，而Toolformer证明这一技能可以在预训练阶段固化到模型参数中。

**数据构造**。Toolformer选取了计算器、问答系统（Q&A）、维基百科搜索、机器翻译、日历五个API。对于每一个预训练样本中的每一个位置，算法执行以下步骤：
1. 在该位置后插入一个候选API调用（例如`<API>Calculator(2+3)<API>`）；
2. 执行该调用，获取返回值（例如`5`）；
3. 将调用与返回值插入原文，形成增强文本；
4. 用一个辅助语言模型（与主模型相同规模）计算增强文本的加权交叉熵损失，并与原始文本对比；
5. 只有当损失下降超过阈值时，才保留该API调用作为训练数据。

这一筛选机制过滤掉了大量无意义的候选调用。例如，在句子"爱因斯坦出生于1879年"后插入"Calculator(1879)"不会降低困惑度，因此会被丢弃；而插入"QA(爱因斯坦出生年份)"并返回"1879"则会降低困惑度，被保留。最终，约5%-10%的预训练token位置被标注了API调用。

**训练目标**。Toolformer采用标准的自回归语言建模目标，但词汇表扩展了`<API>`、`<API>`等控制token。模型在训练时学习预测原始文本中的下一个token，以及API调用后的返回值。重要的是，模型在训练阶段看到API调用位置之前的上下文，但API调用本身与返回值作为输入的一部分被掩码处理——模型只负责预测调用后的原始文本token，而不是直接预测API调用。这种设计确保了模型不会因训练数据中的API调用而产生暴露偏差（exposure bias）。

**推理行为**。在推理时，Toolformer模型生成文本直到遇到`<API>`标签。此时，解码暂停，提取API调用参数，执行外部工具，将返回值（通常截断到固定长度，如100 token）重新插入解码上下文，然后继续生成。这一流程与第8章的ReAct循环高度相似：模型生成→执行工具→更新上下文→再次生成。区别在于，ReAct在prompt层面通过few-shot示例驱动工具调用，而Toolformer的工具调用语法已内化到模型参数中。

**局限性**。Toolformer的评估暴露了三个关键局限：
1. **单步调用**：模型无法处理需要多步工具链组合的任务（例如"先搜索A，再用A的结果搜索B"）。每一个`<API>`调用独立决策，没有跨调用的状态传递机制。
2. **工具闭集**：训练时只见过五个API，无法泛化到未见过的新工具。后续工作ToolLLM [Qin et al., 2024] 通过16,000+真实API的微调解决了这一问题。
3. **无状态性**：Toolformer的工具调用是瞬时的，不维护跨会话的长期记忆。同一用户在不同会话中重复提问时，模型无法访问之前的交互历史。

这些局限直接催生了后续研究：RAG提供了知识持久化，长期记忆系统提供了状态维护，而层次化规划（见9.5节）解决了多步组合问题。

## 9.3 RAG：检索增强生成（向量数据库、Chunk策略、重排序）

RAG系统的核心架构包含三个组件：文档编码器（将知识库转换为向量）、检索器（根据查询向量召回相关文档片段）、生成器（基于检索结果生成回答）。这一架构的工业落地依赖于向量数据库、文本分块（chunking）策略与重排序（reranking）的协同优化。

**向量数据库与Embedding模型**。RAG系统的检索质量首先取决于文档与查询的向量表示。早期RAG使用DPR（Dense Passage Retrieval）的BERT-base编码器，将文档与查询编码为768维向量，通过内积相似度排序。2023年后，sentence-transformers库中的`all-MiniLM-L6-v2`（384维，轻量）与`all-mpnet-base-v2`（768维，高精度）成为工业标准。对于中文场景，则常用`BAAI/bge-large-zh`或`BAAI/bge-m3`。向量数据库的选择涉及精度与延迟的权衡：FAISS（Facebook AI Similarity Search）的IVF-PQ索引在百万级文档上提供毫秒级查询，Milvus支持分布式部署与混合检索（向量+关键词），而Chroma与Qdrant则更适合轻量级应用。

Embedding模型的一个隐性假设是：文档与查询应处于同一语义空间。如果文档是技术手册而查询是口语化问题，二者的向量分布可能不一致，导致检索失败。解决方案包括使用领域特定数据微调embedding模型，或采用查询扩展（query expansion）技术——用LLM生成多个查询变体，分别检索后合并结果。

**Chunk策略**。将文档切分为可嵌入的片段是RAG中最具工程敏感性的决策。Chunk过小（如128 token）会丢失跨句子的上下文依赖；Chunk过大（如2048 token）会稀释关键信息，降低检索精度。常见的策略包括：
- **固定长度滑动窗口**：以固定token数（如512）切分，重叠率20%-50%。实现简单，但会在句子边界处切断语义单元。
- **递归字符分割**：先按段落分割，若段落过长则按句子分割，若句子仍过长则按固定长度分割。LangChain的`RecursiveCharacterTextSplitter`实现了这一策略。
- **语义分割**：用模型检测主题转换点，在语义边界处切分。计算成本更高，但检索精度通常提升5%-15%。
- **代理增强分割**：用LLM直接读取文档并决定最佳切分点。成本最高，适用于高质量要求的场景。

Chunk大小与embedding模型的最大输入长度直接相关。若使用`all-MiniLM-L6-v2`（最大256 token），chunk应控制在256 token以内；若使用支持8192 token的模型（如BGE-M3），则可以采用较大的chunk以保留更多上下文。一个常被忽视的问题是：chunk的元数据（metadata）同样重要。在chunk向量中附加文档标题、章节名、时间戳等信息，能显著提升过滤与排序效果。

**重排序（Reranking）**。向量检索基于余弦相似度或内积，其本质是对比查询向量与文档向量的全局相似性。这种全局度量可能召回语义相关但信息价值低的文档。例如，查询"Transformer的注意力机制复杂度"可能召回一篇介绍Transformer整体架构的综述，而非专门讨论复杂度优化的论文。重排序器（如ColBERT、BGE-Reranker、Cross-Encoder）在检索阶段后执行精排：将查询与每个候选文档拼接，通过一个交叉注意力模型计算相关性分数。ColBERT采用延迟交互（late interaction）机制，分别编码查询与文档，然后在token级别计算最大相似度，兼顾效率与精度。实验表明，在Top-5检索结果上添加重排序，可将问答准确率提升8%-20%，而延迟增加通常不超过50ms。

**RAG与Toolformer的互补性**。RAG将外部知识以隐式方式注入模型（模型看到的是拼接后的文本），适合处理事实性查询；Toolformer生成显式的API调用，适合处理计算、翻译、实时数据获取等任务。现代Agent系统（如AutoGen [Wu et al., 2023]）通常同时集成二者：先通过RAG检索领域知识，再通过工具调用获取实时信息，最终将多源信息整合到统一的推理链中。

## 9.4 长期记忆：感知记忆、工作记忆、长期记忆的架构映射

认知心理学将人类记忆系统分为三层：感知记忆（sensory memory，毫秒级，容量极大但几乎不进入意识）、工作记忆（working memory，秒级，容量4±1个组块）、长期记忆（long-term memory，终身，容量几乎无限）。LLM Agent研究者将这一三层架构映射到计算系统，形成了今日Agent记忆设计的理论框架 [Du, 2026]。

**感知记忆（Sensory Memory）**。在Agent系统中，感知记忆对应于原始环境输入的短暂缓存。对于文本Agent，这是用户输入的原始token流；对于多模态Agent，这是图像特征、音频波形或传感器读数的原始表示。感知记忆的关键需求是低延迟与无损存储，通常直接流入工作记忆而不做复杂处理。与人类的视觉暂留（iconic memory）类似，Agent的感知记忆只保留最近数秒至数分钟的原始输入，过期后丢弃或压缩。

**工作记忆（Working Memory）**。工作记忆对应于LLM的上下文窗口（context window）。当前GPT-4支持128k token，Claude 3支持200k token，Gemini 1.5支持1M token。上下文窗口是Agent进行in-context learning、推理与决策的"思维工作台"。然而，工作记忆存在三个硬性约束：
1. **容量约束**：即使1M token窗口也无法容纳一本书或整个代码库。
2. **衰减约束**：Transformer的注意力机制对所有位置一视同仁，但实践中，中间位置的信息容易被"Lost in the Middle" [Liu et al., 2023]——模型对上下文中间部分的检索显著弱于首尾部分。
3. **成本约束**：上下文长度与推理计算量呈二次或线性关系（取决于注意力实现），长上下文意味着高延迟与高费用。

工作记忆的管理策略包括：滑动窗口（只保留最近K轮对话）、摘要压缩（用LLM将早期对话压缩为摘要）、层次化索引（将历史对话按主题聚类，按需检索加载）。

**长期记忆（Long-term Memory）**。长期记忆是Agent的持久化知识存储，通常分为两类：
- **语义记忆（Semantic Memory）**：存储事实、概念与关系。在Agent中通常由向量数据库实现，通过RAG机制按需检索到工作记忆。
- **情景记忆（Episodic Memory）**：存储事件序列与时间线。例如，用户上周要求"用Python写一个快速排序"，昨天要求"优化这个排序的性能"——这些交互事件按时间顺序存储，支持时间线查询（"用户上周让我做了什么？"）。情景记忆通常用图数据库（如Neo4j）或时间序列数据库（如TimescaleDB）实现，节点为事件，边为时间关系或因果关系。

**记忆架构的工程实现**。一个完整的Agent记忆系统通常采用分层缓存策略：
1. **L1缓存**：工作记忆（上下文窗口），纳秒级访问，容量最小。
2. **L2缓存**：近期对话摘要（存储在内存或Redis中），毫秒级访问，容量数千token。
3. **L3存储**：向量数据库（语义记忆），毫秒至秒级访问，容量百万级文档。
4. **L4存储**：图数据库或关系数据库（情景记忆与符号知识），秒级访问，容量无上限。

当Agent需要回答用户问题时，系统首先查询L1（工作记忆中的当前上下文），若信息不足则检索L2（近期摘要），仍不足则检索L3（向量数据库），最后查询L4（结构化知识库）。这种分层检索与计算机体系结构中的缓存层次（L1/L2/L3/RAM）完全一致，只是在语义层面操作。

**记忆的写入与遗忘**。人类记忆不是简单的"存储-读取"系统，而是经过编码、巩固、遗忘的主动过程。Agent记忆同样需要遗忘策略：向量数据库中的旧知识可能已被新知识取代（例如，软件文档的版本更新），需要定期清理或标记失效；情景记忆中的大量无关事件会污染检索结果，需要基于重要性（如用户是否明确反馈"这很重要"）或时效性（如超过30天的事件）进行衰减。一种简单的实现是：为每个记忆条目维护一个"显著性分数"，每次检索命中时增加，长期未命中时按指数衰减，低于阈值时删除或归档。

## 9.5 规划：从ReAct单步到层次化任务分解

第8章详细介绍了ReAct [Yao et al., 2022] 框架：语言模型通过交替生成推理轨迹（Thought）与行动（Action），在推理与行动的交织中解决任务。ReAct的成功验证了"显式推理链"对工具使用与问题求解的价值，但其原始设计主要针对单步或少数步的简单任务（如HotpotQA的多跳问答）。当任务复杂度上升——例如"为我规划一次日本七日游，包含航班预订、酒店选择、每日行程安排与预算控制"——ReAct的线性推理链面临组合爆炸。

**从单步到多步：工具链的组合**。ReAct的每个Action只调用一个工具，返回的Observation直接进入下一步推理。对于需要多个工具协同的任务（如先搜索航班、再查询酒店、再计算总价），ReAct依赖模型在每一步自主决定下一步工具。这种贪婪的单步决策缺乏全局规划视野，容易陷入局部最优。例如，模型可能先选择了便宜的航班（Action: 搜索航班 → Observation: 东京往返￥3000），然后在安排每日行程时发现该航班到达时间导致第一天无法游览，不得不回退。Kim et al. [2023] 提出的LLM Compiler将这一问题显式化：通过静态分析工具依赖图，将串行工具调用并行化。例如，航班搜索与酒店搜索之间无数据依赖，可以并行执行；而总价计算依赖于航班与酒店的结果，必须串行。这种"编译时优化"将ReAct的线性链转换为DAG（有向无环图）执行计划，显著降低了多步任务的延迟。

**层次化任务分解（Hierarchical Task Decomposition）**。当任务复杂度超过一定阈值时，人类会采用"分而治之"策略：先制定高层计划（"去日本→关东→东京→浅草寺"），再逐步细化每个子计划。LLM Agent的层次化规划遵循同一原则。Wei et al. [2026] 提出的"Planner-Centric Framework"在ReAct之上引入了一个显式的规划器（Planner）模块：规划器接收高层目标，生成一个由子任务组成的计划树（plan tree），每个子任务分配给专门的执行器（Executor）。执行器可以是一个独立的LLM实例、一个工具调用模块或一个子Agent。规划器持续监控执行器的反馈，当检测到偏差（如航班预订失败）时，触发重规划（replanning）。这种架构与机器人领域的HTN（Hierarchical Task Network）规划器一脉相承，但用LLM替代了手工编写的任务分解规则。

**反思与重规划（Reflexion & Replanning）**。层次化规划的前提是规划器能够识别执行失败。ReAct原始框架中，模型一旦开始执行，只能沿着单一路径前进，没有显式的"反思"步骤。Shinn et al. [2023] 的Reflexion框架在执行链的末端添加了一个评估器（Evaluator）：评估器用LLM判断最终结果是否正确，若错误则生成反思文本（"我在第二步错误地使用了加法而非乘法"），将反思注入工作记忆，然后重新执行。这种自我纠正机制使Agent在算术、编程等任务上的成功率提升15%-30%。在更复杂的层次化架构中，反思不仅发生在任务末端，还发生在每个子任务完成后：每个子Agent返回结果时，父级规划器先验证结果合理性，再决定是否接受、修正或重新分配。

**规划的表示形式**。ReAct使用自然语言文本作为推理轨迹的表示，这带来了灵活性（任何任务都可以用自然语言描述）但也带来了不精确性（自然语言无法表达严格的约束关系，如"酒店必须在机场附近 AND 预算低于￥500/晚"）。替代方案包括：
- **符号规划（Symbolic Planning）**：用PDDL（Planning Domain Definition Language）或类似形式语言表示状态、动作与目标。LLM负责将自然语言目标翻译为PDDL问题描述，然后调用传统规划器（如FastDownward）求解。这种混合架构在严格约束场景（如机器人任务规划）中更可靠。
- **程序表示（Program Representation）**：用Python代码或伪代码表示计划。例如，"Plan: flights = search_flights('NRT'); hotels = search_hotels(flights.arrival); itinerary = plan_days(hotels)"这种代码式表示可以直接由LLM生成，然后由解释器执行。Toolformer的API调用语法本质上就是一种受限的程序表示。

现代工业级Agent系统（如LangChain的Plan-and-Execute、AutoGPT的连续任务模式）通常混合使用上述表示：高层用自然语言ReAct进行灵活推理，中层用DAG或代码表示工具依赖，低层用传统规划器处理严格约束。

## 9.6 代码：带向量数据库的RAG系统与长期记忆存储实现

本节提供一个可运行的PyTorch实现，涵盖：(1) 基于FAISS的向量检索（若FAISS未安装则回退到纯PyTorch实现）；(2) 文档分块与Embedding；(3) 重排序；(4) 分层记忆存储（工作记忆+向量语义记忆+情景时间线记忆）。

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
        # Make it slightly text-dependent so same text -> same-ish vector
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

上述代码展示了一个完整的分层RAG与记忆系统。核心设计决策如下：

1. **向量存储的抽象**：`VectorStore`类封装了FAISS与纯PyTorch两种后端。FAISS的`IndexFlatIP`在归一化向量上等价于cosine similarity，但速度更快；当FAISS不可用时，纯PyTorch实现确保了可运行性。生产环境中应使用FAISS的IVF（Inverted File Index）或HNSW（Hierarchical Navigable Small World）索引以处理百万级以上的向量。

2. **分块策略的对比**：`fixed_chunk`实现简单但可能切断语义边界；`recursive_chunk`优先按段落、再按句子、最后按固定长度分割，在实际RAG系统中更为常用。实验数据表明，对于问答任务，递归分割在NQ数据集上的Top-5检索精度通常比固定分割高3-8个百分点。

3. **分层记忆的检索顺序**：`retrieve()`方法按L1→L3→L4的顺序搜索，遵循"越热的数据越先访问"原则。`build_prompt_context()`将检索结果按此顺序拼接，然后用朴素的token计数截断。生产环境中应使用tiktoken或transformers的tokenizer进行精确截断，避免在token边界处切断多字节字符。

4. **重排序的模拟**：`Reranker`类是一个简化示例，实际生产环境应使用`BAAI/bge-reranker-large`或`cross-encoder/ms-marco-MiniLM-L-6-v2`等预训练交叉编码器。交叉编码器将query与document拼接后输入Transformer，计算相关性分数，其精度远高于双编码器（bi-encoder）的点积相似度，但计算成本高一个数量级。

## 9.7 工程实践框

**Chunk大小选择**。Chunk大小没有普适最优值，它取决于三个因素：embedding模型的最大输入长度、文档的平均信息密度、以及下游任务的粒度需求。对于事实性问答（如"某某公司的CEO是谁"），128-256 token的chunk通常最优，因为这类问题的答案往往集中在单个句子中；对于总结性任务（如"总结这篇论文的贡献"），512-1024 token的chunk能保留更多上下文。一个实用的调试方法是：在开发集上运行检索，检查正确答案是否落在Top-3 chunk中，若频繁出现"正确答案被切分在相邻chunk边界"的情况，则增加overlap比例或改用语义分割。

**检索失败模式与诊断**。RAG系统在生产环境中最常见的失败模式不是生成错误，而是检索错误。具体表现包括：
- **语义漂移**：用户用口语化查询（"怎么让模型跑得更快"）检索技术文档，embedding模型无法将口语映射到正式术语（"inference optimization"）。解决：查询扩展（query expansion）或领域微调embedding模型。
- **信息碎片化**：答案分散在多个chunk中，单个chunk无法回答完整问题。解决：在检索阶段使用parent-child chunking（小块用于检索，大块用于生成），或在生成阶段用多轮检索（iterative retrieval）。
- **陈旧知识**：向量数据库中的文档已过时。解决：为每个chunk附加时间戳metadata，在检索时优先返回近期文档；或维护一个"失效检测"管道，定期用LLM判断文档是否仍有效。
- **假阳性高排名**：检索返回的chunk在语义上与查询相关，但不包含实际答案。解决：引入重排序器，或训练一个专门的"答案存在性"分类器过滤检索结果。

**记忆遗忘策略**。长期记忆系统的容量不是无限的，必须设计遗忘机制。推荐的分层策略：
- **工作记忆（L1）**：无遗忘，由上下文窗口硬截断。
- **近期摘要（L2）**：基于token数量的滑动窗口，超出部分用LLM压缩为摘要。压缩提示（compression prompt）应明确指示"保留关键事实、丢弃闲聊"。
- **语义记忆（L3）**：按"显著性分数"指数衰减。每次用户查询命中某chunk时，该chunk分数增加；长期未命中则衰减。衰减公式：`score_new = score_old * exp(-Δt / τ)`，其中τ为时间常数（例如30天）。分数低于阈值的chunk可归档到冷存储。
- **情景记忆（L4）**：按时间衰减与重要性双重过滤。用户标记为"重要"的事件（如"记住我的项目代号是Phoenix"）应永久保存；日常闲聊类事件（如"今天天气不错"）在24小时后衰减至零并删除。

**硬件需求估算**。RAG系统的硬件成本主要集中在三个环节：
1. **Embedding计算**：100万文档 × 512 token/文档，使用`all-MiniLM-L6-v2`（22M参数）在V100 GPU上约需2小时；在CPU上约需20小时。若使用更大的`BAAI/bge-large-en`（335M参数），GPU时间约10小时。
2. **向量存储**：100万条768维float32向量占用约`1e6 × 768 × 4 ≈ 3 GB`内存。FAISS的IVF索引额外增加约20%开销。若使用HNSW索引，内存开销可能翻倍。
3. **生成阶段**：检索本身只贡献毫秒级延迟；真正瓶颈是LLM生成。一个128k上下文、4k输出的请求在GPT-4级别模型上成本约为$3-5，延迟约30-60秒。优化方向：通过摘要与过滤将实际输入上下文压缩到10k token以内，可降低成本约80%。

**超参选择速查表**：

| 组件 | 超参数 | 推荐值 | 调整信号 |
|------|--------|--------|----------|
| Chunking | chunk_size | 256-512 | 答案碎片化 → 增大；噪声混入 → 减小 |
| Chunking | overlap | 10%-20% | 边界切断 → 增大；冗余重复 → 减小 |
| Retrieval | top_k | 5-10 | 召回不足 → 增大；噪声过多 → 减小 |
| Reranking | rerank_top_k | 3-5 | 精度不足 → 增大；延迟过高 → 减小 |
| Memory | working_memory_len | 6-10轮 | 上下文遗忘 → 增大；成本过高 → 减小 |
| Memory | decay_τ | 7-30天 | 记忆丢失过快 → 增大；存储膨胀 → 减小 |

## 9.8 小结

本章从Toolformer的自监督工具调用学习出发，经过RAG的检索增强架构，延伸至长期记忆的分层存储，最终在规划层次上讨论了ReAct的扩展与层次化任务分解。贯穿始终的主线是：语言模型的能力边界不仅取决于参数规模，更取决于它与外部系统的接口设计——工具是能力的扩展，检索是知识的延伸，记忆是状态的持久化，而规划是这些组件的协调器。

Toolformer证明了模型可以在预训练阶段内化工具调用语法，但其单步、闭集、无状态的局限推动了后续研究。RAG将外部知识解耦到向量数据库，实现了知识的动态更新，但检索质量严重依赖于chunk策略、embedding模型与重排序器的协同优化。长期记忆系统通过认知心理学的三层映射（感知→工作→长期），为Agent提供了跨会话的持续性，但记忆的写入、检索与遗忘策略仍是活跃的研究方向。规划模块从ReAct的单步推理扩展到层次化任务分解与DAG并行执行，反映了Agent系统从简单问答工具向复杂工作流自动化平台的演进。

这些组件并非独立存在。一个工业级Agent系统的典型执行流程是：用户输入进入感知记忆与工作记忆；规划器分解任务并决定是否需要检索或工具调用；检索器从向量数据库召回相关知识；工具执行器调用外部API；所有观察结果（Observation）被重新注入工作记忆；最终生成器综合上下文输出回答，并将本轮交互写入长期记忆。这一循环——感知、推理、行动、记忆——构成了现代Agent系统的认知架构基础。

---

## 延伸阅读

[Guu et al., 2020] K Guu, K Lee, Z Tung, P Pasupat, M Chang. "Retrieval augmented language model pre-training." *Proceedings of the 37th International Conference on Machine Learning (ICML)*, 2020. 该论文提出了REALM，首次在预训练阶段联合优化语言模型与稠密检索器，是RAG方向的前置奠基工作。

[Yao et al., 2022] S Yao, J Zhao, D Yu, N Du, I Shafran, K Narasimhan, et al. "ReAct: Synergizing reasoning and acting in language models." arXiv:2210.03629, 2022. 该论文提出了推理与行动交替执行的ReAct框架，为后续工具使用与规划系统提供了基础架构（详见第8章）。

[Qin et al., 2024] Y Qin, S Liang, Y Ye, K Zhu, L Yan, Y Lu, et al. "ToolLLM: Facilitating large language models to master 16000+ real-world APIs." *International Conference on Learning Representations (ICLR)*, 2024. 该论文将Toolformer的闭集工具扩展到16,000+真实API，通过ToolBench数据集与微调策略，解决了模型泛化到新工具的问题。

[Du, 2026] P Du. "Memory for autonomous LLM agents: Mechanisms, evaluation, and emerging frontiers." arXiv:2603.07670, 2026. 该综述系统性地梳理了LLM Agent记忆系统的架构设计、评估基准与新兴研究方向，是长期记忆领域最全面的综述文献。

[Kim et al., 2023] S Kim, S Moon, R Tabrizi, N Lee, MW Mahoney, et al. "An LLM compiler for parallel function calling." arXiv:2312.04511, 2023. 该论文提出将LLM工具调用从线性链编译为DAG并行执行图，显著降低了多步任务的推理延迟，是工具使用效率优化的代表性工作。
