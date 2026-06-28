# 第8章 从 ReAct 到 AutoGen——Agent 架构的范式演进

> **本章动机（养老机器人视角）**
> 机器人需要回答"外婆的血糖记录在哪儿、今天超标了多少、需不需要通知家人"——这是一个三步任务，每步都依赖上一步的结果。
> ReAct 让 LLM 自主编排"查数据 → 推理 → 决定行动"的循环，而不是在单个 prompt 里把所有步骤塞给模型。但 ReAct 真的创新在哪？它不是"让 LLM 调用工具"这么简单——Toolformer 已经做到了。ReAct 的突破在于把"推理"与"行动"放进**同一个自回归循环**，让模型在生成行动之前先"说出来"它想做什么。这个看似微小的设计，为什么会产生如此大的影响？它又有哪些被低估的局限？本章将带你回到2022年的决策情境，还原这个选择背后的约束、候选和权衡。

---

## 核心思想

Agent 的本质不是"调用工具"，而是"在开放环境中维持一个推理—行动—观察的闭环"。ReAct 将这一闭环内化为单模型的交错生成，AutoGen 则将其外化为多模型间的对话协议。理解二者之间的张力与互补，是设计可扩展 Agent 系统的先决条件。

但这一核心思想本身需要被审视：**"推理—行动—观察的闭环"是 Agent 的本质，还是我们对 Agent 的一种隐喻？** [开放问题]——这个问题没有确定答案。从控制论角度看，闭环反馈是任何智能系统的基本结构；但从认知科学角度看，人类的很多行为是"开环"的（如习惯性动作），不需要显式的推理-行动循环。本章将始终区分"工程上有效"和"原理上必然"两个层次，避免把工程范式当作认知真理。

---

## 历史脉络：为什么 Agent 研究在2022年被迫转向？

20世纪80年代的符号 AI（SOAR、ACT-R）已经尝试用"认知循环"来统一推理与行动，但受限于手工规则与状态空间爆炸，从未走出实验室 [已证明]。2010年代的深度强化学习（DQN、A3C）用神经网络逼近价值函数，却在稀疏奖励与长程信用分配问题上反复碰壁 [强实验证据]。2022年，大语言模型（LLM）的涌现能力让研究者重新发现：预训练模型内部已经压缩了关于世界的大量常识，问题不在于"如何学习世界模型"，而在于"如何提示模型把知识显式地展开为推理轨迹"。

Chain-of-Thought [Wei et al., 2022] 率先展示了中间推理步骤对算术与符号推理的提升，但它只生成文本，不与环境交互。Toolformer [Schick et al., 2023]（详见第7章）将 API 调用嵌入自监督预训练，却受限于"只能生成一个调用且无法根据返回结果再推理"的单向范式。ReAct [Yao et al., 2022] 在 CoT 与工具调用之间架起了桥梁：它让 LLM 以自然语言交替生成 Thought（推理）与 Action（行动），并把 Observation（环境反馈）重新注入上下文，形成完整的状态机。随后，研究社区意识到复杂任务往往需要多个角色协作，AutoGen [Wu et al., 2023] 由此提出以"对话"作为多 Agent 编排的原语，把 ReAct 的单 Agent 循环扩展为群聊式的分布式控制。与此同时，Plan-and-Solve 路线质疑 ReAct 的逐步贪心策略，主张在行动前先做全局规划；而工具调用编译器（如 LLM Compiler [Kim et al., 2023]）则尝试将串行的 ReAct 轨迹编译为并行 DAG。

本章的任务，是把这些看似碎片化的工作还原为一条清晰的架构演进线索，并给出可运行的工程实现。但请记住：**演进不等于进步**。ReAct 到 AutoGen 的"演进"是否真的是一种认知能力的提升，还是仅仅是工程复杂度的叠加？[开放问题]——这个问题将在本章末尾的批判性分析中重新讨论。

---

## 8.1 从符号 AI 到 LLM Agent：三次范式转移

### 8.1.1 符号 Agent：推理与行动的分离

#### 问题本质
符号 Agent 要解决的根本问题是：**一个智能系统如何在不从数据中学习的情况下，通过预编码的规则完成推理和行动？** 这个问题的答案在1980年代看起来是可行的——因为当时的任务环境（如积木世界）足够简单，手工规则可以覆盖所有情况。

#### 历史约束：当时的世界是什么样的？

- **计算约束**：1980年代的计算机内存以 KB 计，CPU 主频在 MHz 级别。任何需要大规模矩阵运算的方法（如神经网络）在当时的硬件上不可行。SOAR 和 ACT-R 选择符号规则不是"偏好"，而是**唯一可行**的路径。
- **理论约束**：1980年代是符号主义 AI 的黄金期。非确定性推理的数学框架（贝叶斯网络）尚未成熟，连接主义（神经网络）在1986年反向传播被重新发现后才缓慢复苏。当时的 AI 研究者接受的是逻辑学训练，而非概率论训练——**社区惯性**使得符号规则成为默认选择。
- **数据约束**：不存在大规模标注数据集。专家系统（如 MYCIN）的知识来源于人类专家的访谈记录，获取成本极高。

#### 方案空间：在当时约束下，有哪些候选路径？

**路径A：产生式规则系统（SOAR, ACT-R）**
- 核心思想：用 if-then 规则表示知识，通过模式匹配触发推理和行动
- 在当时约束下的优势：规则可解释、可验证、不需要训练数据
- 在当时约束下的劣势：**知识获取瓶颈**——规则数量随领域复杂度指数增长；规则之间的冲突消解需要手工设计的元规则
- **反事实推演**：如果1980年代符号 AI 获得了自动化规则学习的能力（如从示例中归纳规则），短期（1-2年）内可能在特定领域（如医疗诊断）达到实用水平；长期（5-10年）来看，即使符号 AI 解决了知识获取问题，它仍然面临**符号接地问题**——符号如何与物理世界的感知信号建立对应关系？这个问题在2025年仍未完全解决。符号 AI 的思想在今天的"Neural-Symbolic"路线中以新形式复兴，说明它并非"错误方向"，而是"受限于时代"。

**路径B：连接主义 Agent（早期神经网络, 1986-2000）**
- 核心思想：用神经网络学习状态-行动映射，从数据中自动获取策略
- 在当时约束下的优势：可以从数据中学习，避免知识获取瓶颈
- 在当时约束下的劣势：梯度消失问题使深层网络无法训练；计算资源不足以训练大规模网络；缺乏有效的预训练方法
- **反事实推演**：如果反向传播在1986年后获得了足够的计算资源支持，连接主义 Agent 可能在1990年代就展现出实用价值。但当时的"AI冬天"导致研究资金枯竭，连接主义路线被迫蛰伏了15年。这提醒我们：**技术路线的发展不仅取决于技术本身，还取决于研究资金的周期和社区的信任波动**。

#### 选择逻辑
在1980年代的约束下，产生式规则系统在"可解释性"和"不需要训练数据"两个维度上最优，但牺牲了"可扩展性"和"学习能力"。这不是"错误选择"，而是**在约束空间中的理性优化**。

#### 认知偏差警示
符号 AI 的研究者可能陷入了**工具偏差**——因为接受的是逻辑学训练，所以把所有智能问题都建模为逻辑推理问题。这种偏差导致整个领域忽视了感知和学习的中心地位，直到连接主义复兴才被迫修正方向。

### 8.1.2 深度强化学习：隐式世界模型

#### 问题本质
深度强化学习要解决的根本问题是：**一个智能系统如何在不依赖人工规则的情况下，通过与环境交互自动学习最优策略？**

#### 历史约束：当时的世界是什么样的？

- **计算约束**：2013年 DQN 的训练需要 GPU 集群，这在2013年刚成为可能（感谢游戏 GPU 产业的推动）。但训练一个 Atari Agent 仍然需要数千万帧的游戏画面——这在真实物理环境中不可承受。
- **数据约束**：强化学习不需要标注数据，但需要大量的**环境交互**。在模拟环境中（如 Atari 游戏模拟器），每秒可以生成数千次交互；在真实环境中（如机器人），每秒可能只有1-2次交互。
- **理论约束**：2015年的模型-free RL（DQN, A3C）在理论上是近似贝尔曼最优方程，但实践中的样本效率极低。基于模型的 RL（如 PILCO）在理论上更高效，但2015年的神经网络无法准确建模高维环境的动力学。

#### 方案空间

**路径A：无模型强化学习（DQN, PPO）**
- 核心思想：不学习环境模型，直接从交互中估计状态-行动价值函数
- 优势：不需要环境模型，可以处理高维输入
- 劣势：样本效率极低；学到的策略是黑箱
- **反事实推演**：如果无模型 RL 在2015年后解决了样本效率问题（如通过元学习），它可能在机器人控制领域直接达到实用水平，LLM Agent 路线可能不会在2022年"趁虚而入"。但截至目前，无模型 RL 的样本效率仍然是1-2个数量级低于人类水平 [强实验证据]。

**路径B：基于模型的强化学习（Dreamer, MuZero）**
- 核心思想：先学习环境动力学模型，然后在模型中"想象"训练策略
- 优势：样本效率高（在模型准确的情况下）
- 劣势：模型误差复合传播；难以迁移到开放文本领域
- **关键洞察**：Dreamer 和 MuZero 在连续控制和棋盘游戏上表现出色 [强实验证据]，但它们的状态空间是**结构化的**（像素网格、棋盘状态）。当状态空间变成开放文本时，潜在动力学模型无法有效工作——这是 LLM Agent 路线出现的根本原因。

#### 选择逻辑
在2015年的约束下，无模型 RL 在"可处理性"维度上最优（不需要建模复杂环境），但牺牲了"样本效率"和"可解释性"。这个选择在游戏环境中是合理的（因为交互成本低），但在真实物理环境中导致了瓶颈。

#### 认知偏差警示
DeepMind 在2015-2017年的一系列工作（DQN, AlphaGo）在媒体上获得了巨大成功，这可能导致社区过度乐观地认为"无模型 RL 可以解决一切" [合理推断]。将无模型 RL 从游戏推广到真实世界，是一个在当时被过度泛化的典型案例——Atari 游戏和围棋是**完全可观测、确定性、奖励信号明确**的环境，真实世界则完全不同。

### 8.1.3 LLM Agent：推理即行动，语言即接口

#### 问题本质
LLM Agent 要解决的根本问题是：**如何让一个已经拥有大量世界知识的语言模型，在需要时动态地调用外部工具来获取信息或执行操作？**

#### 历史约束：2022年的世界是什么样的？

- **LLM 能力约束**：GPT-3（2020）和 GPT-3.5（2022）已经展示了 few-shot 学习能力。但 LLM 的知识是**静态的**——训练数据截止后，模型不知道新信息。如何让 LLM 动态获取实时信息？
- **工具生态约束**：2022年，API 经济已经成熟（搜索、计算、代码执行），但将工具编排成任务流需要专门的工程师。没有标准化的"LLM-工具接口"。
- **社区需求约束**：研究者迫切需要一种方法，让 LLM 能够利用外部知识，而不需要为每个任务重新训练模型。微调成本高、周期长，prompt engineering 是更轻量的方案。

#### 方案空间：在当时约束下，有哪些候选路径？

**路径A：硬编码工具链（如 LangChain 早期版本）**
- 核心思想：工程师用代码定义工具调用顺序，LLM 只负责填充参数
- 在当时约束下的优势：可控、可预测、可调试
- 在当时约束下的劣势：无法处理未预见的任务变体；任务复杂度增加时，代码量指数增长
- **反事实推演**：如果硬编码工具链在2022年后继续发展，我们可能会拥有庞大的"工具编排库"。但长期来看，硬编码方法无法应对**开放性任务**（如"帮我规划一个考虑天气、交通和个人偏好的旅行"），因为这类任务的可能性空间是组合爆炸的。硬编码工具链在今天仍然适用于流程固定的企业场景（如 ETL 管道），这再次说明"被替代"的技术并非"错误"，而是"适用范围不同"。

**路径B：程序合成（Program Synthesis, 如 Codex/2021）**
- 核心思想：让 LLM 生成可执行的 Python 代码来调用工具
- 在当时约束下的优势：代码是形式化的，可以被解释器精确执行；可以表达复杂的控制流（循环、条件、递归）
- 在当时约束下的劣势：代码生成的错误率较高；生成的代码可能包含逻辑错误、无限循环或安全漏洞；调试成本高
- **反事实推演**：如果程序合成路线在2022年获得了更多的安全性和可靠性工程投入，它可能在**结构化任务**（如数据分析、自动化测试）上比 ReAct 更高效——因为代码可以批量执行，而 ReAct 是逐步串行的。但程序合成的局限在于：它要求 LLM 在一次生成中完成全部规划，无法根据中间结果动态调整。这在**高度不确定**的任务（如网页搜索）上是致命缺陷。今天的 Code Interpreter（如 GPT-4 的）可以看作是程序合成路线的延续，它在数据分析任务上确实优于 ReAct，但在开放问答上仍需要 ReAct 式的交互循环。

**路径C：Chain-of-Thought + 工具调用（ReAct, 2022）**
- 核心思想：让 LLM 在文本推理链中穿插工具调用（Action）和观察结果（Observation），形成 Thought-Action-Observation 闭环
- 在当时约束下的优势：不需要预定义工具链；LLM 可以根据中间结果调整策略；推理过程可解释
- 在当时约束下的劣势：推理基于文本模式，缺乏对物理后果的预测；工具调用错误可能累积

#### 选择逻辑：为什么 ReAct 成为2022年的主流选择？

ReAct 的流行不是因为它是"理论上最优的"，而是因为在**2022年的约束下**提供了最优的性价比 [强实验证据]：

1. **利用现有能力**：不需要训练新的模型，只需要改变 prompt 格式。这意味着任何拥有 GPT-3 API 的人都可以在五分钟内构建一个 ReAct Agent。
2. **可解释性**：推理链是文本，可以被人类检查。这与黑箱程序合成形成对比。
3. **灵活性**：不需要为每个新任务重写代码。这降低了实验成本，加速了研究迭代。
4. **动态调整**：与硬编码工具链不同，ReAct 可以根据 Observation 调整后续策略——这是处理不确定环境的关键能力。

但请注意：这四个优势都是**工程上的**，而非**原理上的**。ReAct 没有解决 LLM 缺乏物理因果推理的问题——它只是把问题从"Agent 怎么设计"转移到了"LLM 怎么推理"。

#### 认知偏差警示：ReAct 的"文本偏见"

ReAct 的研究者可能犯了**媒介偏差**（medium bias）——因为 LLM 处理的是文本，所以他们倾向于把一切都文本化。工具调用被设计为文本形式的 Action-Observation 循环，而不是更直接的数值信号（如力矩传感器读数）。这种文本化在高层任务（如信息检索）上有效，但在物理操作任务上引入了不必要的间接性 [合理推断]。一个机器人控制器可能需要直接读取关节角度，而不是等待 LLM 生成"我应该弯曲肘关节30度"这样的文本指令。

---

## 8.2 ReAct：推理与行动的交替循环

### 8.2.1 形式化定义

#### 先建立直觉：ReAct 到底在做什么？

在进入形式化之前，先用一个日常类比：你在陌生城市找一家餐厅。

- **纯 CoT**（只想不做）：你在脑中推理——"这家餐厅应该在市中心，因为 Yelp 上说靠近广场，那我可以走过去……"但你没有打开地图，所以你的推理基于过时或不准确的信息。
- **纯 Act**（只做不想）：你直接打开地图搜索"餐厅"，看到一个结果就去了——但你没想过要查营业时间，到了发现关门了。
- **ReAct**（边想边做）：你想"我应该先搜附近的餐厅"（Thought）→ 搜索（Action）→ 看到三个结果（Observation）→ 想"第二家评分最高但距离远，我需要确认它是否营业"（Thought）→ 查营业时间（Action）→ "营业到10点"（Observation）→ 决定去第二家。

ReAct 的核心洞见是：**推理和行动必须交错进行**。纯推理缺乏实时信息，纯行动缺乏整体规划。

#### 形式化：ReAct 作为 POMDP 的近似求解器

ReAct [Yao et al., 2022] 的核心可以形式化为一个部分可观察马尔可夫决策过程（POMDP）的近似求解器。在标准 POMDP 中，Agent 维护一个信念状态（belief state）$b_t(s) = P(s_t = s \mid a_{1:t}, o_{1:t})$，并通过贝叶斯更新来融合新观测。然而，对于基于 LLM 的 Agent，显式维护连续信念状态是不现实的——LLM 的上下文窗口本质上是一个**信息状态（information state）**的近似：它把全部历史轨迹 $h_t = (a_1, o_1, \dots, a_t, o_t)$ 以文本形式编码，让自回归模型直接从中学习策略，而无需显式推断后验分布。

令 $s_t$ 为第 $t$ 步的环境状态（对 LLM 不可直接观察），$o_t$ 为第 $t$ 步的观测（Observation，通常是工具返回的文本）。Agent 的"认知状态"由上下文 $c_t$ 表示：

$$c_t = [\text{system prompt}, \text{task}, \text{ Thought}_1, \text{Action}_1, \text{Observation}_1, \dots, \text{Thought}_t, \text{Action}_t, \text{Observation}_t]$$

LLM 策略 $\pi_\theta$ 以 $c_t$ 为条件，生成下一步的 Thought 与 Action：

$$(\text{Thought}_{t+1}, \text{Action}_{t+1}) \sim \pi_\theta(\cdot \mid c_t)$$

Action 被解析为工具调用 $\tau_{t+1}$，执行后得到 Observation $o_{t+1}$，从而更新上下文。循环终止条件是 Action 为 `Finish[answer]`。

从信息论角度看，ReAct 的上下文 $c_t$ 是一个**有损压缩**的信念状态 [合理推断]：它丢弃了数值概率，保留了自然语言描述的关键决策线索。这一近似的有效性依赖于预训练 LLM 的"世界知识"——模型已经在海量文本中见过类似的推理模式，因此能够从稀疏的文本历史中恢复出隐含的因果关系。

**但这个近似有代价**：信念状态 $b_t(s)$ 是一个概率分布，可以精确量化不确定性；而上下文 $c_t$ 是一段文本，不确定性被隐含在语言的模糊性中。当任务要求精确的概率推理时（如"有70%的概率下雨，是否需要带伞"），ReAct 的文本表征可能无法支持精确决策 [开放问题]。

### 8.2.2 为什么 Thought 必须在 Action 之前

#### 直觉解释
想象你在下棋。如果你每一步都直接落子（纯 Act），你可能在10步内就输掉——因为你没有思考后续变化。如果你一直在脑中推演但从不落子（纯 CoT），时间会耗尽。ReAct 的"先想后做"就像棋手的"计算→落子→观察对手反应→再计算"循环。

#### 实验证据
ReAct 的消融实验揭示了一个关键发现：如果去掉 Thought，只让模型生成 Action（即纯 Act 基线），成功率在 HotpotQA 和 WebShop 上分别下降 20% 与 30% 以上 [强实验证据]。如果去掉 Action，只保留纯 CoT（即模型只生成推理链，不调用工具），则无法获取实时信息，导致事实性错误率上升。

#### Thought 的三个功能
1. **分解**：将复杂目标拆解为子目标，例如"先搜索 A，再验证 B"。
2. **锚定**：在得到 Observation 后，解释该结果与当前任务的关系，避免后续步骤偏离主题。
3. **回溯**：当 Observation 与预期不符时，Thought 可以显式地声明"之前的假设是错误的，我需要改换策略"。

这种"推理作为行动的前置"机制，实际上让 LLM 在上下文窗口中维持了一个显式的计划栈（plan stack）[合理推断]，尽管该栈没有硬编码的数据结构，而是以自然语言的线性序列存在。

#### 认知偏差警示：Thought 的"自我合理化"陷阱
Thought 的回溯功能在理论上很优雅，但在实践中常常失效。当模型在第一步做出了错误选择后，后续的 Thought 并非如预期那样"客观分析 Observation 并调整策略"，而是倾向于**确认偏差**——在错误方向上不断"自我合理化" [强实验证据]。原始论文中的失败案例显示，在 WebShop 任务中，当模型过早选择了一个错误商品类别后，后续 5–6 步的 Thought 都在试图解释该选择的合理性，而非重新检索其他类别。这与人类的确认偏差惊人地相似——**LLM 在文本生成中再现了人类的认知偏差**，这是一个值得深入研究的发现 [开放问题]。

### 8.2.3 局限性与失败模式

ReAct 并非万能。以下四个局限不是"工程上可以修复的 bug"，而是**架构层面的原理性约束**。

#### 局限一：上下文线性增长

每一步 Thought + Action + Observation 都会追加到 prompt，当任务超过 10–15 步时，上下文窗口可能被历史占满，模型注意力分散。Yao et al. 的原始实现并未引入显式的记忆压缩机制，这成为后续工作（如 MemGPT [Packer et al., 2023]）的改进方向。

一个直观的数量级估计：假设每步平均消耗 300 token（100 token Thought + 50 token Action + 150 token Observation），10 步后 prompt 长度即达 3k token；如果系统提示与 few-shot 示例再占 2k token，则总 prompt 接近 5k token。对于 4k 上下文窗口的模型，这已构成硬约束；即使使用 16k 或 32k 窗口，**注意力稀释**（attention dilution）也会导致早期 Thought 被模型忽略 [强实验证据]。

**底层原因**：Transformer 的自注意力复杂度是 O(n²)，但注意力的"有效容量"并不随 n 线性增长。当上下文长度从 2k 增加到 32k 时，模型对早期 token 的注意力权重会衰减——这不是"实现缺陷"，而是 softmax 归一化的数学性质决定的 [已证明]。

#### 局限二：贪心局部最优

ReAct 的每一步决策都是基于当前上下文生成最优的下一步 Action，缺乏全局回溯。如果第一步搜索的关键词就偏离了答案所在的知识区域，后续 Thought 往往只是在错误方向上不断"自我合理化"，难以跳脱。这与传统规划中的 hill-climbing 困境类似。

**错误驱动学习：贪心策略的典型失败**

```
任务：查找"2023年诺贝尔物理学奖获得者"

Step 1:
Thought: 我需要搜索2023年诺贝尔物理学奖。
Action: Search[2023 Nobel Prize Physics]
Observation: "The 2023 Nobel Prize in Physics was awarded for experimental methods..."

Step 2:
Thought: 搜索结果提到了实验方法，我需要更具体地搜索获得者姓名。
Action: Search[2023 Nobel Prize experimental methods winners]
Observation: "Attosecond physics experiments..."

Step 3:
Thought: 结果提到了阿秒物理，但没有人名。让我搜索阿秒物理的先驱。
Action: Search[attosecond physics pioneers]
Observation: "Paul Corkum, Ferenc Krausz..."

Step 4:
Thought: 但这些人不一定是诺贝尔奖获得者。我需要回到原始搜索。
Action: Search[Nobel Prize 2023 physics laureates names]
Observation: "Pierre Agostini, Ferenc Krausz, Anne L'Huillier"

Step 5:
Thought: 终于找到了。答案是 Pierre Agostini, Ferenc Krausz, Anne L'Huillier。
Action: Finish[Pierre Agostini, Ferenc Krausz, Anne L'Huillier]
```

**分析**：Step 2 和 Step 3 是"自我合理化"的典型表现——模型在得到不完整的搜索结果后，没有回到原始任务重新规划搜索策略，而是顺着搜索结果的关键词"实验方法→阿秒物理→先驱"逐步偏离。直到 Step 4 才通过一个更直接的搜索回到正轨。这 2 步的浪费在简单任务中可以承受，但在 15 步的预算中，2 步的浪费意味着 13% 的效率损失。

#### 局限三：工具接口的脆弱性

ReAct 要求 Action 的格式可被正则表达式精确解析，例如 `Search[term]` 或 `Calculate[expr]`。一旦模型生成 `I will search for "term"` 这种非结构化文本，解析器就会失败，导致系统崩溃。这要求精心设计的 few-shot 示例与输出格式约束（如 JSON mode 或 regex 停止条件）。

更隐蔽的问题是**推理-行动错位**（reasoning-action misalignment）：模型在 Thought 中声称要执行 A，但在 Action 中却生成 B。例如 Thought 说"我需要搜索 2023 年的数据"，Action 却是 `Search[2024 data]`。这种不一致在高 temperature 或模型规模较小时尤为常见 [强实验证据]，后处理阶段无法捕获，因为 parser 只检查 Action 格式，不验证 Action 与 Thought 的一致性。

#### 局限四：成本敏感

ReAct 每步都需调用一次 LLM，而每次调用都是完整的自回归生成。对于需要 15 步才能解决的任务，总成本是 15 次 API 调用。如果问题可以通过一次批量查询（batch query）解决，ReAct 的逐步交互就显低效。这也是为什么 Kim et al. [2023] 提出 LLM Compiler 来压缩串行轨迹为并行 DAG。

**成本的数量级估计**（截至2025年中的参考价格）：
- GPT-4o：每步约 1k-2k prompt token + 200-500 completion token，单步成本约 $0.003-$0.015，15 步任务约 $0.045-$0.225
- Claude 3.5 Sonnet：单步成本约 $0.003-$0.009，15 步约 $0.045-$0.135
- 本地 Llama-3-70B（AWQ 4-bit）：需要 1×48GB GPU（如 A6000），单步推理延迟约 0.5-2 秒，无 token 费用但硬件成本约 $4,000-$8,000（2025年中二手价格）

---

## 8.3 AutoGen：多 Agent 对话框架

### 8.3.1 从单循环到对话图

#### 问题动机：为什么单 Agent 不够用？

当任务复杂度增加时，单 Agent 面临三个问题：
1. **上下文污染**：不同子任务的信息混在同一个上下文中，导致模型注意力分散
2. **角色混淆**：让同一个 LLM 在连续对话中既扮演架构师又扮演程序员，会导致指令冲突
3. **上下文窗口耗尽**：长任务的历史轨迹超出上下文窗口

AutoGen [Wu et al., 2023] 的出发点是：现实世界中的复杂任务天然适合分而治之。与其让一个 Agent 扮演"万能专家"，不如让多个 Agent 分别承担规划者、执行者、代码审查员、测试员的角色，通过结构化对话完成协作。

#### 历史约束：2023年的世界是什么样的？

- **LLM 成本约束**：2023年 GPT-4 的 API 成本仍然较高（$30/1M output token），多 Agent 系统的成本倍增效应显著。4 个 Agent 的群聊成本是单 Agent 的 4-6 倍。
- **编排工具约束**：2023年 LangChain 虽然提供了工具编排能力，但缺乏对多角色协作的原生支持。研究者需要自己实现消息路由、状态同步和终止条件。
- **社区认知约束**：2023年初，"多 Agent"概念在 LLM 社区还不普及。人们倾向于把所有功能塞进一个 prompt，而不是分拆为多个角色。

#### 方案空间：在当时约束下，有哪些候选路径？

**路径A：单 Agent + 长上下文（如 GPT-4-32K）**
- 核心思想：用更长的上下文窗口容纳所有任务信息，避免多 Agent 的编排开销
- 优势：无编排开销；无消息传递延迟；实现简单
- 劣势：上下文越长，注意力越稀释；不同任务的 prompt 互相干扰；单点故障
- **反事实推演**：如果2023年的 LLM 上下文窗口已经达到 1M token（如2024年的 Gemini 1.5 Pro），单 Agent 方案可能在很多任务上就够了，AutoGen 的价值会大幅降低。但即使有 1M 窗口，**角色混淆**和**注意力稀释**问题仍然存在——上下文长度不等于注意力质量 [强实验证据]。

**路径B：管道式编排（Pipeline, 如 LangChain Chain）**
- 核心思想：Agent 串行执行，前一个的输出是后一个的输入
- 优势：流程清晰；易于调试；无路由开销
- 劣势：无法处理需要迭代反馈的任务；一个 Agent 的错误会传播到下游所有 Agent
- **反事实推演**：管道式编排适合**线性任务**（如"翻译→摘要→格式化"），但对需要**双向反馈**的任务（如"写代码→测试→修改→再测试"）不适用。AutoGen 的对话图可以包含循环，这正是管道式编排无法表达的。

**路径C：多 Agent 对话（AutoGen）**
- 核心思想：每个 Agent 是独立的对话实体，通过结构化对话进行协作
- 优势：角色分离；支持双向反馈；可持久化对话历史
- 劣势：编排复杂；路由开销；状态同步挑战

#### 选择逻辑
在2023年的约束下，AutoGen 在"灵活性"和"可扩展性"维度上最优，但牺牲了"简洁性"和"成本效率"。这个选择适合**研究探索**和**原型开发**，但在**生产环境**中，管道式编排可能更经济 [合理推断]。

#### 认知偏差警示
AutoGen 的多 Agent 范式可能引发**复杂度偏见**（complexity bias）——因为系统看起来更"高级"、更"像人类团队"，所以认为它一定更好。但多 Agent 系统的复杂度是指数级增长的（每增加一个 Agent，可能的交互路径增加 O(n²)），而性能提升往往是线性的。**更多 Agent ≠ 更好的结果** [强实验证据]。

### 8.3.2 AutoGen 的核心抽象

AutoGen 把每个 Agent 建模为一个三元组：

- **LLM 后端**：决定该 Agent 的"智商"与知识域（可以是 GPT-4、Claude 或本地微调模型）。
- **系统提示（System Message）**：定义角色、约束与技能边界。
- **工具集（Tools）**：该 Agent 可访问的函数签名。

多个 Agent 通过"对话"进行协作。对话不再是自由闲聊，而是带有明确路由规则的**对话图（Conversation Graph）**。最简单的图是双向对话：用户 Agent 提出需求，助手 Agent 生成回复；当助手需要调用代码执行器时，对话路由到代码执行 Agent，执行结果再返回给助手。更复杂的图是群聊（Group Chat）：一个群聊管理器（Group Chat Manager）根据当前轮次与话题，决定下一个发言者是谁。

### 8.3.3 与 ReAct 的对比

| 维度 | ReAct | AutoGen |
|------|-------|---------|
| 控制粒度 | 单 Agent 的内部循环 | 多 Agent 的外部编排 |
| 推理载体 | 自然语言 Thought | 自然语言对话 + 结构化消息 |
| 记忆位置 | 线性上下文 | 可持久化的对话历史 + 外部记忆 |
| 任务分解 | 隐式，由单 Agent 在 Thought 中完成 | 显式，由不同 Agent 分治 |
| 终止条件 | Action = Finish | 对话图到达终止节点 |
| 适用场景 | 单步/多步工具调用，问答 | 软件开发、数据分析、多角色协作 |

**关键澄清**：AutoGen 并未取代 ReAct，而是把 ReAct 的循环内嵌到每个 Agent 实例中 [已证明]。例如，在 AutoGen 的编码 Agent 中，当接收到"写一个排序函数"的任务时，该 Agent 内部可以运行一个 ReAct 式的子循环：Thought（"我需要先写单元测试"）→ Action（调用代码生成器）→ Observation（代码生成结果）→ Thought（"看起来正确，但缺少边界处理"）→ Action（修改代码）。AutoGen 管理的是 Agent 之间的宏观控制流，ReAct 管理的是 Agent 内部的微观推理流。

### 8.3.4 对话路由与群聊机制

AutoGen 的群聊管理器使用一个基于规则的或 LLM 驱动的路由函数。规则路由可以是轮询（round-robin）或基于关键词的触发（如包含"代码"则路由到编码 Agent）。LLM 驱动路由则把整个对话历史输入到一个"路由模型"，由其生成下一个发言者的名字。

#### 工程权衡：路由开销

这带来了一个微妙的工程权衡：路由模型本身消耗 token，如果群聊成员众多，路由开销可能占到总成本的 15%–20% [合理推断]。一个具体的数量级估计：假设群聊有 4 个 Agent，每轮对话历史 2k token，路由模型生成 10 token 来选择发言者，则 10 轮群聊的路由成本为 10 × 2k × 4（因为路由模型通常需要读取所有候选 Agent 的系统提示）≈ 80k token，约等于 4 次 GPT-4 调用的成本。

因此，在生产环境中，规则路由与 LLM 路由的混合策略往往更经济：先用关键词规则做粗筛，仅在歧义情况下启用 LLM 路由。

#### 人机交接（Human-in-the-loop）

AutoGen 允许在对话图的任意节点插入人工审批：当 Agent 生成的代码涉及删除文件或发起网络请求时，系统暂停并等待人类确认。这种"软约束"比硬编码权限列表更灵活，但也引入了延迟与交互设计的复杂性。

在实践中，人机交接的触发条件需要精心设计：过于敏感会导致用户体验断裂（用户每 30 秒就要点一次"确认"），过于宽松则失去安全意义。一个有效的折衷是"白名单 + 敏感操作触发"：常见文件读写自动通过，但 `rm -rf`、`DROP TABLE`、`转账` 等操作强制暂停。

**认知偏差警示**：人机交接的设计容易陷入**安全剧场**（security theater）——表面上设置了审批环节，但用户因为频繁被打断而养成"无脑点确认"的习惯，审批形同虚设。真正的安全需要**最小权限原则**（Agent 默认无权限，只授予完成任务所需的最小权限）而非"默认全权限 + 事后审批" [合理推断]。

---

## 8.4 Plan-and-Solve vs ReAct：何时需要显式规划

### 8.4.1 ReAct 的逐步贪心本质

#### 直觉解释
想象你要从北京开车到上海。

- **ReAct 策略**：每个路口都根据当前路况决定下一步。如果前方堵车，就绕路。这很灵活，但如果你在第一个路口就走错了方向，后续的"灵活绕路"只是在错误方向上不断调整——离目标越来越远。
- **Plan-and-Solve 策略**：出发前先在地图上规划好完整路线（北京→济南→徐州→上海），然后按路线行驶。如果路上遇到堵车，可以在局部绕路，但大方向不变。

ReAct 的每一步决策都基于当前上下文生成"下一步"的最佳行动。这相当于在搜索树上做深度优先的贪婪下降。对于"需要 3 步以内、每步结果可预期"的任务（如查询天气后计算穿衣指数），贪心策略足够高效。但对于"需要 10 步以上、早期错误会级联放大"的任务（如构建一个完整的数据分析管道），贪心策略的成本极高。

### 8.4.2 Plan-and-Solve 的架构

Plan-and-Solve 路线主张在调用任何工具之前，先让 LLM 生成一个完整的计划树（plan tree），然后按顺序或并行执行。Beyond ReAct [Wei et al., 2026] 提出了一个以 Planner 为中心的框架：Planner LLM 接收任务描述，输出一个 DAG（有向无环图），其中节点是工具调用，边是数据依赖。执行器（Executor）根据拓扑排序调度节点，可以并行调用无依赖的工具。

例如，任务"比较 A 公司与 B 公司的财务数据并生成图表"可以分解为：

- 节点 1：检索 A 公司财报
- 节点 2：检索 B 公司财报
- 节点 3（依赖 1, 2）：对比分析
- 节点 4（依赖 3）：生成图表

节点 1 与 2 可以并行，节点 3 必须等待 1 与 2 完成。这种显式规划把 ReAct 的"逐步探索"转化为"先规划后执行"，显著降低了长程任务的总延迟与 token 消耗。

### 8.4.3 何时用 ReAct，何时用 Plan-and-Solve

工程上的决策规则可以归纳为 [合理推断]：

- **任务步数 < 5，且每步结果高度不确定**（如开放式问答、交互式网页浏览）：选 ReAct。其交互性允许模型根据中间结果动态调整方向，避免了"计划赶不上变化"的僵化。
- **任务步数 > 10，且子任务间依赖关系明确**（如批量数据处理、ETL 管道、多步骤科学实验）：选 Plan-and-Solve。显式 DAG 允许并行执行、错误隔离与重试机制。
- **中间地带**：可以采用混合策略，如 ReAct 的顶层用 Plan-and-Solve 生成里程碑，每个里程碑内部用 ReAct 逐步探索。LLM Compiler [Kim et al., 2023] 正是这种混合思想的代表：它先让模型生成一个"并行函数调用计划"，然后编译为 DAG 执行，既保留了 LLM 的推理能力，又避免了逐轮调用的串行瓶颈。

#### 反事实推演：如果 Plan-and-Solve 的计划总是完美的

假设2025年出现了一种 Planner LLM，它生成的计划在 95% 的情况下是完美的（无遗漏、无错误依赖）：

- **短期（1-2年）**：ReAct 在长程任务上会被大幅压缩——只需要在计划执行出错时才启用 ReAct 式的局部调整。
- **长期（5-10年）**：但"完美计划"的假设在开放世界中不成立。网络搜索结果可能为空、API 可能超时、数据格式可能不符预期。Plan-and-Solve 需要一个"异常处理层"来应对这些情况——而这个异常处理层本质上就是一个 ReAct 循环。**这意味着：Plan-and-Solve 不是 ReAct 的替代品，而是 ReAct 的"前置优化"——先规划大方向，再在局部用 ReAct 应对不确定性** [合理推断]。

#### 认知偏差警示
Plan-and-Solve 的推崇者可能犯了**完美计划幻觉**——认为复杂任务可以在执行前被完全规划。但现实世界的任务是**部分可观察**和**非确定性**的：你无法在执行前预知搜索结果是否相关、API 是否可用、数据是否完整。把 Plan-and-Solve 视为"比 ReAct 更高级"的方案，是一种对不确定性的低估 [合理推断]。

---

## 8.5 Agent 架构的组件分解：感知、记忆、推理、行动、评估

尽管 ReAct 与 AutoGen 在具体实现上差异显著，现代 LLM Agent 的架构可以统一分解为五个组件。理解每个组件的设计空间，有助于在工程中根据需求进行裁剪与组合。

### 8.5.1 感知（Perception）

感知层负责将环境输入转换为 LLM 可处理的文本。在 ReAct 中，感知是工具返回的 Observation 字符串。在 AutoGen 中，感知还包括其他 Agent 发送的消息。对于多模态 Agent（如第11章将介绍的视觉-语言 Agent），感知层还需处理图像编码、音频转录等。

关键设计决策是：**原始感知是否经过预处理？** 例如，网页浏览工具返回的 HTML 通常包含大量脚本与样式噪声，直接输入 LLM 会浪费上下文；常见的做法是先用 Readability 或文本提取器做清洗，再作为 Observation。

**工程经济学权衡**：预处理增加了计算开销和延迟，但节省了上下文 token。在 GPT-4o 的定价下，1k token 的成本约 $0.005，而一次 HTML 清洗的计算成本约 $0.0001。如果清洗能节省 5k token，则投入产出比是 25:1 [合理推断]。

### 8.5.2 记忆（Memory）

#### 直觉解释
ReAct 的"记忆"像是一个人的短期记忆——你正在进行的对话内容都在脑子里，但容量有限。AutoGen 的显式记忆像是一个人的笔记本——对话内容可以记下来，需要时翻阅。向量记忆（如 FAISS、Chroma）则像是一个人的图书馆——你可以按主题检索相关内容。

#### 技术细节
ReAct 的"记忆"是隐式的：全部历史都在上下文窗口中。AutoGen 则支持显式记忆——对话历史可以持久化到数据库，并在新会话中检索。更进一步，向量记忆允许 Agent 存储和检索非结构化文档。BOLAA [Liu et al., 2023] 的基准测试表明，引入外部向量记忆后，长程多跳问答的准确率提升 12%–18% [强实验证据]，但延迟增加约 200 ms（取决于检索 top-k）。

记忆的另一个维度是**工作记忆 vs. 长期记忆**。工作记忆对应上下文窗口中的近期内容，长期记忆对应外部存储。MemGPT 提出的分层记忆机制（类似操作系统分页）把旧对话摘要存入长期记忆，在需要时通过 LLM 生成的搜索查询进行检索。

工程上，记忆层的设计需回答三个问题：

1. **写入时机**：是每步都写入，还是仅在完成子任务后写入？频繁写入保证信息完整性，但增加存储与索引开销；稀疏写入减少噪音，但可能遗漏关键中间结论。
2. **读取策略**：是简单的向量相似度检索，还是由 LLM 动态生成检索查询？后者更灵活，但每轮增加一次 LLM 调用。
3. **遗忘机制**：长期记忆是否有过期策略？在持续运行的 Agent 系统中，无限制增长的记忆库会导致检索噪音上升。常见做法是按时间衰减或按重要性打分（如 LLM 判断该信息是否对未来任务有用）。

**认知偏差警示**：记忆系统的设计容易陷入**全保留幻觉**——认为"记住所有信息"一定比"遗忘部分信息"更好。但人类的遗忘机制不是缺陷，而是功能——它过滤噪音，让重要信息凸显。Agent 的记忆系统也需要"智能遗忘"，而非"全量保留" [合理推断]。

### 8.5.3 推理（Reasoning）

推理是 Agent 的"认知核"。ReAct 使用单模型自回归推理，AutoGen 允许多模型协作推理（如一个 Agent 负责逻辑分析，另一个负责创意生成）。当前推理层的设计空间包括：

- **提示策略**：CoT、ReAct、Tree-of-Thoughts [Yao et al., 2023]（允许并行探索多条推理路径并投票）。
- **推理深度**：单步推理（直接生成 Action）vs. 多步推理（先生成 Thought 再 Action）。
- **工具增强**：是否允许模型在推理过程中调用外部符号求解器（如 Wolfram Alpha、Python 解释器）来验证数学推导。

### 8.5.4 行动（Action）

行动层是 Agent 对外部世界施加影响的接口。行动可以是工具调用（API、数据库、代码执行）、环境控制（网页点击、文件操作）或通信（向其他 Agent 发送消息）。

关键设计是**行动空间的抽象级别**：低级别行动（如原始 HTTP 请求）灵活但易错，高级别行动（如 `send_email(to, subject, body)`）安全但受限。AutoGen 的函数签名机制要求每个工具提供 JSON Schema 描述，LLM 生成符合 Schema 的调用参数，这类似于 ReAct 的 `Action[params]` 格式，但结构化程度更高。

行动层还需要考虑**副作用与幂等性**。ReAct 的 Action 序列可能因解析错误或 LLM 幻觉而重复执行同一操作（如重复扣款、重复写入）。工程上，工具设计应遵循幂等原则：同一 Action 执行多次与执行一次结果相同。对于天然非幂等的操作（如支付、文件删除），应在系统层引入去重键（idempotency key）或事务回滚机制。

此外，行动层需要**超时与重试策略**：工具调用可能因网络抖动失败，Agent 不应直接崩溃，而应根据错误类型（可重试 vs. 不可重试）决定下一步。

### 8.5.5 评估（Evaluation）

评估组件决定 Agent 何时停止、何时重试、何时请求人工帮助。ReAct 的评估是硬编码的：当 Action 为 `Finish` 时终止。AutoGen 的评估可以是用户定义的终止函数（如"当代码通过所有单元测试时终止"）。更复杂的评估引入 LLM-as-a-Judge：用另一个 LLM 评估当前 Agent 的输出质量，并给出改进建议。这在 Self-Refine [Madaan et al., 2023] 与 Reflexion [Shinn et al., 2023] 中得到了深入探讨（参见第10章）。

---

## 8.6 代码：完整 ReAct 循环实现

### 8.6.1 常见错误实现：先看怎么写错

在给出正确实现之前，先看两个常见的错误实现，帮助读者建立"免疫记忆"。

#### 错误实现一：解析器过于宽松

```python
# ❌ 错误：解析器接受任何包含 "Search" 的行作为 Action
def parse_action_bad(text: str) -> Tuple[str, str]:
    for line in text.splitlines():
        if "Search" in line:  # [注意] Thought 中提到 "Search" 也会被误解析
            return "Search", line.split("Search")[-1].strip()
    return "", ""
```

**问题分析**：当 LLM 的 Thought 是"我应该 Search for the answer"时，解析器会错误地把这句话当成 Action。这种错误在低 temperature 下发生率约 5%，但在高 temperature 下可达 20% [合理推断]。

**认知偏差根源**：这是**可得性偏差**的一种体现——开发者只测试了"正常"情况（Thought 不包含工具名），没有考虑边界情况。

#### 错误实现二：没有错误恢复机制

```python
# ❌ 错误：解析失败直接抛异常，Agent 崩溃
def run_bad(self, task: str) -> str:
    for step_i in range(self.max_steps):
        prompt = self._build_prompt(task)
        raw_output = self.llm(prompt)
        action_name, action_arg = self._parse_action(raw_output)
        if not action_name:
            raise ValueError(f"Failed to parse action at step {step_i}")  # [关键局限] 无恢复机制
        # ... 执行 Action
```

**问题分析**：LLM 的输出是不可控的。在一次解析失败时就崩溃，意味着 Agent 的鲁棒性完全依赖于 LLM 的格式遵守能力。在生产环境中，这会导致频繁的系统中断。

**正确做法**：当解析失败时，应该把错误信息作为 Observation 注入上下文，让 LLM 在下一轮自我修正。

### 8.6.2 正确实现

下面的代码实现了一个自包含的 ReAct Agent，包含 Thought → Action → Observation 状态机。它使用本地函数模拟工具调用，不依赖任何外部 LLM API（读者可替换 `call_llm` 函数为 OpenAI 或本地 vLLM 调用）。

```python
"""
ReAct Loop Implementation
一个完整的 Thought-Action-Observation 状态机，兼容任意自回归 LLM。
"""
import re
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"

@dataclass
class Step:
    step_type: StepType
    content: str

class ReActAgent:
    def __init__(
        self,
        llm_caller: Callable[[str], str],
        tools: Dict[str, Callable[[str], str]],
        max_steps: int = 10,
        action_pattern: str = r"Action:\s*(\w+)\[(.*?)\]",
        finish_action: str = "Finish",
    ):
        self.llm = llm_caller          # 任意 LLM 调用函数 signature: str -> str
        self.tools = tools             # 工具名 -> 可调用函数
        self.max_steps = max_steps
        self.action_pattern = re.compile(action_pattern, re.DOTALL)
        self.finish_action = finish_action
        self.history: List[Step] = []  # 维护完整认知状态

    def _build_prompt(self, task: str) -> str:
        """将历史轨迹线性化为 prompt。"""
        lines = [
            "Solve the following task by alternating between Thought and Action. "
            "When you have the final answer, use Action: Finish[answer].",
            "Available tools: " + ", ".join(self.tools.keys()),
            "",
            f"Task: {task}",
            "",
        ]
        for step in self.history:
            if step.step_type == StepType.THOUGHT:
                lines.append(f"Thought: {step.content}")
            elif step.step_type == StepType.ACTION:
                lines.append(f"Action: {step.content}")
            elif step.step_type == StepType.OBSERVATION:
                lines.append(f"Observation: {step.content}")
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
        lines.append("Thought:")
        return "\n".join(lines)

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """从 LLM 输出中提取 Action[name[arg]]。"""
        # [注意] 兼容 Thought 与 Action 混排：取最后一个 Action 行
        # [关键局限] 如果 Thought 中包含 "Action:" 关键字，仍可能误解析
        # 生产环境建议使用 JSON mode 或 constrained decoding
        for line in reversed(text.strip().splitlines()):
            match = self.action_pattern.search(line)
            if match:
                return match.group(1), match.group(2)
        return "", ""

    def _extract_thought(self, text: str) -> str:
        """提取 Thought 部分（Action 之前的文本）。"""
        # 简单策略：取最后一个 Action: 之前的全部文本
        idx = text.rfind("Action:")
        if idx != -1:
            return text[:idx].strip()
        return text.strip()

    def run(self, task: str) -> str:
        """
        主循环：交替生成 Thought/Action，执行工具，更新历史。
        返回最终答案或达到 max_steps 后的最佳答案。
        """
        for step_i in range(self.max_steps):
            prompt = self._build_prompt(task)
            raw_output = self.llm(prompt)  # 自回归生成

            thought = self._extract_thought(raw_output)
            self.history.append(Step(StepType.THOUGHT, thought))

            action_name, action_arg = self._parse_action(raw_output)
            if not action_name:
                # [注意] 解析失败时不崩溃，而是通过 Observation 提示模型修正格式
                # 这是 ReAct 鲁棒性的关键设计：错误是 Observation 的一种，模型可以据此调整
                self.history.append(
                    Step(StepType.OBSERVATION, "Error: no parsable Action found. "
                         "Please use the format 'Action: ToolName[arg]'.")
                )
                continue

            self.history.append(Step(StepType.ACTION, f"{action_name}[{action_arg}]"))

            if action_name == self.finish_action:
                return action_arg

            if action_name not in self.tools:
                obs = f"Error: tool '{action_name}' is not available. " \
                      f"Available tools: {list(self.tools.keys())}"
            else:
                try:
                    obs = self.tools[action_name](action_arg)
                except Exception as e:
                    obs = f"Error executing {action_name}: {e}"
            self.history.append(Step(StepType.OBSERVATION, obs))

        # 达到 max_steps 仍未终止，返回最后一次 Thought 作为 fallback
        return f"[Max steps reached] Last thought: {self.history[-1].content}"


# ---------------------------------------------------------------
# 模拟 LLM：在真实场景中替换为 OpenAI API / vLLM / 本地模型
# ---------------------------------------------------------------
class MockLLM:
    """一个基于规则的模拟 LLM，仅用于演示 ReAct 状态机。"""
    def __init__(self, scenario: str = "qa"):
        self.scenario = scenario
        self.step_count = 0

    def __call__(self, prompt: str) -> str:
        self.step_count += 1
        # 根据 prompt 中的历史决定下一步（极其简化的规则）
        if "multiply" in prompt.lower() or "120 * 456" in prompt:
            if "Observation" not in prompt:
                return "Thought: I need to calculate 120 * 456.\nAction: Calculate[120 * 456]"
            else:
                return "Thought: The calculation gives 54720, which is the final answer.\nAction: Finish[54720]"
        elif "search" in prompt.lower() or "capital of France" in prompt.lower():
            if "Observation" not in prompt:
                return "Thought: I should search for the capital of France.\nAction: Search[capital of France]"
            else:
                return "Thought: The search result confirms Paris is the capital.\nAction: Finish[Paris]"
        # 默认 fallback
        return "Thought: I will finish now.\nAction: Finish[unknown]"


# ---------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------
def calculate(expr: str) -> str:
    """安全的数值表达式求值（仅允许算术运算符）。"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expr):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expr))  # [注意] 生产环境应使用 asteval 或 sympy
    except Exception as e:
        return f"Error: {e}"

def search(term: str) -> str:
    """模拟搜索引擎。"""
    database = {
        "capital of France": "Paris is the capital and most populous city of France.",
        "react paper": "ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022).",
    }
    return database.get(term.lower(), f"No results found for '{term}'.")


# ---------------------------------------------------------------
# 运行示例
# ---------------------------------------------------------------
if __name__ == "__main__":
    tools = {
        "Calculate": calculate,
        "Search": search,
    }
    llm = MockLLM(scenario="qa")
    agent = ReActAgent(llm_caller=llm, tools=tools, max_steps=6)

    task1 = "What is 120 * 456?"
    print(f"Task: {task1}")
    answer1 = agent.run(task1)
    print(f"Answer: {answer1}\n")

    # 查看完整认知轨迹
    print("--- Trajectory ---")
    for step in agent.history:
        print(f"[{step.step_type.value.upper():12}] {step.content}")

    # 第二个任务
    agent2 = ReActAgent(llm_caller=MockLLM(), tools=tools, max_steps=6)
    task2 = "What is the capital of France?"
    print(f"\nTask: {task2}")
    answer2 = agent2.run(task2)
    print(f"Answer: {answer2}")
```

### 代码设计要点

1. **状态机显式化**：`Step` 数据类与 `StepType` 枚举让 Thought/Action/Observation 不再是 prompt 中的文本，而是程序中的第一类对象。这为后续的可视化、调试、持久化提供了基础。
2. **解析鲁棒性**：`_parse_action` 使用正则从最后一行提取，避免模型在 Thought 中无意提及 Action 关键字导致的误解析。`_extract_thought` 则把 Action 之前的文本全部视为 Thought，兼容模型在中间插入换行的情况。
3. **错误注入**：当模型未生成合法 Action 时，Observation 不是静默失败，而是显式返回错误提示，让 LLM 在下一轮自我修正。这是 ReAct 鲁棒性的关键：Observation 可以是错误信息，模型可以据此调整策略。
4. **LLM 解耦**：`llm_caller` 是任意 `str -> str` 函数，读者可以无缝替换为 OpenAI 的 `chat.completions.create` 或本地 vLLM 服务。这种解耦是生产代码的必备要求。

---

## 8.7 工程实践框

> **训练与部署 ReAct/AutoGen 风格 Agent 的常见工程陷阱**
>
> **1. Prompt 格式的一致性比内容更重要**
> 在复现 ReAct 时，研究者常犯的错误是在 few-shot 示例中混用不同的 Action 格式（如 `Search[term]`、`Search: term`、`{ "tool": "Search", "arg": "term" }`）。LLM 对格式极其敏感，即使语义等价，格式不一致也会导致生成解析失败率上升 5–10 倍 [强实验证据]。建议：在系统提示中给出严格的 BNF 语法，并用 `pydantic` 或 JSON Schema 在后处理阶段做结构化校验。
> **认知偏差根源**：这是**可得性偏差**——开发者只测试了自己熟悉的格式，没有考虑 LLM 对格式敏感这一非直觉特性。
>
> **2. 上下文窗口的"隐性截断"**
> 当 ReAct 轨迹超过 8k token（对于 GPT-3.5 的 16k 窗口）时，模型会忽略早期 Thought，导致任务遗忘。实践中，如果预计任务超过 10 步，应在每 5 步后插入一个"摘要 Thought"：让 LLM 用 100 字总结当前进展与剩余计划，然后丢弃中间细节。这会将有效步数扩展 2–3 倍，代价是每步增加一次 LLM 调用（用于摘要）。
> **原理说明**：Transformer 的 softmax 注意力在长上下文下会产生"注意力稀释"——早期 token 的注意力权重被后续 token 稀释。这不是模型实现的问题，而是 softmax 归一化的数学性质决定的 [已证明]。
>
> **3. 工具延迟与超时设计**
> 在 AutoGen 多 Agent 系统中，工具调用（如代码执行、网络请求）的延迟差异巨大。如果一个 Agent 调用了耗时 30 秒的数据库查询，整个对话图会阻塞。工程上应采用异步消息队列：每个 Agent 的 Action 被提交到队列，执行完成后通过回调触发下一轮 LLM 调用。Python 的 `asyncio` 或 `Celery` 是常见的选择。
>
> **4. 超参选择：temperature 与 max_tokens**
> - ReAct 的 Thought 生成：temperature=0.1–0.3，max_tokens=256–512。低温保证推理路径的确定性，避免模型在 Thought 中发散到无关话题。**原理**：temperature 控制softmax输出的熵，低温使分布更尖锐，减少随机性。
> - Action 生成：temperature=0。Action 必须可解析，任何随机性都会增加格式错误率。**原理**：temperature=0 等价于贪心解码，完全消除采样随机性。
> - 对于需要创意的 AutoGen 角色（如头脑风暴 Agent），temperature 可提高到 0.7–1.0，但 Action 解析仍需约束输出格式（如通过 JSON mode 或 constrained decoding）。
>
> **5. 硬件需求估算**（时间窗口：2025年中参考价格）
> - **单 ReAct Agent**：如果使用 GPT-4o API，瓶颈不在本地 GPU，而在网络延迟与 token 成本。每步约消耗 1k–2k prompt token + 200–500 completion token，按 2025年中 GPT-4o 定价（$5/1M input, $15/1M output），10 步任务成本约 $0.045–$0.225。
> - **本地 LLM 后端**：若用 Llama-3-70B 或 Qwen-72B 作为本地 Agent 后端，推理需 2×40GB A100（FP16）或 1×48GB A6000（AWQ 4-bit 量化）。量化后的 4-bit 模型在 ReAct 场景下几乎不损失准确率 [强实验证据]，但 latency 增加约 30%。2025年中二手 A6000 价格约 $4,000-$8,000。
> - **AutoGen 多 Agent 并发**：当 4 个 Agent 同时活跃时，如果每个都挂载 70B 模型，需要至少 4×2 A100 = 8 张 A100。更经济的方案是"小模型分角色"：Planner 用 70B，Coder 用 13B–34B，Reviewer 用 7B，通过 vLLM 的 PagedAttention 共享 KV Cache 来降低显存碎片。2025年中 8×A100 80GB 服务器租赁价格约 $12-$20/小时。
>
> **6. 评估指标的陷阱**
> 不要只用"最终答案正确率"衡量 Agent。ReAct 论文报告了 **success rate**（是否到达正确终止状态）、**trajectory length**（步数，反映效率）与 **hallucination rate**（Thought 中是否包含虚假事实）。在 AutoGen 中，还需增加 **conversation turns**（轮数，反映协作效率）与 **human intervention count**（需要人工救场的次数）。只优化成功率会导致 Agent 生成冗长的保守轨迹，牺牲用户体验。
> **认知偏差根源**：只看成功率的评估方式反映了**结果偏差**（outcome bias）——只关注最终结果，忽视了过程的效率和质量。
>
> **7. 安全边界**
> ReAct 的 Action 空间如果包含文件系统或网络操作，必须 sandbox。推荐方案：代码执行使用 `gVisor` 或 `Firecracker` 微虚拟机；网络请求通过代理白名单过滤；敏感操作（如删除、转账）强制触发 human-in-the-loop。AutoGen 的 `human_input_mode` 配置项应在生产环境中默认开启 `ALWAYS`，仅在内部测试时关闭。

---

## 8.8 批判性分析

### 8.8.1 内在局限性：ReAct 在原理上无法解决什么问题？

ReAct 的根本局限在于**文本表征的维度瓶颈**。LLM 的推理基于自然语言，而自然语言是对世界的一种**有损压缩**——它丢弃了物理世界的连续性、精确数值和因果机制 [已证明]。

具体而言：
1. **物理直觉缺失**：ReAct 无法预测"把杯子放在桌子边缘"会导致杯子掉落。LLM 从文本中学到的"边缘"是一个模糊概念，而非精确的几何关系。
2. **因果推理薄弱**：ReAct 的 Thought 是基于相关性而非因果性的。模型知道"下雨和带伞相关"，但不知道"下雨导致带伞"还是"带伞导致下雨"。这在需要干预决策的任务（如医疗诊断）中是致命缺陷 [强实验证据]。
3. **数值精度不足**：当任务需要精确计算（如"给予0.15mg/kg的药物"）时，LLM 的文本生成可能产生数值错误。Thought 中的"大约0.15"可能被生成"0.15"或"0.015"——在医疗场景中这是致命的。

### 8.8.2 实证盲区：关键实验遗漏了什么？

ReAct 原始论文的实验存在几个值得审视的盲区：

1. **任务选择偏差**：HotpotQA、Fever、WebShop 都是**信息检索型任务**——Observation 是搜索引擎返回的文本。但 Agent 任务不只有信息检索：物理操作、社交交互、创意生成等场景在论文中未被测试。将 ReAct 在检索任务上的成功推广到所有 Agent 任务，是一个**领域泛化偏差** [合理推断]。
2. **模型规模效应未充分探索**：原始论文主要使用 GPT-3 (175B) 和 GPT-3.5。小模型（7B-13B）是否能有效执行 ReAct？后续研究表明，7B 模型在 ReAct 格式遵守上的失败率显著高于 70B 模型 [强实验证据]，但原始论文未讨论这一点。
3. **长程任务的失败模式**：论文中的任务大多在 5 步以内完成。当任务超过 10 步时，ReAct 的成功率如何变化？原始论文未提供这一数据。后续工作（如 Reflexion）的实验表明，超过 10 步的任务成功率可能下降 30%-50% [强实验证据]。

### 8.8.3 替代路径的现况：今天是否有更好的方案？

| 方案 | 适用场景 | 相对于 ReAct 的优势 | 为什么未被广泛采用？ |
|------|---------|-------------------|-------------------|
| 程序合成（Code Interpreter） | 数据分析、数值计算 | 代码可精确执行，无解析歧义 | 无法处理需要多轮交互的开放任务 |
| Plan-and-Solve (DAG) | 长程结构化任务 | 并行执行，降低延迟 | 需要预先知道任务结构，对不确定性敏感 |
| Tree-of-Thoughts | 需要探索多种方案的任务 | 并行探索多条路径，投票选最优 | 计算成本是 ReAct 的 5-10 倍 |
| 纯 RL Agent | 游戏环境、物理模拟 | 可直接优化物理性能指标 | 样本效率低，无法处理开放文本 |

**关键洞察**：这些方案未被广泛采用，不一定是"技术劣势"，而可能是**生态锁定**（ecosystem lock-in）——ReAct 的简单性（只需改 prompt）使其成为社区默认选择，而替代方案需要额外的工程投入 [合理推断]。

### 8.8.4 认知偏差警示

ReAct 研究者（以及使用 ReAct 的工程师）可能存在以下认知偏差：

1. **确认偏差**：一旦 ReAct 成为主流范式，研究者倾向于在 ReAct 框架内解决所有问题，即使某些问题可能更适合其他方案。例如，需要精确数值计算的任务本应使用程序合成，但研究者可能在 ReAct 中反复调试 "Calculate" 工具。
2. **幸存者偏差**：发表的论文大多报告 ReAct 的成功案例，失败的案例（尤其是在物理操作任务上的失败）很少被报告。这可能给社区造成"ReAct 适用于所有任务"的错觉。
3. **后见之明偏差**：在 ReAct 发表之后，"推理和行动应该交错"看起来"理所当然"。但在2022年之前，主流做法是 CoT（纯推理）或 Toolformer（单次工具调用）。ReAct 的"交替循环"在当时的约束下并不是"显然的"选择。

---

## 8.9 与其他章节概念的连接

### 前置知识
- **第7章（工具增强 LLM）**：Toolformer 是 ReAct 的直接前驱。第7章讨论了"如何让 LLM 学会调用工具"，本章讨论了"如何在推理过程中动态编排工具调用"。两者的关系是：Toolformer 在**预训练阶段**嵌入工具调用能力，ReAct 在**推理阶段**编排工具调用策略。
- **第2章（Transformer 架构）**：ReAct 的有效性依赖于 Transformer 的自注意力机制——上下文中的早期 Thought 可以通过注意力直接影响后续 Action 的生成。如果 Transformer 的上下文窗口过小（如早期的 GPT-3 只有 2k token），ReAct 的步数会受到硬约束。

### 后续展开
- **第10章（自我反思与改进）**：ReAct 的"贪心局部最优"问题催生了 Reflexion 和 Self-Refine。这些方法在 ReAct 的循环之外增加了一个"反思层"，让 Agent 在失败后总结经验并重试。
- **第12章（世界模型与规划）**：ReAct 缺乏物理因果预测能力，这正是 World Model 要补充的。第12章将讨论如何将 World Model 的预测能力嵌入 Agent 的决策循环。
- **第14章（多 Agent 系统安全）**：AutoGen 的多 Agent 协作引入了新的安全挑战——如何确保 Agent 之间的对话不被恶意利用？这将在第14章深入讨论。

---

## 8.10 小结

本章从 ReAct 的单 Agent 推理—行动循环出发，延伸到 AutoGen 的多 Agent 对话编排，并对比了 ReAct 与 Plan-and-Solve 两种策略的适用边界。核心结论可概括为三点：

1. **ReAct 的价值在于把 LLM 的常识推理外化为可追踪、可干预的符号轨迹** [强实验证据]。它不是一个新算法，而是一种"把 LLM 当作认知核"的系统工程范式。其设计精髓是 Thought 的显式化：让模型在生成 Action 之前先"说出来"它想做什么，这既提高了可解释性，又为错误诊断提供了抓手。但 ReAct 的局限是原理性的——文本表征无法替代物理因果推理 [已证明]。

2. **AutoGen 的价值在于把 ReAct 的循环从单进程扩展到分布式进程** [合理推断]。当任务复杂度超过单 Agent 的上下文承载能力时，角色分解与对话路由是更 scalable 的架构。但多 Agent 不是免费的：路由开销、状态同步、消息序列化都会增加系统复杂度与延迟。"更多 Agent ≠ 更好的结果" [强实验证据]。

3. **没有 universally optimal 的 Agent 架构** [开放问题]。ReAct 适合短程、动态、不确定的任务；Plan-and-Solve 适合长程、结构化、依赖明确的任务。生产系统往往采用混合架构：Planner 生成 DAG 里程碑，每个节点内部用 ReAct 探索，最终通过 Evaluator 组件验证结果。这种"分层控制"的思想将在第12章（世界模型与规划）与第14章（多 Agent 系统安全）中进一步深化。

---

## 本章练习题

### 概念题
> **Q1**：ReAct 的上下文 $c_t$（Thought-Action-Observation 历史）和 POMDP 的信念状态 $b_t(s)$（状态概率分布）都是对不确定世界的"表示"。写出两者的本质区别：信念状态是什么数学对象？上下文如何近似它？近似的代价是什么？

> **Q2**：第8章列举了 ReAct 的四个失败模式。选择你认为在养老机器人场景中最危险的一个，解释原因并提出一种工程缓解方案。

### 推导题
> **Q3**：设 ReAct 每步产生：Thought（100 tokens）+ Action（50 tokens）+ Observation（150 tokens）= 300 tokens/步，系统提示占 2000 tokens。
> (a) 第几步之后总上下文超过 4096 tokens（GPT-3.5 限制）？
> (b) 使用 GPT-4（32K 上下文）时，最多可执行多少步？
> (c) 实际比理论值更保守的原因？（提示：注意力稀释与费用控制）
> (d) **边界条件分析**：当 Observation 的长度不是固定 150 tokens，而是随步数线性增长（因为搜索结果越来越长）时，(a) 和 (b) 的结论如何变化？写出 Observation 长度函数 $L_{obs}(t) = 150 + 50t$ 时的总上下文长度公式。

### 批判性思考题
> **Q4**：用证据强度框架评估以下论断："ReAct 是当前最优的 Agent 架构，因为它在 HotpotQA 和 WebShop 上都取得了最佳性能。" 具体回答：(a) 这个论断的证据强度是什么等级？(b) 它遗漏了哪些边界条件和反例？(c) 如果你要设计一个实验来证伪这个论断，你会选择什么任务和什么对比方案？

### 反事实分析题
> **Q5**：如果2022年 Toolformer 先于 ReAct 发布，并且 Toolformer 已经支持了"根据返回结果再推理"的循环（即 Toolformer 已经具备了 ReAct 的核心功能），ReAct 还会被提出吗？请分析：(a) ReAct 相对于这种"增强版 Toolformer"的独特价值是什么？(b) 如果 ReAct 不会被提出，这对 Agent 研究的发展路径会有什么影响？

### 编程题
> **Q6**：用本章的 ReAct 框架代码，接入维基百科搜索 API（无需密钥），实现以下任务：
> "查询：中国现任国家主席是谁，他什么时候开始担任这一职位？"
> 要求：最多 3 步完成，每步打印 Thought / Action / Observation。
> **进阶要求**：在代码中增加"推理-行动一致性检查"——当 Thought 中提到的工具参数与 Action 中的参数不一致时，输出警告并让模型重新生成。

---

## 核心概念速查

| 术语 | 定义（≤25字） |
|------|-------------|
| ReAct | 交替生成思考和行动的 LLM Agent 单步推理框架 |
| POMDP | 部分可观测马尔可夫决策过程，Agent 的理论基础 |
| 上下文状态 | 用对话历史近似信念状态的 ReAct 信息表示 |
| AutoGen | 微软的多 Agent 对话框架，支持角色分工和群聊 |
| GroupChat | AutoGen 中管理多 Agent 动态发言的中心机制 |
| 角色漂移 | 多轮对话后 Agent 偏离初始系统提示的现象 |
| 注意力稀释 | 上下文过长时模型对早期信息关注度下降 |
| Plan-and-Solve | 先生成全局计划 DAG 再执行的 Agent 策略 |
| LLM Compiler | 将串行 ReAct 轨迹编译为并行 DAG 的方法 |
| 推理-行动错位 | Thought 中声称做A但Action生成B的不一致现象 |


---

## 延伸阅读

1. **[Yao et al., 2022]** ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. 本章的基石论文，首次系统展示了 Thought 与 Action 的交错生成对决策任务的增益，并提供了 HotpotQA、Fever、WebShop 等基准的完整实验对比。

2. **[Wu et al., 2023]** AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155. AutoGen 框架的原始论文，详细定义了 Conversable Agent、Conversation Programming、Group Chat Manager 等核心抽象，并提供了代码库与多个应用案例（数学、编程、网页浏览）。

3. **[Kim et al., 2023]** An LLM Compiler for Parallel Function Calling. arXiv:2312.04511. 提出将 LLM 生成的串行工具调用编译为并行 DAG 执行，是 ReAct 与 Plan-and-Solve 混合路线的重要实践。论文中的"LLM Planner + Parallel Executor"架构对降低多步任务的端到端延迟具有直接工程价值。

4. **[Wei et al., 2026]** Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning. Proceedings of AAAI. 系统论证了纯 ReAct 在长程任务中的局部最优陷阱，提出以显式 Planner 生成 DAG 的替代方案，并给出了与 ReAct 在科学计算、数据分析任务上的对比实验。

5. **[Liu et al., 2023]** BOLAA: Benchmarking and Orchestrating LLM-Augmented Autonomous Agents. arXiv:2308.05960. 对 LLM Agent 的架构维度（单 Agent vs. 多 Agent、不同记忆机制、不同工具集）进行了系统基准测试，为 Agent 设计提供了量化参考。

6. **[Yao et al., 2023]** Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601. 提出允许 LLM 并行探索多条推理路径并通过投票机制选择最优方案，是 ReAct 在搜索深度维度上的扩展。

7. **[Shinn et al., 2023]** Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366. 在 ReAct 循环之外增加"反思层"，让 Agent 从失败中总结经验并重试，是 ReAct 的重要改进方向（详见第10章）。