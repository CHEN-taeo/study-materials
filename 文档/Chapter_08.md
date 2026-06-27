# 第8章 从 ReAct 到 AutoGen——Agent 架构的范式演进

**核心思想** Agent 的本质不是“调用工具”，而是“在开放环境中维持一个推理—行动—观察的闭环”；ReAct 将这一闭环内化为单模型的交错生成，AutoGen 则将其外化为多模型间的对话协议。理解二者之间的张力与互补，是设计可扩展 Agent 系统的先决条件。

**历史脉络** 20 世纪 80 年代的符号 AI（SOAR、ACT-R）已经尝试用“认知循环”来统一推理与行动，但受限于手工规则与状态空间爆炸，从未走出实验室。2010 年代的深度强化学习（DQN、A3C）用神经网络逼近价值函数，却在稀疏奖励与长程信用分配问题上反复碰壁。2022 年，大语言模型（LLM）的涌现能力让研究者重新发现：预训练模型内部已经压缩了关于世界的大量常识，问题不在于“如何学习世界模型”，而在于“如何提示模型把知识显式地展开为推理轨迹”。Chain-of-Thought [Wei et al., 2022] 率先展示了中间推理步骤对算术与符号推理的提升，但它只生成文本，不与环境交互。Toolformer [Schick et al., 2023]（详见第7章）将 API 调用嵌入自监督预训练，却受限于“只能生成一个调用且无法根据返回结果再推理”的单向范式。ReAct [Yao et al., 2022] 在 CoT 与工具调用之间架起了桥梁：它让 LLM 以自然语言交替生成 Thought（推理）与 Action（行动），并把 Observation（环境反馈）重新注入上下文，形成完整的状态机。随后，研究社区意识到复杂任务往往需要多个角色协作，AutoGen [Wu et al., 2023] 由此提出以“对话”作为多 Agent 编排的原语，把 ReAct 的单 Agent 循环扩展为群聊式的分布式控制。与此同时，Plan-and-Solve 路线质疑 ReAct 的逐步贪心策略，主张在行动前先做全局规划；而工具调用编译器（如 LLM Compiler [Kim et al., 2023]）则尝试将串行的 ReAct 轨迹编译为并行 DAG。本章的任务，是把这些看似碎片化的工作还原为一条清晰的架构演进线索，并给出可运行的工程实现。

---

## 8.1 从符号 AI 到 LLM Agent：三次范式转移

### 8.1.1 符号 Agent：推理与行动的分离

早期认知架构 SOAR [Laird, 2012] 和 ACT-R [Anderson, 1996] 把 Agent 设计为“产生式规则”的循环：感知缓冲区接收外部刺激，工作记忆匹配规则，行动执行器修改环境。这些系统的缺陷是众所周知的：规则需要人工编写，知识获取瓶颈（knowledge acquisition bottleneck）扼杀了泛化能力；更重要的是，推理与行动虽然共享一个循环，但规则语言本身不具备可微性，无法从数据中学习。符号 Agent 的“规划”是前向或后向搜索，在状态空间指数级增长时迅速失效。

### 8.1.2 深度强化学习：隐式世界模型

2015 年 DQN 在 Atari 上的成功标志着“无模型强化学习”（model-free RL）的崛起。但无模型方法并不真正理解环境，只是从像素到动作的价值函数映射。Dreamer [Hafner et al., 2020] 与 MuZero [Schrittwieser et al., 2020]（参见第12章世界模型）试图通过潜在动力学模型（latent dynamics model）进行规划，将 Agent 的“世界模型”外显为可预测下一状态的神经网络。这些工作在连续控制领域表现出色，却难以迁移到语言、知识推理等离散符号任务——状态空间不再是像素网格，而是开放文本。

### 8.1.3 LLM Agent：推理即行动，语言即接口

LLM 的出现改变了问题定义。GPT-3 及后续模型已经通过预训练获得了关于物理、社会、常识的隐式知识，问题是“如何把这些知识结构化地调用出来”。Chain-of-Thought 提示让模型在回答问题之前先输出“让我想想……”的推理链，显著提高了数学与逻辑题的准确率。但 CoT 是静态的：它生成一段文本，然后停止。如果推理过程中需要查询数据库、计算数学表达式、或者调用搜索引擎，CoT 无能为力。Toolformer 尝试在预训练阶段把 API 调用嵌入到文本生成中，但它是“一次性”的：模型生成一个工具调用，文本拼接结果，再预测后续 token，没有循环。

ReAct 的突破在于把“推理”与“行动”放进同一个自回归循环：模型的输出不再是单一的答案，而是交替出现的 Thought 与 Action。Action 被解析为工具调用，执行后得到 Observation，Observation 与 Thought 一起被追加到上下文，再输入模型生成下一个 Thought。这实际上是一个以 LLM 为“认知核”的符号-神经混合架构——LLM 负责不可微的常识推理，外部工具负责可精确执行的符号操作。

AutoGen 的进一步观察是：当任务足够复杂（例如开发一个完整的软件项目、执行多步骤数据分析），单一 Agent 的上下文窗口会被历史轨迹占满，且不同子任务需要不同的技能组合。与其让一个 Agent 扮演“万能专家”，不如让多个 Agent 分别承担规划者、执行者、代码审查员、测试员的角色，通过结构化对话完成协作。AutoGen 把 ReAct 的“内部独白”变成了“外部群聊”，并引入了对话管理、群聊路由、人机交接等协议级抽象。

---

## 8.2 ReAct：推理与行动的交替循环

### 8.2.1 形式化定义

ReAct [Yao et al., 2022] 的核心可以形式化为一个部分可观察马尔可夫决策过程（POMDP）的近似求解器。在标准 POMDP 中，Agent 维护一个信念状态（belief state）$b_t(s) = P(s_t = s \mid a_{1:t}, o_{1:t})$，并通过贝叶斯更新来融合新观测。然而，对于基于 LLM 的 Agent，显式维护连续信念状态是不现实的——LLM 的上下文窗口本质上是一个**信息状态（information state）**的近似：它把全部历史轨迹 $h_t = (a_1, o_1, \dots, a_t, o_t)$ 以文本形式编码，让自回归模型直接从中学习策略，而无需显式推断后验分布。

令 $s_t$ 为第 $t$ 步的环境状态（对 LLM 不可直接观察），$o_t$ 为第 $t$ 步的观测（Observation，通常是工具返回的文本）。Agent 的“认知状态”由上下文 $c_t$ 表示：

$$c_t = [\text{system prompt}, \text{task}, \text{ Thought}_1, \text{Action}_1, \text{Observation}_1, \dots, \text{Thought}_t, \text{Action}_t, \text{Observation}_t]$$

LLM 策略 $\pi_\theta$ 以 $c_t$ 为条件，生成下一步的 Thought 与 Action：

$$(\text{Thought}_{t+1}, \text{Action}_{t+1}) \sim \pi_\theta(\cdot \mid c_t)$$

Action 被解析为工具调用 $\tau_{t+1}$，执行后得到 Observation $o_{t+1}$，从而更新上下文。循环终止条件是 Action 为 `Finish[answer]`。

从信息论角度看，ReAct 的上下文 $c_t$ 是一个**有损压缩**的信念状态：它丢弃了数值概率，保留了自然语言描述的关键决策线索。这一近似的有效性依赖于预训练 LLM 的“世界知识”——模型已经在海量文本中见过类似的推理模式，因此能够从稀疏的文本历史中恢复出隐含的因果关系。

### 8.2.2 为什么 Thought 必须在 Action 之前

ReAct 的消融实验揭示了一个关键发现：如果去掉 Thought，只让模型生成 Action（即纯 Act 基线），成功率在 HotpotQA 和 WebShop 上分别下降 20% 与 30% 以上。如果去掉 Action，只保留纯 CoT（即模型只生成推理链，不调用工具），则无法获取实时信息，导致事实性错误率上升。Thought 的作用有三：

1. **分解**：将复杂目标拆解为子目标，例如“先搜索 A，再验证 B”。
2. **锚定**：在得到 Observation 后，解释该结果与当前任务的关系，避免后续步骤偏离主题。
3. **回溯**：当 Observation 与预期不符时，Thought 可以显式地声明“之前的假设是错误的，我需要改换策略”。

这种“推理作为行动的前置”机制，实际上让 LLM 在上下文窗口中维持了一个显式的计划栈（plan stack），尽管该栈没有硬编码的数据结构，而是以自然语言的线性序列存在。

### 8.2.3 局限性与失败模式

ReAct 并非万能。其第一个局限是**上下文线性增长**：每一步 Thought + Action + Observation 都会追加到 prompt，当任务超过 10–15 步时，上下文窗口可能被历史占满，模型注意力分散。Yao et al. 的原始实现并未引入显式的记忆压缩机制，这成为后续工作（如 MemGPT [Packer et al., 2023]）的改进方向。一个直观的数量级估计：假设每步平均消耗 300 token（100 token Thought + 50 token Action + 150 token Observation），10 步后 prompt 长度即达 3k token；如果系统提示与 few-shot 示例再占 2k token，则总 prompt 接近 5k token。对于 4k 上下文窗口的模型，这已构成硬约束；即使使用 16k 或 32k 窗口，注意力稀释（attention dilution）也会导致早期 Thought 被模型忽略。

第二个局限是**贪心局部最优**。ReAct 的每一步决策都是基于当前上下文生成最优的下一步 Action，缺乏全局回溯。如果第一步搜索的关键词就偏离了答案所在的知识区域，后续 Thought 往往只是在错误方向上不断“自我合理化”，难以跳脱。这与传统规划中的 hill-climbing 困境类似。原始论文中的失败案例显示，在 WebShop 任务中，当模型过早选择了一个错误商品类别后，后续 5–6 步的 Thought 都在试图解释该选择的合理性，而非重新检索其他类别。

第三个局限是**工具接口的脆弱性**。ReAct 要求 Action 的格式可被正则表达式精确解析，例如 `Search[term]` 或 `Calculate[expr]`。一旦模型生成 `I will search for "term"` 这种非结构化文本，解析器就会失败，导致系统崩溃。这要求精心设计的 few-shot 示例与输出格式约束（如 JSON mode 或 regex 停止条件）。更隐蔽的问题是**推理-行动错位（reasoning-action misalignment）**：模型在 Thought 中声称要执行 A，但在 Action 中却生成 B。例如 Thought 说“我需要搜索 2023 年的数据”，Action 却是 `Search[2024 data]`。这种不一致在高 temperature 或模型规模较小时尤为常见，后处理阶段无法捕获，因为 parser 只检查 Action 格式，不验证 Action 与 Thought 的一致性。

第四个局限是**成本敏感**。ReAct 每步都需调用一次 LLM，而每次调用都是完整的自回归生成。对于需要 15 步才能解决的任务，总成本是 15 次 API 调用。如果问题可以通过一次批量查询（batch query）解决，ReAct 的逐步交互就显低效。这也是为什么 Kim et al. [2023] 提出 LLM Compiler 来压缩串行轨迹为并行 DAG。

---

## 8.3 AutoGen：多 Agent 对话框架

### 8.3.1 从单循环到对话图

AutoGen [Wu et al., 2023] 的出发点是：现实世界中的复杂任务天然适合分而治之。让同一个 LLM 实例在连续对话中既扮演架构师又扮演程序员，会导致角色混淆与上下文污染。AutoGen 把每个 Agent 建模为一个三元组：

- **LLM 后端**：决定该 Agent 的“智商”与知识域（可以是 GPT-4、Claude 或本地微调模型）。
- **系统提示（System Message）**：定义角色、约束与技能边界。
- **工具集（Tools）**：该 Agent 可访问的函数签名。

多个 Agent 通过“对话”进行协作。对话不再是自由闲聊，而是带有明确路由规则的**对话图（Conversation Graph）**。最简单的图是双向对话：用户 Agent 提出需求，助手 Agent 生成回复；当助手需要调用代码执行器时，对话路由到代码执行 Agent，执行结果再返回给助手。更复杂的图是群聊（Group Chat）：一个群聊管理器（Group Chat Manager）根据当前轮次与话题，决定下一个发言者是谁。

### 8.3.2 与 ReAct 的对比

| 维度 | ReAct | AutoGen |
|------|-------|---------|
| 控制粒度 | 单 Agent 的内部循环 | 多 Agent 的外部编排 |
| 推理载体 | 自然语言 Thought | 自然语言对话 + 结构化消息 |
| 记忆位置 | 线性上下文 | 可持久化的对话历史 + 外部记忆 |
| 任务分解 | 隐式，由单 Agent 在 Thought 中完成 | 显式，由不同 Agent 分治 |
| 终止条件 | Action = Finish | 对话图到达终止节点 |
| 适用场景 | 单步/多步工具调用，问答 | 软件开发、数据分析、多角色协作 |

AutoGen 并未取代 ReAct，而是把 ReAct 的循环内嵌到每个 Agent 实例中。例如，在 AutoGen 的编码 Agent 中，当接收到“写一个排序函数”的任务时，该 Agent 内部可以运行一个 ReAct 式的子循环：Thought（“我需要先写单元测试”）→ Action（调用代码生成器）→ Observation（代码生成结果）→ Thought（“看起来正确，但缺少边界处理”）→ Action（修改代码）。AutoGen 管理的是 Agent 之间的宏观控制流，ReAct 管理的是 Agent 内部的微观推理流。

### 8.3.3 对话路由与群聊机制

AutoGen 的群聊管理器使用一个基于规则的或 LLM 驱动的路由函数。规则路由可以是轮询（round-robin）或基于关键词的触发（如包含“代码”则路由到编码 Agent）。LLM 驱动路由则把整个对话历史输入到一个“路由模型”，由其生成下一个发言者的名字。这带来了一个微妙的工程权衡：路由模型本身消耗 token，如果群聊成员众多，路由开销可能占到总成本的 15%–20%。一个具体的数量级估计：假设群聊有 4 个 Agent，每轮对话历史 2k token，路由模型生成 10 token 来选择发言者，则 10 轮群聊的路由成本为 10 × 2k × 4（因为路由模型通常需要读取所有候选 Agent 的系统提示）≈ 80k token，约等于 4 次 GPT-4 调用的成本。因此，在生产环境中，规则路由与 LLM 路由的混合策略往往更经济：先用关键词规则做粗筛，仅在歧义情况下启用 LLM 路由。

另一个设计细节是**人机交接（Human-in-the-loop）**。AutoGen 允许在对话图的任意节点插入人工审批：当 Agent 生成的代码涉及删除文件或发起网络请求时，系统暂停并等待人类确认。这种“软约束”比硬编码权限列表更灵活，但也引入了延迟与交互设计的复杂性。在实践中，人机交接的触发条件需要精心设计：过于敏感会导致用户体验断裂（用户每 30 秒就要点一次“确认”），过于宽松则失去安全意义。一个有效的折衷是“白名单 + 敏感操作触发”：常见文件读写自动通过，但 `rm -rf`、`DROP TABLE`、`转账` 等操作强制暂停。

---

## 8.4 Plan-and-Solve vs ReAct：何时需要显式规划

### 8.4.1 ReAct 的逐步贪心本质

ReAct 的每一步决策都基于当前上下文生成“下一步”的最佳行动。这相当于在搜索树上做深度优先的贪婪下降。对于“需要 3 步以内、每步结果可预期”的任务（如查询天气后计算穿衣指数），贪心策略足够高效。但对于“需要 10 步以上、早期错误会级联放大”的任务（如构建一个完整的数据分析管道），贪心策略的成本极高。

### 8.4.2 Plan-and-Solve 的架构

Plan-and-Solve 路线主张在调用任何工具之前，先让 LLM 生成一个完整的计划树（plan tree），然后按顺序或并行执行。Beyond ReAct [Wei et al., 2026] 提出了一个以 Planner 为中心的框架：Planner LLM 接收任务描述，输出一个 DAG（有向无环图），其中节点是工具调用，边是数据依赖。执行器（Executor）根据拓扑排序调度节点，可以并行调用无依赖的工具。例如，任务“比较 A 公司与 B 公司的财务数据并生成图表”可以分解为：

- 节点 1：检索 A 公司财报
- 节点 2：检索 B 公司财报
- 节点 3（依赖 1, 2）：对比分析
- 节点 4（依赖 3）：生成图表

节点 1 与 2 可以并行，节点 3 必须等待 1 与 2 完成。这种显式规划把 ReAct 的“逐步探索”转化为“先规划后执行”，显著降低了长程任务的总延迟与 token 消耗。

### 8.4.3 何时用 ReAct，何时用 Plan-and-Solve

工程上的决策规则可以归纳为：

- **任务步数 < 5，且每步结果高度不确定**（如开放式问答、交互式网页浏览）：选 ReAct。其交互性允许模型根据中间结果动态调整方向，避免了“计划赶不上变化”的僵化。
- **任务步数 > 10，且子任务间依赖关系明确**（如批量数据处理、ETL 管道、多步骤科学实验）：选 Plan-and-Solve。显式 DAG 允许并行执行、错误隔离与重试机制。
- **中间地带**：可以采用混合策略，如 ReAct 的顶层用 Plan-and-Solve 生成里程碑，每个里程碑内部用 ReAct 逐步探索。LLM Compiler [Kim et al., 2023] 正是这种混合思想的代表：它先让模型生成一个“并行函数调用计划”，然后编译为 DAG 执行，既保留了 LLM 的推理能力，又避免了逐轮调用的串行瓶颈。

---

## 8.5 Agent 架构的组件分解：感知、记忆、推理、行动、评估

尽管 ReAct 与 AutoGen 在具体实现上差异显著，现代 LLM Agent 的架构可以统一分解为五个组件。理解每个组件的设计空间，有助于在工程中根据需求进行裁剪与组合。

### 8.5.1 感知（Perception）

感知层负责将环境输入转换为 LLM 可处理的文本。在 ReAct 中，感知是工具返回的 Observation 字符串。在 AutoGen 中，感知还包括其他 Agent 发送的消息。对于多模态 Agent（如第11章将介绍的视觉-语言 Agent），感知层还需处理图像编码、音频转录等。关键设计决策是：原始感知是否经过预处理？例如，网页浏览工具返回的 HTML 通常包含大量脚本与样式噪声，直接输入 LLM 会浪费上下文；常见的做法是先用 Readability 或文本提取器做清洗，再作为 Observation。

### 8.5.2 记忆（Memory）

ReAct 的“记忆”是隐式的：全部历史都在上下文窗口中。AutoGen 则支持显式记忆——对话历史可以持久化到数据库，并在新会话中检索。更进一步，向量记忆（如 FAISS、Chroma）允许 Agent 存储和检索非结构化文档。BOLAA [Liu et al., 2023] 的基准测试表明，引入外部向量记忆后，长程多跳问答的准确率提升 12%–18%，但延迟增加约 200 ms（取决于检索 top-k）。

记忆的另一个维度是**工作记忆 vs. 长期记忆**。工作记忆对应上下文窗口中的近期内容，长期记忆对应外部存储。MemGPT 提出的分层记忆机制（类似操作系统分页）把旧对话摘要存入长期记忆，在需要时通过 LLM 生成的搜索查询进行检索。工程上，记忆层的设计需回答三个问题：

1. **写入时机**：是每步都写入，还是仅在完成子任务后写入？频繁写入保证信息完整性，但增加存储与索引开销；稀疏写入减少噪音，但可能遗漏关键中间结论。
2. **读取策略**：是简单的向量相似度检索，还是由 LLM 动态生成检索查询？后者更灵活，但每轮增加一次 LLM 调用。
3. **遗忘机制**：长期记忆是否有过期策略？在持续运行的 Agent 系统中，无限制增长的记忆库会导致检索噪音上升。常见做法是按时间衰减或按重要性打分（如 LLM 判断该信息是否对未来任务有用）。

### 8.5.3 推理（Reasoning）

推理是 Agent 的“认知核”。ReAct 使用单模型自回归推理，AutoGen 允许多模型协作推理（如一个 Agent 负责逻辑分析，另一个负责创意生成）。当前推理层的设计空间包括：

- **提示策略**：CoT、ReAct、Tree-of-Thoughts [Yao et al., 2023]（允许并行探索多条推理路径并投票）。
- **推理深度**：单步推理（直接生成 Action）vs. 多步推理（先生成 Thought 再 Action）。
- **工具增强**：是否允许模型在推理过程中调用外部符号求解器（如 Wolfram Alpha、Python 解释器）来验证数学推导。

### 8.5.4 行动（Action）

行动层是 Agent 对外部世界施加影响的接口。行动可以是工具调用（API、数据库、代码执行）、环境控制（网页点击、文件操作）或通信（向其他 Agent 发送消息）。关键设计是**行动空间的抽象级别**：低级别行动（如原始 HTTP 请求）灵活但易错，高级别行动（如 `send_email(to, subject, body)`）安全但受限。AutoGen 的函数签名机制要求每个工具提供 JSON Schema 描述，LLM 生成符合 Schema 的调用参数，这类似于 ReAct 的 `Action[params]` 格式，但结构化程度更高。

行动层还需要考虑**副作用与幂等性**。ReAct 的 Action 序列可能因解析错误或 LLM 幻觉而重复执行同一操作（如重复扣款、重复写入）。工程上，工具设计应遵循幂等原则：同一 Action 执行多次与执行一次结果相同。对于天然非幂等的操作（如支付、文件删除），应在系统层引入去重键（idempotency key）或事务回滚机制。此外，行动层需要**超时与重试策略**：工具调用可能因网络抖动失败，Agent 不应直接崩溃，而应根据错误类型（可重试 vs. 不可重试）决定下一步。

### 8.5.5 评估（Evaluation）

评估组件决定 Agent 何时停止、何时重试、何时请求人工帮助。ReAct 的评估是硬编码的：当 Action 为 `Finish` 时终止。AutoGen 的评估可以是用户定义的终止函数（如“当代码通过所有单元测试时终止”）。更复杂的评估引入 LLM-as-a-Judge：用另一个 LLM 评估当前 Agent 的输出质量，并给出改进建议。这在 Self-Refine [Madaan et al., 2023] 与 Reflexion [Shinn et al., 2023] 中得到了深入探讨（参见第10章）。

---

## 8.6 代码：完整 ReAct 循环实现

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
        # 兼容 Thought 与 Action 混排：取最后一个 Action 行
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
                # 解析失败：让 Observation 提示模型修正格式
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
        return str(eval(expr))  # 生产环境应使用 asteval 或 sympy
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
2. **解析鲁棒性**：`parse_action` 使用正则从最后一行提取，避免模型在 Thought 中无意提及 Action 关键字导致的误解析。`extract_thought` 则把 Action 之前的文本全部视为 Thought，兼容模型在中间插入换行的情况。
3. **错误注入**：当模型未生成合法 Action 时，Observation 不是静默失败，而是显式返回错误提示，让 LLM 在下一轮自我修正。这是 ReAct 鲁棒性的关键：Observation 可以是错误信息，模型可以据此调整策略。
4. **LLM 解耦**：`llm_caller` 是任意 `str -> str` 函数，读者可以无缝替换为 OpenAI 的 `chat.completions.create` 或本地 vLLM 服务。这种解耦是生产代码的必备要求。

---

## 8.7 工程实践框

> **训练与部署 ReAct/AutoGen 风格 Agent 的常见工程陷阱**
>
> **1. Prompt 格式的一致性比内容更重要**
> 在复现 ReAct 时，研究者常犯的错误是在 few-shot 示例中混用不同的 Action 格式（如 `Search[term]`、`Search: term`、`{ "tool": "Search", "arg": "term" }`）。LLM 对格式极其敏感，即使语义等价，格式不一致也会导致生成解析失败率上升 5–10 倍。建议：在系统提示中给出严格的 BNF 语法，并用 `pydantic` 或 JSON Schema 在后处理阶段做结构化校验。
>
> **2. 上下文窗口的“隐性截断”**
> 当 ReAct 轨迹超过 8k token（对于 GPT-3.5 的 16k 窗口）时，模型会忽略早期 Thought，导致任务遗忘。实践中，如果预计任务超过 10 步，应在每 5 步后插入一个“摘要 Thought”：让 LLM 用 100 字总结当前进展与剩余计划，然后丢弃中间细节。这会将有效步数扩展 2–3 倍，代价是每步增加一次 LLM 调用（用于摘要）。
>
> **3. 工具延迟与超时设计**
> 在 AutoGen 多 Agent 系统中，工具调用（如代码执行、网络请求）的延迟差异巨大。如果一个 Agent 调用了耗时 30 秒的数据库查询，整个对话图会阻塞。工程上应采用异步消息队列：每个 Agent 的 Action 被提交到队列，执行完成后通过回调触发下一轮 LLM 调用。Python 的 `asyncio` 或 `Celery` 是常见的选择。
>
> **4. 超参选择：temperature 与 max_tokens**
> - ReAct 的 Thought 生成：temperature=0.1–0.3，max_tokens=256–512。低温保证推理路径的确定性，避免模型在 Thought 中发散到无关话题。
> - Action 生成：temperature=0。Action 必须可解析，任何随机性都会增加格式错误率。
> - 对于需要创意的 AutoGen 角色（如头脑风暴 Agent），temperature 可提高到 0.7–1.0，但 Action 解析仍需约束输出格式（如通过 JSON mode 或 constrained decoding）。
>
> **5. 硬件需求估算**
> - **单 ReAct Agent**：如果使用 GPT-4 API，瓶颈不在本地 GPU，而在网络延迟与 token 成本。每步约消耗 1k–2k prompt token + 200–500 completion token，按 2024 年 GPT-4 定价，10 步任务成本约 $0.03–$0.10。
> - **本地 LLM 后端**：若用 Llama-3-70B 或 Qwen-72B 作为本地 Agent 后端，推理需 2×40GB A100（FP16）或 1×48GB A6000（AWQ 4-bit 量化）。量化后的 4-bit 模型在 ReAct 场景下几乎不损失准确率，但 latency 增加约 30%。
> - **AutoGen 多 Agent 并发**：当 4 个 Agent 同时活跃时，如果每个都挂载 70B 模型，需要至少 4×2 A100 = 8 张 A100。更经济的方案是“小模型分角色”：Planner 用 70B，Coder 用 13B–34B，Reviewer 用 7B，通过 vLLM 的 PagedAttention 共享 KV Cache 来降低显存碎片。
>
> **6. 评估指标的陷阱**
> 不要只用“最终答案正确率”衡量 Agent。ReAct 论文报告了 **success rate**（是否到达正确终止状态）、**trajectory length**（步数，反映效率）与 **hallucination rate**（Thought 中是否包含虚假事实）。在 AutoGen 中，还需增加 **conversation turns**（轮数，反映协作效率）与 **human intervention count**（需要人工救场的次数）。只优化成功率会导致 Agent 生成冗长的保守轨迹，牺牲用户体验。
>
> **7. 安全边界**
> ReAct 的 Action 空间如果包含文件系统或网络操作，必须 sandbox。推荐方案：代码执行使用 `gVisor` 或 `Firecracker` 微虚拟机；网络请求通过代理白名单过滤；敏感操作（如删除、转账）强制触发 human-in-the-loop。AutoGen 的 `human_input_mode` 配置项应在生产环境中默认开启 `ALWAYS`，仅在内部测试时关闭。

---

## 8.8 小结

本章从 ReAct 的单 Agent 推理—行动循环出发，延伸到 AutoGen 的多 Agent 对话编排，并对比了 ReAct 与 Plan-and-Solve 两种策略的适用边界。核心结论可概括为三点：

1. **ReAct 的价值在于把 LLM 的常识推理外化为可追踪、可干预的符号轨迹**。它不是一个新算法，而是一种“把 LLM 当作认知核”的系统工程范式。其设计精髓是 Thought 的显式化：让模型在生成 Action 之前先“说出来”它想做什么，这既提高了可解释性，又为错误诊断提供了抓手。

2. **AutoGen 的价值在于把 ReAct 的循环从单进程扩展到分布式进程**。当任务复杂度超过单 Agent 的上下文承载能力时，角色分解与对话路由是更 scalable 的架构。但多 Agent 不是免费的：路由开销、状态同步、消息序列化都会增加系统复杂度与延迟。

3. **没有 universally optimal 的 Agent 架构**。ReAct 适合短程、动态、不确定的任务；Plan-and-Solve 适合长程、结构化、依赖明确的任务。生产系统往往采用混合架构：Planner 生成 DAG 里程碑，每个节点内部用 ReAct 探索，最终通过 Evaluator 组件验证结果。这种“分层控制”的思想将在第12章（世界模型与规划）与第14章（多 Agent 系统安全）中进一步深化。

---

## 延伸阅读

1. **[Yao et al., 2022]** ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. 本章的基石论文，首次系统展示了 Thought 与 Action 的交错生成对决策任务的增益，并提供了 HotpotQA、Fever、WebShop 等基准的完整实验对比。

2. **[Wu et al., 2023]** AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155. AutoGen 框架的原始论文，详细定义了 Conversable Agent、Conversation Programming、Group Chat Manager 等核心抽象，并提供了代码库与多个应用案例（数学、编程、网页浏览）。

3. **[Kim et al., 2023]** An LLM Compiler for Parallel Function Calling. arXiv:2312.04511. 提出将 LLM 生成的串行工具调用编译为并行 DAG 执行，是 ReAct 与 Plan-and-Solve 混合路线的重要实践。论文中的“LLM Planner + Parallel Executor”架构对降低多步任务的端到端延迟具有直接工程价值。

4. **[Wei et al., 2026]** Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning. Proceedings of AAAI. 系统论证了纯 ReAct 在长程任务中的局部最优陷阱，提出以显式 Planner 生成 DAG 的替代方案，并给出了与 ReAct 在科学计算、数据分析任务上的对比实验。

5. **[Liu et al., 2023]** BOLAA: Benchmarking and Orchestrating LLM-Augmented Autonomous Agents. arXiv:2308.05960. 对 LLM Agent 的架构维度（单 Agent vs. 多 Agent、不同记忆机制、不同工具集）进行了系统基准测试，为 Agent 设计提供了量化参考。
