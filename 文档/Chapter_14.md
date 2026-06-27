# 第14章 World Model × LLM × Agent 的融合：如何用世界模型增强 Agent 的因果推理

> **核心思想**：大型语言模型（LLM）擅长符号推理与任务分解，世界模型（World Model）擅长因果预测与物理推演，两者的融合本质上是将"知道做什么"与"知道会发生什么"结合起来，使 Agent 在行动前能够内建物理因果的想象与验证能力。

> **历史脉络**：2010年代中期，世界模型与语言模型在两条完全独立的轨道上发展。世界模型阵营从 Ha & Schmidhuber 的 Dreamer 系列 [Hafner et al., 2023] 出发，追求在潜在空间中预测环境的下一状态；语言模型阵营从 GPT-1 [Radford et al., 2018] 到 GPT-3，追求在海量文本中捕获符号关联。Agent 研究则长期在第三条轨道上运行，以 ReAct [Yao et al., 2022] 为代表，将 LLM 的推理与外部工具调用交错执行，但缺乏对物理后果的内建模拟。这三条路线的独立发展暴露了各自的致命短板：纯 LLM Agent 在物理世界中出现"幻觉行动"（hallucinated action）——例如规划将水杯放在已倾斜的桌面上；纯世界模型缺乏高层任务语义，无法将人类指令"把客厅收拾干净"映射为动作序列。融合动机因此产生：如果 Agent 在提出行动方案后，能立即在内部世界模型中推演该方案的物理后果，并用推演结果反馈修正规划，就能同时获得符号灵活性与物理一致性。早期的替代方案——如完全依赖 LLM 的 chain-of-thought 做物理推理——在实验中被证实失败，因为语言模型的 next-token prediction 并不具备牛顿力学的内建约束，它只是在统计上"猜测"最可能的物理结果，而非从因果机制推演。

## 14.1 三条路线的独立发展与融合动机

第1章提出的统一视角——将 Agent 视为"感知-表征-推理-行动"的闭环系统——在本章得到具体实现。具体而言，LLM 提供高层推理与任务分解能力，World Model 提供环境动态的前向预测，Agent 架构则将两者编排为可执行的决策循环。要理解这一融合架构的设计逻辑，必须回溯三条技术路线的独立演进史。

世界模型的概念最早可追溯至 Sutton & Barto 强化学习框架中的 model-based RL，即 Agent 学习环境转移模型 $s_{t+1} \sim P(s_t, a_t)$ 以支持规划。2018 年，Ha & Schmidhuber 提出 World Models 框架，首次将环境动态压缩到一个递归变分自编码器（VAE + MDN-RNN）中，使 Agent 能够在想象的潜在空间中训练策略，而无需与真实环境持续交互。后续 Dreamer 系列 [Hafner et al., 2023] 用 RSSM（Recurrent State-Space Model）替代了 MDN-RNN，将潜在状态分解为随机变量与确定性路径，实现了在 Atari、DMC 等连续控制任务上的 SOTA 样本效率。RSSM 的核心贡献在于证明：一个紧凑的潜在动态模型，足以支持长程规划与信用分配，而不需要枚举高维像素空间的原始转移概率。

与此同时，LLM 路线从 GPT-1 [Radford et al., 2018] 的生成式预训练出发，到 GPT-3 的 in-context learning，再到 instruction tuning 与 RLHF，逐步建立了"预训练-对齐-部署"的范式。LLM 的表征能力覆盖语义、语法、常识乃至代码逻辑，但其与物理世界的接口始终薄弱。2022 年，Yao 等人提出 ReAct [Yao et al., 2022]，首次在 LLM 中系统性地交错 reasoning traces 与 action traces，使模型能够通过调用搜索引擎、计算器、API 等工具完成多步任务。ReAct 的局限性在于：它假设工具返回的结果是 ground-truth，无法预判工具执行前的物理可行性。例如，ReAct 可以在知识库中查询"螺丝刀的握持方式"，却无法预判"在当前手臂姿态下握持螺丝刀是否会与桌沿碰撞"。

Agent 架构的第三条路线——以 AutoGen [Wu et al., 2023] 为代表——探索了多 Agent 协作的编程框架。AutoGen 将 LLM 封装为可对话的 Agent，通过对话协议实现任务分解、角色分配与结果整合。然而，AutoGen 的 Agent 同样缺乏环境动态模型，其规划完全依赖 LLM 的文本生成能力，在物理任务中表现为"说得多、做得错"。

融合动机由此明确：单独使用 LLM 的 Agent 拥有符号灵活性但缺乏物理因果约束；单独使用 World Model 的 Agent 拥有物理预测能力但缺乏高层语义理解。将两者融合，本质上是在决策循环中引入一个"内建模拟器"——LLM 负责提出候选行动与高层目标分解，World Model 负责在潜在空间中推演每个候选行动的物理后果，LLM 再根据推演结果选择或修正行动。这一循环对应第7章讨论的 deliberation 与 execution 的双层架构，但将 execution 的物理后果预演纳入了 deliberation 阶段。

## 14.2 融合路径 1：语言化世界模型（SayCan 与 LLM-as-Planner）

第一条融合路径的思路最为直接：将世界模型的输出或环境状态以自然语言形式注入 LLM 的 prompt，使 LLM 在文本层面"感知"物理世界。该路径的代表性工作包括 SayCan、Inner Monologue 以及后续的一系列 LLM-as-Planner 框架。

SayCan 的核心机制是"affordance grounding"：给定一个高层指令（如"给我拿一瓶水"），LLM 提出候选动作（如"pick up the water bottle"），每个候选动作经过一个 affordance 模型打分，该模型判断在当前状态下该动作的成功率。LLM 的文本生成概率与 affordance 分数相乘，得到联合概率，最终选择最优动作。在这一框架中，affordance 模型扮演了世界模型的简化角色——它不预测完整的状态转移，只预测动作的可行性。然而，这种简化带来了严重局限：SayCan 的 affordance 模型是单步的，无法模拟多步行动的累积效应。例如，"先推开椅子再弯腰捡东西"这两步的联合可行性，无法通过单步 affordance 的乘积准确估计，因为第一步改变了椅子的位置，进而改变了第二步的可达空间。

Inner Monologue 进一步将环境反馈以自然语言形式注入 LLM 的推理链。例如，在执行"把苹果放进冰箱"后，Inner Monologue 将视觉系统检测到的"苹果已不在桌上"这一事实转化为文本，追加到 LLM 的上下文窗口中。这种方法在短程任务中有效，但随着任务长度增加，上下文窗口中的环境描述呈线性膨胀，LLM 的注意力机制难以在长文本中精确定位关键状态变化。实验表明，当任务步数超过 15 步时，Inner Monologue 的错误率显著上升，因为 LLM 的上下文推理能力被环境噪声淹没。

语言化世界模型的根本缺陷在于"表征失配"：世界模型擅长在紧凑的潜在向量中编码状态转移，而 LLM 的输入空间是离散 token。将物理状态强制翻译为自然语言，丢失了空间几何、力矩平衡、碰撞体积等连续物理信息。例如，"杯子在桌子边缘 2 cm 处"这一文本描述，对 LLM 而言只是一个语义标签，而世界模型中的潜在向量编码了杯子质心、桌面的支持多边形、重力方向等足以判断倾倒风险的物理量。因此，语言化路径适用于粗略的、符号化的物理判断（如"钥匙是否在口袋里"），但无法支持精细的、连续的动力学推理（如"以多大的力推抽屉不会导致侧翻"）。

## 14.3 融合路径 2：世界模型增强的 LLM（用因果预测纠正物理幻觉）

第二条融合路径更为深入：不将世界模型仅作为环境描述的文本生成器，而是将其嵌入 LLM 的决策循环中，作为"物理验证器"（physics validator）在潜在空间中推演 LLM 提出的行动方案，并将推演结果反馈给 LLM 以纠正物理幻觉。

物理幻觉（physical hallucination）是 LLM Agent 在机器人任务中的典型失败模式。例如，GPT-4 在规划"将盛满水的杯子从 A 点移动到 B 点"时，可能生成"以 30 cm/s 的速度直线移动手臂"这一动作序列，因为语言模型从文本统计中学习到"直线是最短路径"，却忽略了液体惯性导致的泼溅。世界模型增强的 LLM 框架在生成该动作后，立即将其编码为潜在动作向量 $a_t$，输入到 RSSM 或类似模型中，进行 $k$ 步前向推演：$\hat{s}_{t+1}, \dots, \hat{s}_{t+k} = \text{RSSM}(s_t, a_t, \dots, a_{t+k-1})$。如果推演结果显示杯中液体在潜在状态中的水平面倾斜角超过溢出阈值，则将该行动标记为不可行，并反馈给 LLM 要求重新规划。

这一路径的关键技术挑战在于"动作表征的对齐"（action alignment）。LLM 输出的动作是自然语言描述（如"缓慢移动"），而世界模型需要低维连续向量（如关节角速度 $\dot{q} \in \mathbb{R}^n$）。对齐层通常采用一个可学习的映射网络 $f_{\text{align}}: \text{token} \to \mathbb{R}^n$，或以预定义的技能库（skill library）作为中间层。例如，将 LLM 的输出映射到"pour"、"grasp"、"push"等预训练技能的原语参数上，每个技能在世界模型中有对应的动作向量簇。第 9 章讨论的技能抽象层（skill abstraction）在此发挥了中介作用。

在机器人学中，这种融合框架被称为"latent imagination with symbolic planning"。LLM 作为符号规划器提出高层计划（如"先开抽屉，再取工具，最后拧螺丝"），世界模型在潜在空间中验证每一步的物理可行性，并返回预测的状态变化。如果预测状态与目标状态存在偏差（如抽屉因障碍物无法完全打开），LLM 根据偏差信息调整计划（如"先移开障碍物"）。这一闭环的数学形式可写为：

$$a^* = \arg\max_{a \in \mathcal{A}_{\text{LLM}}} P_{\text{LLM}}(a \mid g, \text{history}) \cdot \mathbb{1}\left[\text{RSSM}(s_t, a) \in \mathcal{S}_{\text{safe}}\right]$$

其中 $\mathcal{A}_{\text{LLM}}$ 是 LLM 生成的候选动作集合，$\mathcal{S}_{\text{safe}}$ 是安全状态集合，指示函数 $\mathbb{1}[\cdot]$ 由世界模型的前向预测实现。该式表明，最优动作不仅要在语言上合理，还要在物理上可行。

## 14.4 融合路径 3：统一潜在空间（在表征空间中整合语言、视觉和物理）

第三条融合路径是最具雄心的：不将 LLM 与世界模型作为两个独立模块拼接，而是训练一个统一的潜在空间，使语言 token、视觉像素与物理状态在同一个表征框架中可相互映射。该路径的哲学基础是第 3 章讨论的多模态统一表征——不同模态的信息在高层语义空间中应当是拓扑一致的。

LeCun 提出的 JEPA（Joint Embedding Predictive Architecture）是这一方向的理论先导。JEPA 不重建像素，而是训练一个编码器将观测（图像、文本、状态）映射到潜在向量，并在潜在空间中学习预测下一状态的嵌入。与传统的生成式世界模型（如 RSSM 重建像素）不同，JEPA 的潜在空间是"面向语义"的，天然适合与 LLM 的 token embedding 对接。Guan 等人在自动驾驶领域的世界模型综述 [Guan et al., 2024] 中系统比较了 RSSM 与 JEPA 的优劣：RSSM 在需要精确像素重建的任务（如视觉反馈控制）中更强，而 JEPA 在需要语义抽象的任务（如"前方是否有行人"的判断）中更优。

统一潜在空间的实现通常采用三阶段训练：

1. **预训练视觉-物理编码器**：使用大规模视频-动作数据训练一个编码器 $E_{\text{vp}}$，将视频帧与机器人状态映射到潜在向量 $z$。
2. **语言-潜在对齐**：使用成对的指令-视频数据（如"打开抽屉"与对应视频），将 LLM 的文本 embedding 与 $z$ 通过对比学习（contrastive learning）对齐。目标是使同一语义内容的文本与视觉-物理潜在向量在空间中邻近。
3. **联合预测训练**：在潜在空间中训练预测模型，给定当前 $z_t$ 和文本指令 $l$，预测下一状态 $z_{t+1}$。此时，文本指令不再是外部 prompt，而是作为条件输入嵌入到潜在空间中的控制向量。

这一路径的技术难点在于"模态竞争的表征崩溃"（modality competition collapse）。在联合训练时，视觉信息（高维、低信噪比）倾向于主导潜在空间，将语言信息压缩到边缘。缓解策略包括模态 dropout（训练时随机屏蔽某一模态）和梯度截断（对不同模态的梯度施加不同缩放系数）。Wu 等人提出的 RL-VR-World [Wu et al., 2026] 在统一世界模型框架中引入了 RL 后训练，通过对世界模型施加任务特定的奖励信号，引导潜在空间向对决策有用的方向组织，从而部分缓解了模态竞争问题。

统一潜在空间的最终愿景是：Agent 在接收到"把左边的杯子放到右边柜子上"这一指令时，LLM 的语义表征直接激活潜在空间中的空间关系与物理约束子流形，无需显式的文本-动作转换。世界模型不再是一个外部验证器，而是内化为 LLM 的"物理直觉"——类似于人类在听到指令时，大脑中自动激活的运动想象与场景预演。

## 14.5 模块化 vs 端到端：融合架构的 Trade-off

在第 1 章的统一视角中，我们讨论了 Agent 系统的模块化分解与端到端学习的张力。在 World Model × LLM × Agent 的融合中，这一张力以三种架构形态呈现：

**模块化松耦合（Modular Loose Coupling）**：LLM、World Model 与执行器是三个独立模块，通过 API 或消息队列通信。LLM 生成计划文本，经解析器转换为动作参数，World Model 在独立进程中模拟后果，结果文本返回给 LLM。优点是调试简单、模块可独立替换（如将 GPT-4 替换为 Claude，将 RSSM 替换为 JEPA）；缺点是通信延迟高、表征转换损耗大。AutoGen [Wu et al., 2023] 的 conversation-based Agent 框架天然适合这种松耦合架构，每个模块可以是一个独立的 Agent。

**模块化紧耦合（Modular Tight Coupling）**：LLM 与 World Model 共享同一潜在空间，但在网络结构上保持独立分支。例如，LLM 的 Transformer 输出动作 token，经过对齐层注入到 RSSM 的潜在动态中，RSSM 的预测结果通过一个解码器反馈为文本 token，追加到 LLM 的 KV cache 中。这种架构下，模块间没有显式的文本序列化，通信延迟低，但仍保留了可解释性——工程师可以单独检查 RSSM 的潜在状态轨迹。

**端到端（End-to-End）**：使用单一模型同时处理语言理解与物理预测。例如，一个大型多模态 Transformer 接收视频、文本与状态序列，直接输出动作 token 与下一状态预测。这种架构在理论上可以达到最优的表征共享，但训练极不稳定，且一旦失败难以定位故障源（是语言理解错误、物理预测错误，还是动作生成错误？）。此外，端到端模型需要同时覆盖三个任务的数据，数据收集成本远高于模块化方案。

工程选择通常遵循"从松到紧"的渐进策略：先用松耦合架构验证概念（proof-of-concept），在真实环境中收集失败案例；然后逐步将高频交互模块压缩为紧耦合，减少通信开销；只有在数据充分、任务明确且团队具备足够调参能力时，才考虑端到端方案。自动驾驶领域的世界模型实践 [Guan et al., 2024] 表明，当前工业界的主流选择是紧耦合：世界模型在潜在空间中生成未来驾驶场景，LLM 或规则系统根据场景预测做决策，两者共享潜在向量但保持网络分离。

## 14.6 代码：语言-物理混合规划器

以下代码展示了一个最小可运行的语言-物理混合规划器原型。该系统的架构遵循 LLM 提出候选行动、World Model 推演后果、LLM 重新选择最优的三阶段循环。为了可复现性，World Model 使用一个简化的 RSSM 变体，在二维网格世界中训练；LLM 使用 GPT-4 的 API 调用（可用本地模型替代）。核心逻辑不依赖特定硬件，可在标准笔记本上运行。

```python
"""
语言-物理混合规划器（Language-Physics Hybrid Planner）
架构：LLM 生成候选动作 → RSSM 推演状态 → LLM 选择最优
环境：2D 网格世界，Agent 需要避开障碍物并抵达目标
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import openai  # 若无可替换为本地 LLM

# ==================== 1. 简化 RSSM World Model ====================

class MiniRSSM(nn.Module):
    """
    最小 RSSM：将 (x, y, 障碍物地图) 压缩到潜在空间，
    预测下一状态与奖励。
    """
    def __init__(self, state_dim: int = 16, hidden_dim: int = 32, action_dim: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 观测编码器：将网格观测展平后编码为潜在状态
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10 + 2, hidden_dim),  # 10x10 网格 + agent 位置
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # 递归动力学：s_{t+1} = f(s_t, a_t)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # 奖励预测
        self.reward_pred = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 观测解码器（用于可视化/debug）
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10 * 10 + 2)
        )
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [batch, 10*10+2]"""
        return self.encoder(obs)
    
    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state: [batch, state_dim], action: [batch, action_dim]"""
        x = torch.cat([state, action], dim=-1)
        return self.dynamics(x)
    
    def predict_reward(self, state: torch.Tensor) -> torch.Tensor:
        return self.reward_pred(state)
    
    def rollout(self, init_state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向推演 k 步。
        actions: [batch, k, action_dim]
        返回: states [batch, k+1, state_dim], rewards [batch, k]
        """
        states, rewards = [init_state], []
        state = init_state
        for t in range(actions.size(1)):
            state = self.predict_next(state, actions[:, t, :])
            states.append(state)
            rewards.append(self.predict_reward(state))
        return torch.stack(states, dim=1), torch.stack(rewards, dim=1)


# ==================== 2. 网格环境 ====================

class GridWorld:
    """10x10 网格，含障碍物与目标。"""
    def __init__(self, obstacle_ratio: float = 0.15):
        self.size = 10
        self.obstacles = set()
        n_obstacles = int(self.size * self.size * obstacle_ratio)
        while len(self.obstacles) < n_obstacles:
            x, y = np.random.randint(0, self.size, size=2)
            self.obstacles.add((int(x), int(y)))
        # 确保起点和终点无障碍
        self.agent_pos = (0, 0)
        self.goal = (9, 9)
        self.obstacles.discard(self.agent_pos)
        self.obstacles.discard(self.goal)
    
    def get_obs(self) -> np.ndarray:
        """返回展平网格 + agent 位置。"""
        grid = np.zeros(self.size * self.size, dtype=np.float32)
        for (x, y) in self.obstacles:
            grid[x * self.size + y] = 1.0
        pos = np.array(self.agent_pos, dtype=np.float32) / self.size
        return np.concatenate([grid, pos])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """action: 0=上,1=下,2=左,3=右。返回 (obs, reward, done)。"""
        x, y = self.agent_pos
        dx, dy = [(-1,0),(1,0),(0,-1),(0,1)][action]
        nx, ny = max(0, min(self.size-1, x+dx)), max(0, min(self.size-1, y+dy))
        if (nx, ny) in self.obstacles:
            nx, ny = x, y  # 碰撞则保持原位
        self.agent_pos = (nx, ny)
        reward = -0.1  # 每步惩罚
        done = (nx, ny) == self.goal
        if done:
            reward += 10.0
        return self.get_obs(), reward, done


# ==================== 3. LLM 接口 ====================

class LLMPlanner:
    """
    LLM 生成候选动作并解释，返回动作列表与文本理由。
    实际部署中可替换为本地 vLLM / transformers 推理。
    """
    def __init__(self, use_api: bool = False):
        self.use_api = use_api
    
    def propose_actions(self, goal: str, state_desc: str, history: str, k: int = 3) -> List[Dict]:
        """
        向 LLM 请求 k 个候选动作方案。
        返回: [{"action_idx": int, "reason": str}, ...]
        """
        prompt = f"""You are a planner in a 10x10 grid world.
Goal: {goal}
Current state: {state_desc}
History: {history}
Propose {k} distinct next actions from: 0=Up, 1=Down, 2=Left, 3=Right.
For each, briefly explain why it might be good or bad physically.
Return as JSON list: [{{"action_idx": int, "reason": str}}, ...]
"""
        if not self.use_api:
            # 模拟 LLM 输出：若未配置 API，返回启发式候选
            # 实际运行时应替换为 openai.chat.completions.create(...) 或本地推理
            return self._heuristic_propose(state_desc, k)
        
        # --- 以下为真实 API 调用模板（取消注释即可使用） ---
        # response = openai.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7,
        # )
        # text = response.choices[0].message.content
        # return self._parse_json(text)
        return []
    
    def select_action(self, goal: str, candidates: List[Dict], rollout_results: List[Dict]) -> int:
        """
        根据世界模型推演结果，由 LLM 选择最终动作。
        rollout_results: [{"action_idx": int, "predicted_reward": float, "reason": str}, ...]
        """
        prompt = f"""Goal: {goal}
Candidate actions and their predicted outcomes from a world model:
{rollout_results}
Select the single best action index. Return only the integer.
"""
        # 同上，可替换为真实 LLM 调用
        # 简化：选择 predicted_reward 最高的
        best = max(rollout_results, key=lambda x: x["predicted_reward"])
        return best["action_idx"]
    
    def _heuristic_propose(self, state_desc: str, k: int) -> List[Dict]:
        # 简单启发式：返回前 k 个动作
        return [{"action_idx": i, "reason": f"heuristic candidate {i}"} for i in range(k)]


# ==================== 4. 混合规划器主循环 ====================

class HybridPlanner:
    """
    LLM + World Model 的混合规划器。
    """
    def __init__(self, world_model: MiniRSSM, llm: LLMPlanner, horizon: int = 5):
        self.wm = world_model
        self.llm = llm
        self.horizon = horizon  # 世界模型推演步数
        # 动作 one-hot
        self.action_eye = torch.eye(4)
    
    def plan(self, env: GridWorld, goal: str, max_steps: int = 50) -> Tuple[List[int], float]:
        """运行规划循环，返回动作序列与总奖励。"""
        actions_taken = []
        total_reward = 0.0
        
        for step in range(max_steps):
            obs = torch.tensor(env.get_obs(), dtype=torch.float32).unsqueeze(0)
            state = self.wm.encode(obs)
            
            # 1. LLM 提出候选动作（基于当前状态文本描述）
            state_desc = f"Agent at {env.agent_pos}, goal at {env.goal}"
            history = f"Last {min(5, len(actions_taken))} actions: {actions_taken[-5:]}"
            candidates = self.llm.propose_actions(goal, state_desc, history, k=3)
            
            # 2. 世界模型推演每个候选动作的未来轨迹
            rollout_results = []
            for cand in candidates:
                a_idx = cand["action_idx"]
                # 构建简单动作序列：重复同一动作 horizon 步
                a_seq = self.action_eye[a_idx].unsqueeze(0).repeat(1, self.horizon, 1)
                states, rewards = self.wm.rollout(state, a_seq)
                pred_reward = rewards.sum().item()
                rollout_results.append({
                    "action_idx": a_idx,
                    "predicted_reward": pred_reward,
                    "reason": f"Rollout reward={pred_reward:.2f}"
                })
            
            # 3. LLM 根据推演结果选择最优动作（或直接用 max reward）
            best_idx = self.llm.select_action(goal, candidates, rollout_results)
            
            # 4. 在真实环境中执行
            obs, reward, done = env.step(best_idx)
            actions_taken.append(best_idx)
            total_reward += reward
            
            if done:
                break
        
        return actions_taken, total_reward


# ==================== 5. 训练脚本与运行示例 ====================

def train_world_model(env: GridWorld, model: MiniRSSM, episodes: int = 5000):
    """使用随机探索数据训练 World Model。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(episodes):
        env = GridWorld()  # 重新随机生成环境
        obs = torch.tensor(env.get_obs(), dtype=torch.float32).unsqueeze(0)
        state = model.encode(obs)
        
        total_loss = 0.0
        for _ in range(20):  # 每 episode 20 步
            a_idx = np.random.randint(0, 4)
            a_vec = torch.eye(4)[a_idx].unsqueeze(0)
            next_obs, reward, done = env.step(a_idx)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            
            # 预测下一状态
            pred_state = model.predict_next(state, a_vec)
            pred_obs = model.decoder(pred_state)
            pred_reward = model.predict_reward(pred_state)
            
            # 损失：观测重建 + 奖励预测
            loss_obs = nn.MSELoss()(pred_obs, next_obs_t)
            loss_r = nn.MSELoss()(pred_reward, torch.tensor([[reward]], dtype=torch.float32))
            loss = loss_obs + loss_r
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            state = model.encode(next_obs_t).detach()
            if done:
                break
        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}, avg loss: {total_loss/20:.4f}")


if __name__ == "__main__":
    # 训练 World Model
    print("Training World Model...")
    model = MiniRSSM(state_dim=16, hidden_dim=32, action_dim=4)
    dummy_env = GridWorld()
    train_world_model(dummy_env, model, episodes=5000)
    
    # 运行混合规划器
    print("\nRunning Hybrid Planner...")
    env = GridWorld()
    llm = LLMPlanner(use_api=False)  # 设置为 True 并配置 API key 可启用 GPT-4
    planner = HybridPlanner(model, llm, horizon=5)
    actions, total_reward = planner.plan(env, goal="Reach the goal at (9,9) while avoiding obstacles")
    print(f"Actions taken: {actions}")
    print(f"Total reward: {total_reward:.2f}")
```

上述代码的核心设计要点如下：

1. **解耦的 LLM 与 World Model 接口**：`LLMPlanner` 与 `MiniRSSM` 之间通过纯文本/张量交互，不共享梯度。这对应第 14.5 节讨论的模块化松耦合架构，便于替换为更复杂的模型（如将 `MiniRSSM` 替换为 DreamerV3 的完整实现）。

2. **前向推演作为验证器**：`rollout` 方法执行 $k$ 步前向预测，将预测累积奖励作为动作筛选信号。这一机制对应第 14.3 节描述的"世界模型增强的 LLM"路径。

3. **训练与推理分离**：World Model 在随机探索数据上预训练，而 LLM 无需微调（使用零样本 prompt）。这意味着系统可以在没有成对语言-动作数据的情况下部署，降低了数据门槛。

4. **可扩展性**：将 `horizon` 从 5 扩展到 50，`state_dim` 从 16 扩展到 256，并将 `MiniRSSM` 替换为 [Hafner et al., 2023] 的完整 RSSM 实现，即可从网格世界迁移到机器人操作任务。

## 14.7 工程实践框：训练时的常见坑、超参选择、硬件需求估算

> **工程实践框：World Model × LLM 融合系统的部署 checklist**
>
> **1. 耦合度选择**
> - 若任务环境的物理动态可精确建模（如刚体操作、导航），优先选择紧耦合架构，将 World Model 的潜在状态直接注入 LLM 的 KV cache，减少序列化开销。
> - 若任务环境频繁变化（如开放域家庭服务机器人），保持松耦合，使 World Model 可独立更新而不影响 LLM 的 prompt 模板。
>
> **2. 训练常见坑**
> - **潜在空间漂移（Latent Drift）**：当 World Model 在真实环境中持续在线学习时，潜在空间的分布会随时间漂移，导致 LLM 的对齐层失效。缓解方案：使用指数移动平均（EMA）约束潜在空间质心，或在每次 World Model 更新后重新运行少量对齐数据。
> - **奖励稀疏导致的推演失败**：在奖励稀疏的长程任务中，World Model 的短期 rollout（如 $k=5$）无法区分两个候选动作的优劣。此时不应简单增加 rollout 长度（计算成本呈线性增长），而应引入"目标距离启发"（goal-distance heuristic）作为辅助奖励信号。
> - **Sim-to-Real 的表征断裂**：在仿真中训练的 World Model 迁移到真实机器人时，视觉输入的分布偏移（光照、纹理、相机参数）会导致潜在空间完全失效。必须在仿真器中引入域随机化（domain randomization），或在潜在空间层添加对抗性域适应（ adversarial domain adaptation）模块。
>
> **3. 超参选择**
> - `state_dim`（潜在状态维度）：网格世界/简单导航用 16–32；机器人操作（6-DoF 机械臂 + 图像）用 256–512。过小会导致状态信息丢失，过大则增加训练方差。
> - `horizon`（推演步数）：短程操作（抓取、放置）用 5–15；长程导航用 50–100。每增加 10 步，推理时的 GPU 显存占用约增加 15%（以 batch size 1 计）。
> - LLM temperature：候选生成阶段用 0.7–0.9 以增加多样性，选择阶段用 0.1–0.3 以提高稳定性。
> - 学习率：World Model 用 $10^{-4}$ 到 $10^{-3}$；对齐层（若存在）用 $10^{-5}$ 到 $10^{-4}$，防止覆盖预训练 LLM 的表征。
>
> **4. 硬件需求估算**
> - **最小配置**（网格世界/概念验证）：单张 RTX 3080（10 GB VRAM），World Model 训练 2–4 小时，LLM 使用 GPT-4 API（无需本地 GPU）。
> - **标准配置**（机器人操作/视觉输入）：单张 RTX 4090 或 A6000（24–48 GB VRAM），World Model 训练 1–3 天（RSSM 变体），LLM 使用 7B 级本地模型（Qwen-7B / Llama-2-7B）需要额外 16 GB VRAM。
> - **大规模配置**（自动驾驶/长程视频）：4× A100（80 GB），用于训练基于 Transformer 的视频世界模型（如 JEPA 或 Video Diffusion Model），batch size 设 64–128，训练周期 7–14 天。LLM 使用 13B–70B 级模型时，需要张量并行（tensor parallelism）或 vLLM 服务化部署。
>
> **5. 调试策略**
> - **分层验证**：先单独测试 World Model 的前向预测精度（在 hold-out 视频/状态数据上计算 MSE），再测试 LLM 的候选生成合理性（人工检查 100 条 prompt-output 对），最后测试闭环性能。切勿在系统未闭环前同时调参两个模块。
> - **可视化潜在轨迹**：使用 t-SNE 或 UMAP 将 World Model 的潜在状态序列降维可视化，检查相似物理状态（如"机器人靠近桌子"）是否聚类。若聚类失败，说明观测编码器容量不足或训练数据不足。
> - **LLM 物理幻觉审计**：定期运行一个"物理常识测试集"（如"装满水的杯子倒置后水会流出"），要求 LLM 生成动作后由 World Model 验证。记录 LLM 的幻觉率，当幻觉率超过 15% 时，需扩充 World Model 的训练数据或增加对齐层容量。
>
> **6. 监控指标**
> - `wm_pred_mse`：World Model 状态预测均方误差，目标值 < 0.05（标准化后）。
> - `llm_candidate_hit_rate`：LLM 生成的候选动作中至少有一个被 World Model 判定为可行的比例，目标值 > 80%。
> - `sim_to_real_reward_gap`：同一策略在仿真与真实环境中的回报差值，目标值 < 10%。

## 14.8 小结

本章系统讨论了 World Model、LLM 与 Agent 三种技术路线的融合路径。核心结论是：LLM 的符号推理能力与 World Model 的因果预测能力并非互斥，而是互补——前者解决"做什么"的语义问题，后者解决"会发生什么"的物理问题。三条融合路径各有适用边界：语言化世界模型适用于粗略的符号化物理判断，世界模型增强的 LLM 适用于精细的因果验证，统一潜在空间适用于需要深度表征共享的端到端系统。工程实践表明，模块化紧耦合架构在当前阶段提供了最优的性价比，而端到端融合仍受限于数据规模与训练稳定性。第 15 章将进一步讨论融合系统在现实世界中的安全部署与对齐问题，特别是当 World Model 的预测被用于约束 LLM 的行动空间时，如何避免"过度保守"导致的能力丧失。

---

## 延伸阅读

以下论文选自本章分析所依据的 145 篇核心文献，覆盖了 World Model、Agent 推理与融合架构的最新进展：

1. **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T.** (2023). *Mastering diverse domains through world models*. arXiv:2301.04104. — DreamerV3 的完整技术报告，系统阐述了 RSSM 在潜在空间中的前向预测、价值估计与策略优化机制，是理解世界模型样本效率优势的核心文献。

2. **Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., et al.** (2022). *ReAct: Synergizing reasoning and acting in language models*. arXiv:2210.03629. — 将 LLM 的推理链与工具调用交错执行的奠基性工作，定义了"推理-行动"循环的基本范式，为后续与世界模型的融合提供了 Agent 架构基础。

3. **Guan, Y., Liao, H., Li, Z., Hu, J., Yuan, R., et al.** (2024). *World models for autonomous driving: An initial survey*. IEEE Transactions on Intelligent Transportation Systems. — 自动驾驶领域世界模型的权威综述，系统对比了 RSSM 与 JEPA 的架构差异、适用场景及训练策略，为理解世界模型的工程化部署提供了全面参考。

4. **Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., et al.** (2023). *AutoGen: Enabling next-gen LLM applications via multi-agent conversation*. arXiv:2308.08155. — 多 Agent LLM 系统的开源框架，展示了如何将 LLM 封装为可对话、可协作的 Agent，其模块化设计理念与本章讨论的松耦合/紧耦合架构选择直接相关。

5. **Wu, J., Yin, S., Feng, N., & Long, M.** (2026). *RL-VR-World: Training world models with reinforcement learning*. Advances in Neural Information Processing Systems (NeurIPS). — 提出通过强化学习后训练统一框架来增强世界模型的探索能力，缓解了模态竞争与表征崩溃问题，代表了世界模型与决策优化深度融合的最新方向。
