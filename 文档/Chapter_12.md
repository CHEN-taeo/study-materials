# 第12章 基于模型的强化学习（MBRL）——从 Dreamer 到 MuZero

**核心思想**：基于模型的强化学习（Model-Based Reinforcement Learning, MBRL）在智能体内部构建一个环境动力学的前馈预测模型，并完全在该模型生成的虚构轨迹中训练策略，从而将数据收集与策略优化解耦。Dreamer 通过循环状态空间模型（RSSM）在压缩的潜在空间中进行想象性推演，MuZero 则通过蒙特卡洛树搜索（MCTS）在学习的表征空间中规划——二者代表了 MBRL 的两大技术路线：连续控制的潜在想象与离散决策的表征搜索。

**历史脉络**：MBRL 的思想可追溯至1991年 Sutton 的 Dyna-Q 架构，该框架在格子世界中证明了一个简单的学习模型即可生成合成经验以加速 Q-learning。然而，Dyna-Q 的表格表示法在高维状态空间中失效，且早期尝试将 GP 回归直接用于原始像素（如 PILCO）面临 O(n³) 的计算瓶颈与维数灾难。2018至2020年间，两条突破路线几乎同时出现：Ha 与 Schmidhuber 的 World Models 以及 Hafner 等人的 PlaNet/Dreamer 将深度潜变量动力学引入控制领域；Schrittwieser 等人的 MuZero 则证明 MCTS 无需环境模拟器即可在围棋、国际象棋和雅达利游戏中实现超人类表现。这些工作的共同教训是：直接在像素空间建模物理世界代价过高，而在压缩的潜在空间中学习动力学才是可扩展的路径。第3章的表征学习为 RSSM 的潜在空间建模奠定了基础，第7章的循环神经网络为处理部分可观测序列提供了计算基础，第9章的无模型 RL（PPO、SAC）则为理解 MBRL 的策略优化目标提供了参照系。

---

## 12.1 从无模型 RL 到 MBRL：动机与演化

在标准无模型 RL（Model-Free RL）中，智能体从环境中收集转移样本 $(s_t, a_t, r_{t+1}, s_{t+1})$，并直接更新价值函数或策略参数。PPO、SAC、TD3 等算法在雅达利游戏和机器人控制中取得了显著成果，但存在一个结构性瓶颈：每单位策略改进需要消耗大量环境交互。样本效率的低下源于两个事实：其一，每个转移样本仅被使用一次或数次用于梯度更新；其二，策略网络与价值网络并未从数据中提取可重用的环境动力学知识。

MBRL 的替代方案是显式学习一个转移模型 $\hat{p}(s_{t+1}, r_{t+1} | s_t, a_t)$，并利用该模型进行规划或生成虚构轨迹。理论上，一旦模型被准确习得，智能体即可在"脑中"模拟数百万步交互，而无需真实的物理交互。这一思想的最早实现之一是 Dyna-Q [Sutton, 1991]，它将环境交互、模型学习与 Q-learning 更新编织在一个统一循环中。然而，Dyna-Q 的表格表示法无法泛化到未访问状态，在视觉输入或连续控制任务中完全失效。

2011年提出的 PILCO 采用高斯过程（Gaussian Process, GP）作为转移模型，并通过解析地传播不确定性来规划策略。PILCO 在低维控制任务（如摆杆、小车倒立摆）上展示了极高的样本效率（约数十次交互即可收敛），但其计算复杂度随数据量呈立方增长，且对高维观测（如图像）缺乏可行的表示方法。PILCO 的失败教训清晰地表明：连续状态空间中的 MBRL 必须依赖可扩展的神经网络表示，而非核方法。

2015年前后，深度 MBRL 的第一次浪潮尝试将神经网络直接拟合原始像素的转移函数。这些工作（如基于卷积网络的下一帧预测）在简单环境（如弹跳球、移动方块）中取得了有限成功，但很快暴露出一个核心问题：像素空间的预测是病态的。模型必须预测每一个像素的精确颜色变化，包括背景、光照、纹理细节，而这些信息对控制决策往往无关紧要。当模型将计算资源浪费在预测水面的反光波动时，机械臂的抓取策略并未因此受益。这种"预测目标与决策目标错位"的困境，迫使研究者重新思考：世界模型是否必须是一个生成像素 perfect 的物理模拟器？

2018年，Ha 与 Schmidhuber 的 World Models 提出了一种生成式潜在动力学方法：一个变分自编码器（VAE）将图像压缩为潜在向量，一个循环网络在潜在空间中预测未来，一个进化策略控制器在压缩的表征中行动。World Models 首次展示了在潜空间中"做梦"（dreaming）训练策略的可行性，但其控制器采用 CMA-ES 进化算法，优化效率低下，且 VAE 的潜在空间与任务奖励缺乏耦合。

2019年提出的 PlaNet 将世界建模带入了可微分端到端框架。PlaNet 使用循环状态空间模型（Recurrent State Space Model, RSSM）在潜在空间中预测未来，并通过模型预测控制（MPC）在动作空间中执行交叉熵方法（CEM）搜索。然而，PlaNet 的纯 MPC 规划在动作维度较高时计算开销巨大，且其策略优化并非基于梯度，限制了收敛速度。Dreamer 系列（2020–2023）保留了 RSSM 的潜在动力学框架，但将 MPC 替换为梯度优化的 actor-critic 策略，使策略网络能够直接在想象的轨迹上通过反向传播进行更新。这一改进将 Dreamer 的样本效率与渐进性能推向了无模型 RL 的同等水平，同时在多个连续控制基准上实现了数量级更少的环境交互。

几乎在 Dreamer 发展的同时，DeepMind 的 MuZero 在 2019 年末以预印本形式出现，并于 2020 年正式发表。MuZero 的出发点截然不同：它不试图学习一个可生成像素或物理状态的生成模型，而是学习一个抽象的表征空间，其中仅保留对决策有用的信息。在该表征空间中，MuZero 通过学习的动力学函数和 MCTS 进行前瞻搜索。MuZero 在围棋、国际象棋、将棋和雅达利游戏上均达到或超过了 AlphaZero 与无模型方法的性能，且无需任何游戏规则的先验知识。这一成果证明：规划与学习的结合可以完全不依赖显式环境模型，而是通过一个任务驱动的隐式动力学模型实现。

## 12.2 Dreamer 的 RSSM：循环状态空间模型

Dreamer 系列的核心计算图是循环状态空间模型（RSSM）。RSSM 将环境的部分可观测性（partial observability）与随机动力学（stochastic dynamics）统一在一个潜在变量框架中，其设计哲学是：环境的真实状态无法被完全观测，但可以通过一个压缩的潜在分布进行概率推断。

### 12.2.1 模型结构：确定性路径与随机路径的分离

RSSM 在每个时间步 $t$ 维护一个隐藏状态 $h_t$，该状态由两部分组成：一个确定性路径（deterministic path）和一个随机路径（stochastic path）。

**确定性路径**：由一个循环神经网络（通常为 GRU）编码历史信息：
$$h_t = f_\theta(\text{GRU}(h_{t-1}, z_{t-1}, a_{t-1}))$$

其中 $z_{t-1}$ 是上一时刻的随机潜在变量，$a_{t-1}$ 是上一时刻的动作。GRU 的引入保证了模型能够捕捉长期的时间依赖性，且确定性路径避免了信念状态分布随时间步传播时的方差爆炸问题。

**随机路径**：在确定性状态 $h_t$ 的基础上，模型推断一个随机潜在变量 $z_t$ 的后验分布。在训练时，$z_t$ 通过变分推断从观测 $o_t$ 中编码：
$$q(z_t | h_t, o_t) = \mathcal{N}(\mu_t, \text{diag}(\sigma_t^2))$$

其中 $\mu_t$ 和 $\sigma_t$ 由神经网络输出。在想象（imagination）或预测时，观测 $o_t$ 不可用，此时 $z_t$ 从先验分布中采样：
$$p(z_t | h_t) = \mathcal{N}(\hat{\mu}_t, \text{diag}(\hat{\sigma}_t^2))$$

这种后验/先验分离的设计是 RSSM 的关键创新：训练时通过观测校准信念，推理时通过先验进行自回归预测。

**观测模型**：RSSM 通过解码器从 $(h_t, z_t)$ 重建观测：
$$p(o_t | h_t, z_t)$$

在 DreamerV3 中，解码器通常使用转置卷积网络（图像域）或 MLP（低维状态空间），损失函数为像素级 MSE 或散度损失。值得注意的是，DreamerV3 引入了对称对数变换（symlog）处理图像像素与奖励值，以压缩极端值的影响。

**奖励模型与终止模型**：
$$\hat{r}_t = r_\theta(h_t, z_t), \quad \hat{c}_t = c_\theta(h_t, z_t)$$

其中 $c_t$ 是 episode 继续标志（continue flag），DreamerV3 用它替代传统的终止信号（done flag），使价值函数在边界处仍能传播梯度。

### 12.2.2 训练目标：KL 平衡与多任务损失

RSSM 的训练目标由四项组成：
$$\mathcal{L} = \underbrace{-\mathbb{E}_{q(z_{1:T})}[\log p(o_{1:T} | h_{1:T}, z_{1:T})]}_{\text{重构损失}} + \underbrace{\beta_{\text{kl}} \cdot \text{KL}[q(z_{1:T} | h_{1:T}, o_{1:T}) \| p(z_{1:T} | h_{1:T})]}_{\text{KL 正则}} - \underbrace{\mathbb{E}[\log p(r_{1:T})]}_{\text{奖励预测}} - \underbrace{\mathbb{E}[\log p(c_{1:T})]}_{\text{继续标志预测}}$$

DreamerV3 引入了一个关键工程技巧：KL 平衡（KL balancing）。标准 VAE 的 KL 散度在潜在维度较高时容易导致后验坍塌（posterior collapse），即 $q(z_t | h_t, o_t)$ 退化为与先验无异的分布。DreamerV3 将 KL 项拆分为两个不对称的权重：
$$\text{KL}_\text{loss} = \alpha \cdot \text{KL}[q \| p] + (1-\alpha) \cdot \text{KL}[p \| q]$$

其中 $\alpha$ 通常设为 0.8，强制后验分布向先验靠近的速度慢于先验向后验靠近，从而保留更多的观测信息。Guan 等人 [2024] 的综述指出，这一技巧是 RSSM 在 Dreamer 系列中稳定训练的核心机制之一。

### 12.2.3 从 PlaNet 到 DreamerV3 的演进

PlaNet（2019）使用 RSSM 进行模型预测控制：在动作空间中采样候选动作序列，通过 RSSM 展开预测累积奖励，以 CEM 优化动作序列。这一方法的缺陷在于计算量随规划 horizon 线性增长，且无法对策略参数进行梯度下降。

Dreamer（2020）将 RSSM 与 actor-critic 架构结合：在想象的潜在轨迹上训练策略网络和价值网络，使用 straight-through estimator 或重参数化技巧传播梯度。这一改变使 Dreamer 的样本效率大幅提升，但长期想象（超过 15 步）的梯度消失问题仍然突出。

DreamerV2（2021）引入了两项改进：离散潜在变量（通过 Gumbel-Softmax 实现）和 SYML 损失函数。离散潜在变量在部分可观测环境中表现更稳定，因为确定性分类分布比高斯分布更适合捕捉多模态信念状态。

DreamerV3（2023）则进一步简化了调参需求：去除了 DreamerV2 的多种超参数依赖，引入 symlog 和 symexp 变换统一处理奖励与图像的尺度问题，并采用固定的网络规模（RSSM 隐层 1024 维，潜在维度 32）在超过 150 个不同任务上实现零调参迁移。Hafner 等人 [2023] 的实验表明，DreamerV3 在 Atari、DeepMind Control Suite 和 Robodesk 上均以 1M 至 10M 环境步（远低于无模型方法的 50M–200M 步）达到或超过人类水平与无模型基线。Sarowar [2026] 在机器人学习框架的比较研究中确认，Dreamer 家族在真实机器人部署场景中代表了当前最成熟的样本高效学习方案之一。

## 12.3 MuZero：不依赖环境动力学的规划

MuZero 代表了一条与 Dreamer 截然不同的 MBRL 路线。如果说 Dreamer 致力于学习一个生成式的潜在世界模型以支持连续控制，那么 MuZero 的目标则是学习一个判别式的表征空间，以支持基于 MCTS 的离散决策。MuZero 的核心假设是：智能体不需要预测环境的真实未来状态（如像素或物理坐标），只需要预测那些对决策有用的抽象量——即价值、策略和即时奖励。

### 12.3.1 三大网络函数

MuZero 由三个可学习的神经网络组成：

**表征函数（Representation function）** $h_\theta$：将观测历史（通常为最近的 $k$ 帧图像）编码为根节点的内部状态 $s^0$：
$$s^0 = h_\theta(o_1, o_2, \ldots, o_t)$$

$h_\theta$ 在 Atari 中通常是一个 ResNet 或 ConvNet，在棋类游戏中则直接编码棋盘状态。关键在于，$s^0$ 是一个无显式语义的人工向量，其唯一的优化目标是为后续的预测与动力学提供足够信息。

**动力学函数（Dynamics function）** $g_\theta$：在当前内部状态 $s^k$ 和动作 $a^k$ 上预测下一状态 $s^{k+1}$ 和即时奖励 $r^{k+1}$：
$$s^{k+1}, r^{k+1} = g_\theta(s^k, a^k)$$

$g_\theta$ 通常是一个全连接网络。与 Dreamer 的 RSSM 不同，MuZero 的动力学不生成观测，而是生成一个纯粹抽象的下一个状态向量。这种"value-equivalent"建模方式极大地减轻了模型学习的负担：模型只需预测与价值估计相关的特征，无需承担像素重建的代价。

**预测函数（Prediction function）** $f_\theta$：从任意内部状态 $s^k$ 输出策略概率 $\pi^k$ 和价值估计 $v^k$：
$$\pi^k, v^k = f_\theta(s^k)$$

策略 $\pi^k$ 是通过 MCTS 搜索后验蒸馏得到的，价值 $v^k$ 则通过 bootstrap 的 n-step returns 训练。

### 12.3.2 MCTS 在学习的表征空间中展开

MuZero 的 MCTS 与 AlphaZero 的搜索过程高度相似，但每一步的转移不再由环境规则提供，而是由学习的动力学函数 $g_\theta$ 生成。在搜索树中，每条边 $(s, a)$ 存储三个统计量：先验概率 $P(s, a)$、访问次数 $N(s, a)$ 和动作价值 $Q(s, a)$。选择公式遵循改进的 UCB 准则：

$$U(s, a) = Q(s, a) + P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot \left(c_1 + \log\frac{\sum_b N(s, b) + c_2 + 1}{c_2}\right)$$

其中 $c_1$ 和 $c_2$ 控制探索强度。当搜索到达叶节点时，预测函数 $f_\theta$ 提供叶节点的价值与策略先验，随后该信息沿着搜索路径反向传播。具体而言，叶节点的价值被逐层回溯至根节点，更新路径上每个节点的 $Q$ 值与访问计数。MCTS 执行 $M$ 次模拟后，根节点的访问计数分布被转化为一个训练目标：

$$\pi_\text{target}(a) = \frac{N(s^0, a)^{1/\tau}}{\sum_b N(s^0, b)^{1/\tau}}$$

其中 $\tau$ 是温度参数。智能体在执行动作时，从该分布中采样（训练时保持探索，测试时选择访问次数最多的动作）。

### 12.3.3 无需规则：从数据中习得一切

MuZero 最引人注目的特性是它在围棋、国际象棋和将棋上的应用无需任何游戏规则的先验知识。表征函数、动力学函数和预测函数全部从自对弈或环境交互的轨迹中通过反向传播学习。在国际象棋中，MuZero 从未被告知"王车易位"或"兵的升变"规则，但它从胜利的棋局中习得了这些规则的内在等效动力学。在 Atari 游戏中，MuZero 同样无需知道碰撞检测、得分机制或游戏物理。

这一特性的代价是极高的计算需求。MuZero 的训练需要大规模分布式计算：在 Atari 上，每个实验通常使用 1000 个 TPU 进行数日的自对弈与网络训练。与 Dreamer 在单 GPU 上训练数小时即可收敛不同，MuZero 的样本效率（以环境交互计）并不突出，但其最终渐近性能（asymptotic performance）在无模型方法中居于领先地位。Sarowar [2026] 指出，MuZero 的真正贡献在于证明了树搜索与深度学习模型的结合可以泛化到任意环境，而不仅限于已知规则的完美信息博弈。

### 12.3.4 统一训练目标：将搜索、价值与动力学耦合

MuZero 的三个网络函数（表征、动力学、预测）共享一个统一的损失函数，该损失函数同时优化策略预测、价值估计与奖励预测：

$$\mathcal{L} = \sum_{k=0}^{K} \left( l^\pi(u_{t+k}, \pi^k) + l^v(z_{t+k}, v^k) + l^r(r_{t+k}, \hat{r}^k) \right)$$

其中 $u_{t+k}$ 是 MCTS 产生的目标策略（访问计数归一化），$z_{t+k}$ 是 n-step bootstrapped return（对于棋类游戏是最终胜负，对于 Atari 是折扣累积奖励），$r_{t+k}$ 是真实环境奖励。$l^\pi$ 通常采用交叉熵损失，$l^v$ 和 $l^r$ 采用 MSE 或 Huber 损失。展开步数 $K$ 通常设为 5，即模型同时学习从当前步起未来 5 步的 unrolled 动力学。这种 unrolled 训练是 MuZero 的关键：它不仅训练动力学函数在单步上准确，更要求该函数在 multi-step rollout 中保持与真实环境的 value-equivalence。Sarowar [2026] 强调，MuZero 的 model 不需要预测物理状态，只需要预测对决策有用的"价值等价"表征——这与 Dreamer 的生成式重建目标形成了根本性的分野。

### 12.3.5 MuZero 的变体：从棋盘到连续控制

MuZero 的原始设计针对离散动作空间。后续工作扩展了其在连续控制中的应用。MuZero Unplugged（2021）和 Sampled MuZero 引入了连续动作采样策略，通过在动作空间中采样候选动作并在 MCTS 叶节点中评估，将 MuZero 的框架推广到机器人控制。然而，连续动作空间中的 MCTS 面临维度灾难：动作维度 $d$ 的增大导致搜索树的组合爆炸。为此，TD-MPC（Temporal Difference Model Predictive Control）系列放弃了 MCTS，转而采用基于交叉熵优化的模型预测路径积分控制，在连续任务中取得了与 Dreamer 相当的样本效率。Sainburg 与 Weinreb [2026] 在分析 Agentic AI 的表征选择时指出，MuZero 与 Dreamer 分别代表了"决策耦合"与"动作解耦"的两种动力学学习范式，二者在潜在空间的建构目标上存在根本差异。

## 12.4 Dreamer vs MuZero：两条路线的对比

Dreamer 与 MuZero 并非简单的竞争关系，而是代表了 MBRL 在两种不同假设下的最优解。理解二者的 trade-off 是设计 MBRL 系统的关键。

| 维度 | Dreamer (RSSM) | MuZero (MCTS + Learned Model) |
|------|-----------------|-------------------------------|
| 动作空间 | 连续控制为主 | 离散决策为主 |
| 观测空间 | 高维图像、低维状态 | 高维图像、结构化状态 |
| 潜在空间 | 生成式：需重建观测 | 判别式：无需重建观测 |
| 策略优化 | Actor-Critic 梯度下降 | MCTS 搜索 + 策略蒸馏 |
| 梯度传播 | 通过想象轨迹反向传播 | 树搜索中无梯度，仅在网络训练时有梯度 |
| 计算瓶颈 | 想象 rollout 的内存 | MCTS 的模拟次数 |
| 样本效率 | 极高（10K–1M 步） | 中等（1M–10M 步） |
| 硬件需求 | 单 GPU 可行 | 大规模分布式 TPU 集群 |

**生成式 vs 判别式潜在模型**：Dreamer 的 RSSM 是一个生成模型，必须承担观测重建的责任。这一约束带来了双重效应：一方面，像素重建为潜在空间提供了丰富的自监督信号，使模型能够在无奖励信号的环境中也能学习有意义的表征；另一方面，像素重建消耗了大量模型容量，且重建目标可能与控制任务无关（例如，预测背景像素的变化对机器人抓取任务没有帮助）。MuZero 的表征函数完全摒弃了重建，其表征由价值与策略的预测目标直接塑造，因此模型容量全部集中于决策相关的特征。代价是 MuZero 无法在无奖励或探索环境中自举学习，因为它缺乏独立的观测预测目标。

**连续梯度 vs 离散搜索**：Dreamer 在想象的轨迹上执行端到端的梯度下降，策略网络和价值网络在潜在空间中直接优化。这一连续优化框架天然适合高维连续动作空间（如机器人关节扭矩），但长期想象中的梯度截断（DreamerV3 通常使用 $\lambda$-return 截断至 15 步）限制了模型对远期后果的推理能力。MuZero 的 MCTS 通过树搜索显式枚举动作序列的前后果，能够捕捉到局部梯度优化难以发现的远期策略依赖（如围棋中的弃子战术）。然而，MCTS 的动作枚举在动作空间庞大时失效，这也是 MuZero 难以直接应用于多关节机器人控制的核心原因。

**数据效率与计算效率的权衡**：Dreamer 在样本效率上具有压倒性优势。在 DeepMind Control Suite 的 cartpole-swingup 任务上，DreamerV3 通常仅需约 4 万次环境交互即可达到满分，而 MuZero 的 Atari 训练需要数亿帧。然而，Dreamer 的每次策略更新需要展开数百步的想象轨迹并在潜在空间中执行反向传播，其单步计算开销较高。MuZero 的每次环境交互需要执行数百次 MCTS 模拟，但网络训练本身相对轻量。因此，若环境交互成本高昂（如真实机器人），Dreamer 是更优选择；若环境交互廉价且计算资源丰富（如模拟器集群），MuZero 的渐进性能更有吸引力。

**模型偏差的敏感性**：Dreamer 的策略在想象轨迹上训练，若在真实环境中遇到模型未见过的情况，模型偏差（model bias）会导致策略失效。RSSM 通过随机潜在变量和确定性路径的分离在一定程度上缓解了这一问题，但长期 rollout（超过 50 步）的复合误差仍然显著。MuZero 的模型偏差表现为 MCTS 中的错误转移预测，但由于搜索树的深度有限（通常为 5–10 步），且每次动作选择都重新进行搜索，MuZero 对模型误差的敏感度相对较低。Huang 等人 [2024] 提出的 SafeDreamer 在 RSSM 框架中引入拉格朗日约束，显式地将安全成本纳入世界模型的优化目标，通过限制想象轨迹中的风险状态来降低模型偏差在真实部署中的危害。

**可解释性与调试的对比**：MuZero 的 MCTS 搜索树天然提供了可解释性。研究者可以可视化根节点下各动作的分支价值、访问频率与置信区间，从而理解模型"为何选择这一步"。在围棋中，MuZero 的搜索树能够揭示它对人类定式的习得程度；在 Atari 中，搜索树的价值估计变化能够暴露模型对关键状态转移（如即将被敌人击中）的敏感度。相比之下，Dreamer 的 RSSM 是一个潜在空间中的黑箱动力学系统。虽然可以可视化重建的观测图像，但潜在变量 $z_t$ 的语义是不可解释的。调试一个表现不佳的 Dreamer 策略通常需要检查重建损失、KL 散度与奖励预测的曲线，而无法直接"阅读"其推理过程。这一差异在实际工程部署中影响深远：在安全关键领域（如自动驾驶），MuZero 的搜索可解释性更受青睐；在需要快速迭代的机器人原型开发中，Dreamer 的端到端梯度优化更便捷。Guan 等人 [2024] 的综述指出，自动驾驶领域的世界模型研究正在尝试融合两种路线的优点：使用 Dreamer 风格的 RSSM 进行高效的连续控制学习，同时引入 MCTS 的决策树分析以满足安全审计需求。

## 12.5 想象性推演：在潜在空间中训练策略

Dreamer 系列的核心训练机制是想象性推演（imaginative rollout）。与无模型 RL 直接从回放缓冲区中采样真实转移不同，Dreamer 从世界模型中生成虚构的潜在轨迹，并在这类"梦境"中训练 actor-critic。

### 12.5.1 想象轨迹的生成过程

训练时，Dreamer 从回放缓冲区中随机抽取一个初始状态 $h_\tau$（通过编码真实观测序列得到）。随后，策略网络 $\pi_\theta(a | h, z)$ 在潜在空间中自回归地采样动作，RSSM 根据动作生成下一潜在状态：

$$h_{t+1}, z_{t+1} = \text{RSSM}(h_t, z_t, a_t), \quad a_t \sim \pi_\theta(a | h_t, z_t)$$

该过程展开 $H$ 步（通常 $H=15$），生成一条纯想象的轨迹 $(h_\tau, z_\tau, a_\tau, \hat{r}_\tau, h_{\tau+1}, z_{\tau+1}, \ldots)$。策略网络和价值网络仅在这条想象轨迹上计算梯度。真实环境的数据仅用于训练 RSSM 本身，不直接用于策略更新。这种解耦意味着：即使环境交互速度很慢（如真实机器人），策略仍可在高速想象的"心智模拟"中快速迭代。

### 12.5.2 $\lambda$-Return 与价值目标

DreamerV3 采用 TD-$\lambda$ 目标估计价值函数的 bootstrap 目标：

$$V_\lambda^{(t)} = \sum_{n=1}^{H-t} (1-\lambda)\lambda^{n-1} \hat{G}_t^{(n)} + \lambda^{H-t} \hat{G}_t^{(H)}$$

其中 $n$-step return 为：
$$\hat{G}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \hat{r}_{t+k} + \gamma^n \hat{v}_\psi(h_{t+n}, z_{t+n})$$

$\lambda$ 控制 bootstrap 的混合程度，DreamerV3 通常使用 $\lambda=0.95$。价值网络的损失函数采用 symlog 变换后的 Huber 损失：
$$\mathcal{L}_V = \mathbb{E}[\text{symexp}(\text{symlog}(V_\lambda) - \text{symlog}(\hat{v}_\psi(h, z)))]$$

symlog 变换定义为 $\text{symlog}(x) = \text{sign}(x) \cdot \log(|x|+1)$，其作用是压缩极端奖励值（如 Atari 游戏中一次性获得的大量分数）对梯度更新的冲击，避免价值网络在稀疏大奖励面前崩溃。

### 12.5.3 策略目标：最大化预期回报与熵正则化

策略网络的目标是在想象轨迹上最大化预期回报，同时维持适度的动作熵以鼓励探索：
$$\mathcal{L}_\pi = -\mathbb{E}_{\text{imagination}}\left[\sum_{t=0}^{H-1} \gamma^t \hat{r}_t + \gamma^H \hat{v}_\psi(h_H, z_H) \right] + \eta \cdot \mathbb{H}[\pi(a | h, z)]$$

在 DreamerV3 中，策略梯度通过重参数化技巧（若动作连续）或 straight-through Gumbel-Softmax（若动作离散）传播。值得注意的是，DreamerV3 对策略网络的优化采用了动态熵调节：若最近几个 episode 的回报方差过低，则自动增加熵系数 $\eta$；若回报方差过高，则降低 $\eta$。这一机制使 DreamerV3 在 150 余个不同任务上无需手动调节探索参数。

### 12.5.4 想象与现实的校准：模型误差如何影响策略

想象性推演的致命弱点是模型误差（model error）的复合。当 RSSM 在想象轨迹中偏离真实动力学时，策略会基于错误的世界假设进行优化，导致在真实环境中表现骤降。这一现象被称为"模型偏差导致的策略退化"（model-bias-induced policy degradation）。DreamerV3 通过以下机制缓解该问题：

1. **短 horizon rollout**：限制想象长度 $H \leq 15$，避免长期误差累积。虽然 15 步的想象不足以捕捉某些长期依赖（如围棋中的定式），但在连续控制中，15 步通常覆盖了足够多的短期动力学信息以支撑策略优化。
2. **真实数据的比例约束**：每次策略更新混合使用真实转移与想象转移，确保策略不脱离现实。DreamerV3 的经验比例约为 1:1，即策略网络在真实轨迹与想象轨迹上各训练一次。
3. **继续标志预测**：通过预测 episode 何时终止，价值函数在想象边界处自动回退到 bootstrap，避免在不可预测的远处硬编码回报。
4. **潜在空间中的不确定性量化**：虽然 DreamerV3 的标准实现不维护显式的不确定性估计，但随机潜在变量 $z_t$ 的方差可间接反映模型对当前状态的置信度。在改进的 Dreamer 变体中，研究者通过集成多个 RSSM 实例或引入贝叶斯神经网络层，显式估计模型不确定性，并在不确定性高的区域缩短想象 horizon。

Wu 等人 [2026] 提出的 RLVR-World 进一步探索了通过强化学习后训练世界模型的可能性。他们的研究表明，在预训练的世界模型之上，通过 RL 对模型的预测行为进行精调，可以显著改善长 horizon 想象的质量。这一工作暗示了未来的 MBRL 系统可能采用两阶段训练：首先通过观测重建自监督学习世界模型，然后通过环境交互的稀疏奖励对模型进行策略对齐的精调。

## 12.6 代码：简化版 RSSM 与想象性 Rollout

以下代码提供了一个可在 PyTorch 中运行的简化版 RSSM 实现，包含编码器、RSSM 核心、解码器、想象性 rollout 和策略价值更新。为清晰起见，代码采用连续动作空间与图像观测，省略了分布式训练与多环境并行。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class Encoder(nn.Module):
    """将图像观测编码为特征向量。"""
    def __init__(self, obs_shape=(3, 64, 64), embed_dim=1024):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 4, 2), nn.ELU(),   # 32x31x31
            nn.Conv2d(32, 64, 4, 2), nn.ELU(),  # 64x14x14
            nn.Conv2d(64, 128, 4, 2), nn.ELU(), # 128x6x6
            nn.Conv2d(128, 256, 4, 2), nn.ELU() # 256x2x2
        )
        self.fc = nn.Linear(256 * 2 * 2, embed_dim)
    
    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(obs.size(0), -1)
        return self.fc(x)

class RSSM(nn.Module):
    """简化版循环状态空间模型。"""
    def __init__(self, stoch_size=32, deter_size=256, hidden_size=256, action_size=6):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_size = action_size
        
        # 确定性路径：GRU
        self.gru = nn.GRUCell(hidden_size, deter_size)
        
        # 从 (h, z, a) 到 GRU 输入
        self.fc_input = nn.Linear(deter_size + stoch_size + action_size, hidden_size)
        
        # 先验：p(z | h)
        self.fc_prior = nn.Linear(deter_size, 2 * stoch_size)
        
        # 后验：q(z | h, o_embed)
        self.fc_posterior = nn.Linear(deter_size + 1024, 2 * stoch_size)
    
    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.deter_size, device=device)
        z = torch.zeros(batch_size, self.stoch_size, device=device)
        return h, z
    
    def observe(self, h_prev, z_prev, a_prev, o_embed):
        """训练时：利用观测编码 o_embed 推断后验。"""
        x = torch.cat([h_prev, z_prev, a_prev], dim=-1)
        x = self.fc_input(x)
        h = self.gru(x, h_prev)
        
        prior_mean, prior_std_log = self.fc_prior(h).chunk(2, dim=-1)
        prior_std = F.softplus(prior_std_log) + 0.1
        
        post_input = torch.cat([h, o_embed], dim=-1)
        post_mean, post_std_log = self.fc_posterior(post_input).chunk(2, dim=-1)
        post_std = F.softplus(post_std_log) + 0.1
        
        z = post_mean + post_std * torch.randn_like(post_std)
        return h, z, post_mean, post_std, prior_mean, prior_std
    
    def imagine(self, h_prev, z_prev, a_prev):
        """想象时：无观测，从先验采样。"""
        x = torch.cat([h_prev, z_prev, a_prev], dim=-1)
        x = self.fc_input(x)
        h = self.gru(x, h_prev)
        
        prior_mean, prior_std_log = self.fc_prior(h).chunk(2, dim=-1)
        prior_std = F.softplus(prior_std_log) + 0.1
        z = prior_mean + prior_std * torch.randn_like(prior_std)
        return h, z, prior_mean, prior_std

class Decoder(nn.Module):
    """从 (h, z) 重建图像。"""
    def __init__(self, stoch_size=32, deter_size=256, obs_shape=(3, 64, 64)):
        super().__init__()
        self.fc = nn.Linear(stoch_size + deter_size, 256 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2), 
        )
        self._obs_shape = obs_shape
    
    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        x = self.fc(x).reshape(-1, 256, 2, 2)
        x = self.deconv(x)
        # 裁剪至目标尺寸
        if x.shape[-2:] != self._obs_shape[-2:]:
            x = F.interpolate(x, size=self._obs_shape[-2:], mode='bilinear', align_corners=False)
        return x

class RewardModel(nn.Module):
    def __init__(self, stoch_size=32, deter_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_size + deter_size, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))

class ActorCritic(nn.Module):
    """策略与价值网络。"""
    def __init__(self, stoch_size=32, deter_size=256, action_size=6):
        super().__init__()
        state_dim = stoch_size + deter_size
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 2 * action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1)
        )
        self.action_size = action_size
    
    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        actor_out = self.actor(state)
        mean, std_log = actor_out.chunk(2, dim=-1)
        std = F.softplus(std_log) + 0.1
        value = self.critic(state)
        return mean, std, value
    
    def sample_action(self, h, z):
        mean, std, _ = self.forward(h, z)
        dist = Independent(Normal(mean, std), 1)
        return dist.rsample()

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def compute_lambda_returns(rewards, values, bootstrap, gamma=0.99, lambda_=0.95):
    """计算 TD-lambda returns。"""
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    last = bootstrap
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * last - values[t]
        returns[t] = delta + gamma * lambda_ * last
        last = returns[t] + values[t]
    return returns

def rssm_imagination_rollout(rssm, actor, h0, z0, horizon=15):
    """在潜在空间中展开想象轨迹。"""
    h, z = h0, z0
    imagined_states = []
    for _ in range(horizon):
        mean, std, _ = actor(h, z)
        dist = Independent(Normal(mean, std), 1)
        a = dist.rsample()
        h, z, _, _ = rssm.imagine(h, z, a)
        imagined_states.append((h, z, a))
    return imagined_states

# === 训练循环示意 ===
def train_step(obs_seq, action_seq, rssm, encoder, decoder, reward_model, actor_critic):
    """
    obs_seq: [T, B, C, H, W] 真实观测序列
    action_seq: [T-1, B, action_dim] 动作序列
    """
    T, B = obs_seq.shape[:2]
    device = obs_seq.device
    
    # 编码观测
    o_embed = encoder(obs_seq.reshape(T * B, *obs_seq.shape[2:])).reshape(T, B, -1)
    
    # 初始化状态
    h, z = rssm.initial_state(B, device)
    
    # 收集 RSSM 输出
    hs, zs, post_ms, post_ss, prior_ms, prior_ss = [], [], [], [], [], []
    for t in range(T):
        if t == 0:
            # 首步无动作，用零动作占位
            a = torch.zeros(B, rssm.action_size, device=device)
        else:
            a = action_seq[t-1]
        h, z, post_m, post_s, prior_m, prior_s = rssm.observe(h, z, a, o_embed[t])
        hs.append(h); zs.append(z)
        post_ms.append(post_m); post_ss.append(post_s)
        prior_ms.append(prior_m); prior_ss.append(prior_s)
    
    hs = torch.stack(hs)  # [T, B, deter_size]
    zs = torch.stack(zs)  # [T, B, stoch_size]
    post_ms = torch.stack(post_ms)
    post_ss = torch.stack(post_ss)
    prior_ms = torch.stack(prior_ms)
    prior_ss = torch.stack(prior_ss)
    
    # 重建损失
    obs_pred = decoder(hs.reshape(T*B, -1), zs.reshape(T*B, -1)).reshape(T, B, *obs_seq.shape[2:])
    recon_loss = F.mse_loss(obs_pred, obs_seq)
    
    # 奖励预测损失
    rewards_pred = reward_model(hs.reshape(T*B, -1), zs.reshape(T*B, -1)).reshape(T, B)
    # 假设 rewards_true 已准备
    # reward_loss = F.mse_loss(rewards_pred, rewards_true)
    
    # KL 平衡（简化版：标准 KL）
    kl_loss = torch.mean(
        Normal(post_ms, post_ss).log_prob(zs) - Normal(prior_ms, prior_ss).log_prob(zs)
    )
    
    # 世界模型总损失
    world_model_loss = recon_loss + 0.1 * kl_loss  # beta_kl = 0.1
    
    # 想象性训练策略
    # 从最后一个真实状态开始想象
    h_im, z_im = hs[-1], zs[-1]
    imagined = rssm_imagination_rollout(rssm, actor_critic, h_im, z_im, horizon=15)
    
    # 提取想象的奖励与价值
    im_rewards, im_values = [], []
    for h_t, z_t, a_t in imagined:
        r_t = reward_model(h_t, z_t)
        _, _, v_t = actor_critic(h_t, z_t)
        im_rewards.append(r_t)
        im_values.append(v_t)
    
    im_rewards = torch.stack(im_rewards).squeeze(-1)  # [H, B]
    im_values = torch.stack(im_values).squeeze(-1)      # [H, B]
    bootstrap = im_values[-1]
    
    # Lambda returns
    returns = compute_lambda_returns(im_rewards, im_values, bootstrap)
    
    # 价值损失
    value_loss = F.mse_loss(im_values, returns.detach())
    
    # 策略损失：最大化 returns
    policy_loss = -returns.mean()
    
    # 注意：实际 DreamerV3 使用 symlog 与 symexp 变换，以及动态熵调节，此处为简化示意
    return world_model_loss, value_loss, policy_loss
```

代码中省略了数据加载、回放缓冲区管理、熵正则化与目标网络等工程细节，但保留了 RSSM 的核心结构：确定性 GRU 路径、随机高斯后验/先验分离、观测重建、想象性 rollout 以及 actor-critic 的潜在空间训练。实际部署时，应将 `horizon` 设为 15，`stoch_size` 设为 32，`deter_size` 设为 1024（与 DreamerV3 的默认配置一致），并引入 symlog 变换处理奖励目标。

## 12.7 工程实践框：训练 MBRL 系统的常见陷阱

**模型偏差与幻觉状态**：世界模型在训练分布之外的区域会产生荒谬的预测。一个典型场景是：在机械臂控制中，若训练时物体从未掉落到桌面边缘，模型在想象时可能预测物体会穿过桌面继续下落。策略基于这种幻觉优化，会在真实环境中失败。缓解策略包括：在想象 rollout 中注入真实数据（通常保持真实:想象比例为 1:1），以及使用模型集成（ensemble）估计不确定性。当集成成员分歧过大时，丢弃该想象轨迹并回退到真实数据。

**KL 平衡的权重选择**：RSSM 的 KL 项权重 $\beta_{\text{kl}}$ 若过大，后验 $q(z|h,o)$ 会退化为先验 $p(z|h)$，观测信息丢失，重建模糊；若过小，模型在训练时过度依赖观测，先验无法学习有效的预测分布，导致想象 rollout 发散。DreamerV3 的默认配置是 `kl_free = 1.0`（若 KL 小于 1.0 则不计入损失）与 `kl_balance = 0.8`，但针对不同视觉复杂度（如静态背景 vs 动态摄像机），该参数可能需要 ±20% 的微调。

**奖励尺度与 symlog 的必要性**：在 Atari 等游戏中，单次奖励可达数十至数百，而多数时间步奖励为零。若直接使用 MSE 训练奖励模型，零奖励步的梯度会被极端值淹没。DreamerV3 的 symlog 变换将 $[-1000, 1000]$ 的奖励压缩到 $[-7, 7]$ 的范围内，使价值网络稳定学习。若忽略该变换，在 Breakout 或 Pong 等游戏中价值网络通常在 100K 步内发散至 NaN。

**长期想象的梯度截断**：RSSM 的想象 rollout 在 PyTorch 中构成一个计算图。若 horizon=50 且 batch_size=16，该图会消耗超过 20GB 的 GPU 显存。解决方案是在想象过程中对 RSSM 的隐藏状态使用 `.detach()` 阻断梯度，仅保留最后几步用于策略梯度。DreamerV3 的实际实现使用动态梯度截断：在每一想象步中，以 0.5 的概率截断梯度传播，从而在内存与信用分配之间取得平衡。

**离散动作空间的连续化**：若将 Dreamer 应用于离散动作（如 Atari），需将策略输出从连续高斯分布替换为 Gumbel-Softmax 或直接使用分类分布。但 DreamerV3 的原始设计针对连续控制，其价值网络在离散动作空间中的 bootstrap 稳定性较差。此时，MuZero 的 MCTS 框架更为合适。工程上的一条经验法则是：当动作维度 $< 10$ 且动作离散时，优先尝试 MuZero；当动作维度 $> 10$ 或动作连续时，优先尝试 Dreamer。

**硬件与训练时间估算**：在单张 NVIDIA A100 上训练 DreamerV3 处理 DMControl 的连续控制任务（图像输入，64x64），batch_size=16，约需 4–8 小时达到收敛（以 100 万环境步计）。在 Atari 上（图像输入，64x64，离散动作），收敛需 24–48 小时。MuZero 的 Atari 训练在 1000 TPU v3 上需约 5–7 天。若仅有单 GPU 资源，MuZero 的复现通常不可行；此时应选择 DreamerV3 或轻量化的 EfficientZero（MuZero 的采样优化版本，可在 4 GPU 上训练）。

**回放缓冲区的遗忘问题**：由于世界模型在训练早期对动力学的估计不准确，早期收集的转移样本可能包含错误的模型预测。若这些样本长期保留在回放缓冲区中，会持续污染模型训练。建议在 Dreamer 中采用优先经验回放（PER），但优先级应基于模型预测误差而非 TD 误差：预测误差大的样本优先被重放，使模型快速学习那些尚未被准确建模的动力学区域。Wang 等人 [2026] 的 WorldCompass 工作进一步指出，对长 horizon 世界模型进行 RL 后训练时，回放缓冲区的样本分布应随策略迭代动态更新，以避免旧策略产生的过时的转移样本干扰模型。

## 12.8 小结

基于模型的强化学习通过构建环境动力学的内部预测模型，将策略学习从昂贵的物理交互中解放出来。Dreamer 系列通过 RSSM 在压缩的潜在空间中执行想象性推演，以 actor-critic 的梯度优化实现连续控制的高效学习；MuZero 通过学习的表征函数与动力学函数，在抽象的决策空间内执行 MCTS 搜索，以无需规则先验的方式在离散决策中达到超人类水平。二者代表了 MBRL 的两大技术范式：生成式潜在想象与判别式表征规划。前者以样本效率见长，后者以渐进性能与可解释性（树搜索的可视化）见长。它们的共同基础是第3章讨论的表征学习——无论是 RSSM 的随机潜在变量还是 MuZero 的根节点表征，其本质都是从高维观测中提取对决策有用的压缩表示。MBRL 的当前挑战在于如何融合这两条路线：将 Dreamer 的潜在动力学与 MuZero 的搜索能力结合，在统一框架中支持连续动作与长期规划。此外，如何量化并约束世界模型的不确定性、如何在真实机器人中实现安全部署（如 SafeDreamer 的安全约束），以及如何利用大规模无标签视频数据预训练世界模型（如 RLVR-World 的 RL 后训练范式），构成了该领域未来三至五年的核心研究方向。

---

**延伸阅读**

1. Hafner, Pasukonis, Ba, & Lillicrap, 2023. *Mastering diverse domains through world models*. arXiv:2301.04104.  DreamerV3 的原始论文，展示了 RSSM 在超过 150 个不同任务上的零调参迁移能力，是理解现代潜在空间 MBRL 的必读文献。

2. Guan, Liao, Li, Hu, Yuan, et al., 2024. *World models for autonomous driving: An initial survey*. IEEE Transactions on Intelligent Vehicles.  系统综述了 RSSM 与 JEPA 在自动驾驶世界建模中的应用，提供了 RSSM 架构细节与 Dreamer 系列在真实驾驶场景中的技术演进脉络。

3. Huang, Ji, Xia, Zhang, et al., 2024. *SafeDreamer: Safe reinforcement learning with world models*. ICLR 2024.  在 Dreamer 框架中引入拉格朗日安全约束，将安全成本显式纳入世界模型的优化目标，是 MBRL 安全部署的关键工程参考。

4. Wu, Yin, Feng, & Long, 2026. *RLVR-World: Training world models with reinforcement learning*. NeurIPS 2025.  提出通过强化学习后训练世界模型的统一框架，展示了在多种模态上通过 RL 精调世界模型的可行性与性能增益。

5. Wang, Wang, Zhang, Zuo, Wu, et al., 2026. *WorldCompass: Reinforcement learning for long-horizon world models*. arXiv:2602.09022.  针对长 horizon 交互视频世界模型的 RL 后训练方法，探讨了如何通过动态回放缓冲区策略改善长程想象 rollouts 的稳定性与探索效率。
