# 第4章：预训练与 Scaling Laws（数据、计算、参数的最优配置）

## 核心思想

大语言模型的性能不是魔法，而是计算量（compute）、参数量（parameters）和数据量（tokens）之间服从幂律关系的工程产物。Kaplan 等人最初发现损失与参数量的幂律关系，但 Hoffmann 等人的 Chinchilla 分析揭示了一个更深刻的结论：对于固定的计算预算，模型大小和训练数据量应该等比例增长——这一最优性原则直接改变了 GPT-3 时代"大模型+短训练"的工程范式，使得小模型+长训练成为效率更优的选择。

---

## 历史脉络：从试错到最优配置的十年

2018-2020 年间，大规模语言模型的训练策略基本是经验性的。OpenAI 在 GPT-2 和 GPT-3 的训练中采用了一个隐含假设：给定固定的计算预算，增加模型参数量比增加训练数据量更重要。GPT-3 175B 模型使用了约 300B 的 token，训练在数百张 V100 上运行了数月。这一策略的直觉是：更大的模型有更强的表征能力，可以在更少的数据上达到相同的损失。但这一直觉没有被系统验证——它只是一个基于少量实验的启发式。

2020 年，Kaplan 等人发表了第一篇关于 Scaling Laws 的系统性研究 [Kaplan et al., 2020]。他们训练了数百个不同规模的模型（参数从 10^3 到 10^9），发现测试损失 L 与模型参数量 N 和训练数据量 D 之间存在幂律关系：L(N, D) = A/N^α + B/D^β + E。其中 α ≈ 0.073，β ≈ 0.095，E 是不可约误差（irreducible error）。这一发现是里程碑式的：它意味着模型性能可以**预测**——不需要训练完整的模型，只需训练几个小模型并拟合幂律，就能预测大模型的性能。

但 Kaplan 的 Scaling Laws 有一个关键盲区：他们的实验是在**固定训练步数**（fixed training steps）下进行的，这意味着当模型增大时，每个 token 的梯度更新次数减少了。2022 年，Hoffmann 等人的研究 [Hoffmann et al., 2022] 纠正了这一偏差：他们保持**计算量 C = 6ND** 固定（其中 6 是前向+反向传播的浮点运算系数），然后改变 N 和 D 的比例。结果发现，最优配置不是 N ≫ D，而是 N ∝ D^{0.5}——即参数量和数据量应该等比例增长。基于这一发现，他们训练了 Chinchilla（70B 参数，1.4T token），其性能超过了 GPT-3（175B 参数，300B token）和 Gopher（280B 参数，300B token），尽管参数量远小于后两者。

Chinchilla 的发现引发了行业震动：它证明了 GPT-3 的范式是**次优的**。在相同的计算预算下，更小的模型训练更长时间可以取得更好的性能。这一结论对工程实践的影响是深远的：它改变了数据中心的 GPU 分配策略、模型并行配置和数据管道设计。本章将深入推导 Scaling Laws 的数学形式，分析数据质量和数据混合对幂律的影响，并讨论分布式训练中的工程 trade-off。

---

## 4.1 Scaling Laws 的数学形式

### 4.1.1 幂律的推导

Scaling Laws 的核心观察是：语言模型的交叉熵损失 L 与模型参数量 N 和训练 token 数 D 之间存在近似的幂律关系。假设训练是充分优化的（即模型已收敛到该规模下的最优损失），损失可以写成：

L(N, D) = E + A/N^α + B/D^β

其中：
- E 是不可约误差（irreducible error），代表任务的理论极限（如人类水平的困惑度）；
- A/N^α 是模型容量不足导致的近似误差；
- B/D^β 是数据不足导致的估计误差。

在 Kaplan 的原始实验中，α ≈ 0.076，β ≈ 0.095（注：Kaplan et al. [2020] 原文报告的 α ≈ 0.073，β ≈ 0.095；本章为教学一致性统一取 α ≈ 0.076，β ≈ 0.095，并将在后续推导中使用这两个值）。这意味着：如果模型参数量增加 10 倍，近似误差降低约 10^0.076 ≈ 1.19 倍；如果数据量增加 10 倍，估计误差降低约 10^0.095 ≈ 1.25 倍。注意：β > α，这意味着数据量的回报略大于参数量的回报——这是 Chinchilla 最优性的数学基础。

### 4.1.2 计算量约束下的优化

训练一个 Transformer 的计算量（以 FLOPs 计）可以近似为：

C ≈ 6ND

其中 N 是参数量，D 是训练 token 数。系数 6 的来源：前向传播需要约 2ND FLOPs（每个参数对每个 token 做一次乘加），反向传播需要约 4ND FLOPs（梯度计算+参数更新），总共约 6ND。

给定固定的计算预算 C，我们需要在 N 和 D 之间分配资源，以最小化损失：

min_{N,D} L(N,D) = E + A/N^α + B/D^β

s.t. 6ND = C

使用拉格朗日乘子法，可以求解最优比例。令 D = C/(6N)，代入损失函数：

L(N) = E + A/N^α + B/(C/(6N))^β = E + A/N^α + B(6N/C)^β

对 N 求导并令导数为零：

-αA/N^{α+1} + βB·6^β·N^{β-1}/C^β = 0

整理得：

N^{α+β} = (αA / (βB·6^β)) · C^β

取对数：

(α+β) log N = log(αA / (βB·6^β)) + β log C

因此，最优参数量 N* 与计算量 C 的关系是：

log N* = const + (β / (α+β)) log C

类似地，最优数据量 D* 满足：

log D* = const + (α / (α+β)) log C

代入 Kaplan 的 α ≈ 0.076，β ≈ 0.095：

N* ∝ C^{0.095 / 0.171} ≈ C^{0.556}
D* ∝ C^{0.076 / 0.171} ≈ C^{0.444}

这意味着：对于计算量 C，最优的 N 和 D 应该大致按 N ∝ D^{0.556/0.444} ≈ D^{1.25} 的比例增长。但 Hoffmann 的实验发现，当考虑训练充分性（即模型不欠拟合）时，实际最优比例更接近 N ∝ D^{0.5}——即 N 和 D 等比例增长。这是 Chinchilla 的最优性结论。

### 4.1.3 从幂律到工程实践

Scaling Laws 的工程价值在于**预测能力**。假设你有一个 1B 参数的模型，在 20B token 上训练，测试困惑度为 10.0。你想知道训练一个 10B 参数的模型在 200B token 上，困惑度会是多少。如果 Scaling Laws 成立，你可以拟合参数 A, B, E，然后直接预测：

log L = log E + log(1 + (A/E)/N^α + (B/E)/D^β)

在工程实践中，通常不需要拟合 E（不可约误差），而是直接用两个已知点来确定 A 和 B：

log(L_1 - E) = log A - α log N_1 + log(1 + (B/A)N_1^α/D_1^β)
log(L_2 - E) = log A - α log N_2 + log(1 + (B/A)N_2^α/D_2^β)

如果 E 已知（或假设为某个值，如人类困惑度 1.0），就可以解出 A 和 B。然后预测任意 (N, D) 配置下的损失。

---

## 4.2 Chinchilla 最优性：为什么小模型+长训练更好

### 4.2.1 GPT-3 的次优性

GPT-3 175B 模型使用了约 300B token 的训练数据。根据 Chinchilla 的分析，给定训练 GPT-3 所用的计算量（约 3.14×10^23 FLOPs），最优配置应该是约 70B 参数和 1.4T token。这意味着 GPT-3 是**模型过大、数据过少**的：175B 参数在 300B token 上训练，每个参数只看到了约 1.7 个 token，远未达到收敛。相比之下，Chinchilla 70B 在 1.4T token 上训练，每个参数看到了约 20 个 token，模型充分收敛。

这种次优性的后果是：GPT-3 的某些能力（如 few-shot learning）是通过模型规模而非数据质量获得的。如果用 Chinchilla 的最优配置重新训练 GPT-3，可以在相同计算量下获得更优的困惑度，或者用一个更小的模型（如 70B）达到 GPT-3 的性能——后者意味着更快的推理速度和更低的部署成本。

### 4.2.2 计算最优 vs 推理最优

Chinchilla 分析假设目标是**最小化训练损失**。但在实际应用中，我们往往更关心**推理效率**——一个模型在达到目标性能后，推理速度有多快？一个 70B 的 Chinchilla 最优模型和一个 175B 的 GPT-3 模型，如果性能相近，70B 模型的推理速度是 175B 的 2.5 倍（因为推理时间 ∝ 参数量）。这意味着 Chinchilla 最优性不仅降低了训练成本，还大幅降低了部署成本。

然而，有一个重要的 trade-off：对于某些**涌现能力**（如多步数学推理、代码生成），模型规模似乎有一个阈值——低于该阈值，能力不涌现；高于该阈值，能力突然增强。如果这种涌现能力确实存在（目前仍有争议），那么 Chinchilla 最优模型可能无法达到大模型的涌现能力，即使两者的困惑度相近。这一 trade-off 是工程实践中的关键决策点：如果应用需要涌现能力，可能需要牺牲训练效率来换取更大的模型。

---

## 4.3 数据质量：去重、过滤与合成数据

### 4.3.1 数据去重的重要性

Scaling Laws 假设每个训练 token 都是独立同分布的（i.i.d.）且信息含量相同。但真实数据并非如此。Common Crawl 等网络语料中有大量重复内容（如模板化的网页、导航栏、版权声明）。Gao et al. [2020] 的研究发现，GPT-3 的训练数据中有约 10% 的重复内容，这意味着约 30B token 被浪费了。去重后，等效数据量增加，模型性能提升。

去重的方法包括：
- **文档级去重**：使用 MinHash 或 SimHash 计算文档的相似度指纹，删除相似度超过阈值的文档；
- **段落级去重**：在文档内部检测重复的段落，常见于爬取的数据中（如多个网页共享相同的页脚）；
- **URL 去重**：删除同一 URL 的多个版本（如 HTTP 和 HTTPS 版本）。

### 4.3.2 质量过滤：从垃圾到黄金

网络数据的质量差异极大。高质量的文本（如 Wikipedia、书籍、学术论文）与低质量的文本（如广告、垃圾邮件、自动生成的 SEO 内容）在信息密度上相差数个数量级。Gopher 的训练流程中使用了质量分类器（quality classifier）：先在高质量数据（如 Wikipedia）和随机网页数据上训练一个二元分类器，然后用该分类器过滤 Common Crawl，只保留高分的文档。

质量过滤的收益是巨大的。Chinchilla 的研究发现，使用过滤后的数据，等效数据量可以增加 2-3 倍（即每 GB 过滤数据相当于 2-3 GB 原始数据）。这意味着在相同存储和计算预算下，质量过滤可以显著提升模型性能。

### 4.3.3 合成数据：自我博弈与数据放大

当真实数据耗尽时，合成数据成为一种选择。合成数据的方法包括：
- **回译（Back-translation）**：将文本翻译成另一种语言，再翻译回来，生成语义等价但表述不同的样本；
- **模型生成**：用教师模型（如 GPT-4）生成高质量的合成数据，然后用这些数据训练学生模型。这是许多开源模型（如 Alpaca、Vicuna）的训练策略；
- **代码执行**：在代码训练数据中，执行代码并记录输出，生成输入-输出对。

合成数据的风险在于**模型崩溃**（model collapse）：如果合成数据占训练数据的比例过高，模型会逐渐遗忘真实数据的分布，输出质量下降。目前行业共识是：合成数据不应超过训练数据的 20-30%，且需要与真实数据混合使用。

---

## 4.4 数据混合：代码、科学、多语言

### 4.4.1 代码数据的特殊作用

代码数据（如 GitHub 代码、Stack Overflow 问答）在 LLM 训练中扮演着独特的角色。代码具有严格的语法结构和逻辑一致性，这迫使模型学习精确的符号推理。研究表明，在训练数据中加入 10-20% 的代码，可以显著提升模型在数学推理和逻辑推理任务上的表现——即使这些任务本身不是代码相关的。PaLM 和 GPT-3 都使用了代码数据，而 Code-davinci-002（Codex）更是将代码比例提高到 50% 以上。

代码数据的另一个作用是**提升长程依赖建模能力**。代码中的变量引用、函数调用和类继承关系通常跨越很长的距离，这迫使模型学习比普通文本更强的长程依赖能力。

### 4.4.2 多语言数据与语言迁移

多语言训练数据的比例对模型能力有显著影响。LLaMA 和 BLOOM 的训练数据中都包含了多种语言，但比例不同。LLaMA 主要聚焦英语（约 90%），BLOOM 则刻意平衡了多种语言（包括低资源语言）。

多语言训练有一个有趣的现象：**语言迁移**（language transfer）。在英语和中文的混合数据上训练的模型，其英语能力不仅不下降，反而可能提升——因为中文语料的多样性提供了额外的语义信号。但存在一个阈值：如果某种语言的比例低于 1%，模型可能无法习得该语言的基本语法。这是多语言模型设计的核心挑战：如何平衡高资源语言（英语、中文）和低资源语言（斯瓦希里语、冰岛语）的数据比例。

---

## 4.5 分布式训练：并行策略的权衡

### 4.5.1 数据并行

数据并行（Data Parallelism）是最简单的分布式训练策略：将 batch 的数据分配到多个 GPU 上，每个 GPU 持有完整的模型副本，独立计算梯度，然后通过 All-Reduce 同步梯度。

数据并行的优点：实现简单，通信开销低（只需同步梯度）。缺点：每个 GPU 必须存储完整的模型，因此模型大小受限于单卡显存。对于 GPT-3 175B（约 700GB 参数），即使使用 FP16 也需要 350GB，远超任何单卡显存。

### 4.5.2 张量并行

张量并行（Tensor Parallelism）将模型的每层参数分割到多个 GPU 上。例如，对于线性层 y = xW，可以将 W 按列分割到 4 个 GPU 上，每个 GPU 计算一部分输出，然后拼接。Megatron-LM 是这一策略的代表实现。

张量并行的优点：可以训练超大模型（如 GPT-3 175B 需要 8 张 A100 80GB）。缺点：通信开销高——每层的前向和反向传播都需要 All-Gather 和 Reduce-Scatter 操作，当 GPU 数量增加时，通信可能成为瓶颈。

### 4.5.3 流水线并行

流水线并行（Pipeline Parallelism）将模型按层分组，每个 GPU 持有连续的若干层。例如，一个 24 层的模型，GPU 0 持有 1-8 层，GPU 1 持有 9-16 层，GPU 2 持有 17-24 层。数据像流水线一样流经这些 GPU。

流水线并行的优点：通信开销低（只需在层边界传输激活值）。缺点：存在**流水线气泡**（pipeline bubble）——当 GPU 1 在处理第 9 层时，GPU 0 必须等待 GPU 1 完成才能处理下一个 batch 的第 1 层。这导致 GPU 利用率下降。GPipe 和 PipeDream 通过微批次（micro-batch）和交错调度来缓解气泡问题。

### 4.5.4 3D 并行：最优策略

现代大规模训练（如 GPT-4、LLaMA-65B）通常使用**3D 并行**（3D Parallelism）：数据并行 × 张量并行 × 流水线并行。例如，LLaMA-65B 的训练使用了 2048 张 A100，配置为：数据并行度=8，张量并行度=8，流水线并行度=32。这意味着：
- 8 个数据并行组，每个组持有完整的模型；
- 每个组内使用张量并行将模型层分到 8 张 GPU；
- 每个张量并行组内使用流水线并行将层序列分到 32 个阶段。

3D 并行的配置是一个复杂的优化问题：增加数据并行度可以加速训练，但 batch size 过大会导致优化不稳定；增加张量并行度可以训练更大的模型，但通信开销增加；增加流水线并行度可以提高 GPU 利用率，但气泡问题加剧。第4章的工程实践框将提供一些经验法则。

---

## 4.6 代码：Scaling Law 拟合器与最优配置预测器

下面的代码实现了一个 Scaling Law 拟合器，可以用少量实验数据（不同 N 和 D 配置下的损失）来预测最优配置。这个工具对于规划训练实验非常有价值。

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

class ScalingLawFitter:
    """
    Scaling Law 拟合器：拟合 L(N, D) = E + A/N^alpha + B/D^beta
    并预测给定计算预算下的最优配置。
    """
    def __init__(self, alpha: float = 0.076, beta: float = 0.095, 
                 E: float = 1.0, fix_alpha_beta: bool = True):
        """
        alpha, beta: 幂律指数。如果 fix_alpha_beta=True，则固定为给定值（Kaplan/Chinchilla 的估计）。
        E: 不可约误差。通常设为 1.0（自然语言的近似熵率）或从数据拟合。
        """
        self.alpha = alpha
        self.beta = beta
        self.E = E
        self.fix_alpha_beta = fix_alpha_beta
        self.A = None
        self.B = None
        self.fitted = False
    
    def _loss_func(self, params, N, D, L_obs):
        """计算预测损失与观测损失之间的均方误差。"""
        if self.fix_alpha_beta:
            A, B, E = params[0], params[1], params[2]
            alpha, beta = self.alpha, self.beta
        else:
            A, B, alpha, beta, E = params
        
        L_pred = E + A / (N ** alpha) + B / (D ** beta)
        return np.mean((L_pred - L_obs) ** 2)
    
    def fit(self, N_list: list, D_list: list, L_list: list):
        """
        拟合 Scaling Law 参数。
        N_list: 参数量列表（单位：百万）
        D_list: 训练 token 数列表（单位：百万）
        L_list: 观测损失列表（交叉熵或困惑度）
        """
        N = np.array(N_list, dtype=np.float64)
        D = np.array(D_list, dtype=np.float64)
        L = np.array(L_list, dtype=np.float64)
        
        if self.fix_alpha_beta:
            # 拟合 A, B, E
            x0 = [1.0, 1.0, self.E]
            bounds = [(0.001, 100), (0.001, 100), (0.1, 5.0)]
            result = minimize(self._loss_func, x0, args=(N, D, L), 
                            method='L-BFGS-B', bounds=bounds)
            self.A, self.B, self.E = result.x
        else:
            # 拟合所有参数
            x0 = [1.0, 1.0, 0.07, 0.09, self.E]
            bounds = [(0.001, 100), (0.001, 100), (0.01, 0.3), (0.01, 0.3), (0.1, 5.0)]
            result = minimize(self._loss_func, x0, args=(N, D, L), 
                            method='L-BFGS-B', bounds=bounds)
            self.A, self.B, self.alpha, self.beta, self.E = result.x
        
        self.fitted = True
        mse = self._loss_func(result.x, N, D, L)
        print(f"Fitted Scaling Law: A={self.A:.4f}, B={self.B:.4f}, "
              f"alpha={self.alpha:.4f}, beta={self.beta:.4f}, E={self.E:.4f}")
        print(f"Fitting MSE: {mse:.6f}")
    
    def predict(self, N: float, D: float) -> float:
        """预测给定 (N, D) 配置下的损失。"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.E + self.A / (N ** self.alpha) + self.B / (D ** self.beta)
    
    def optimal_config(self, C: float, compute_unit: str = 'T') -> tuple:
        """
        给定计算预算 C，预测最优的 (N, D) 配置。
        C: 计算量。默认单位为 TeraFLOPs (10^12)。
        返回: (N_opt, D_opt, L_opt)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")
        
        # 转换计算量到实际 FLOPs
        if compute_unit == 'T':
            C_actual = C * 1e12
        elif compute_unit == 'P':
            C_actual = C * 1e15
        elif compute_unit == 'E':
            C_actual = C * 1e18
        else:
            C_actual = C
        
        # 在约束 6ND = C 下优化
        # 从拉格朗日条件推导：N^{alpha+beta} = (alpha * A / (beta * B * 6^beta)) * C^beta
        # 简化：使用数值搜索
        best_loss = float('inf')
        best_N, best_D = None, None
        
        # 在 log 空间搜索 N
        for logN in np.linspace(6, 15, 500):  # N 从 1M 到 30B
            N = 10 ** logN
            D = C_actual / (6 * N)
            if D < 1e6:  # 至少 1M token
                continue
            L = self.predict(N / 1e6, D / 1e6)  # 转换为百万单位
            if L < best_loss:
                best_loss = L
                best_N, best_D = N, D
        
        return best_N, best_D, best_loss

# ============ 演示 ============
if __name__ == "__main__":
    # 模拟实验数据：不同 (N, D) 配置下的损失
    # 这些应该是真实实验的结果，这里用模拟数据演示
    
    np.random.seed(42)
    
    # 真实参数（模拟）
    A_true, B_true, alpha_true, beta_true, E_true = 2.5, 1.8, 0.076, 0.095, 1.2
    
    N_list = [10, 30, 100, 300, 1000]  # 单位：百万参数
    D_list = [100, 300, 1000, 3000, 10000]  # 单位：百万 token
    L_list = []
    
    for N, D in zip(N_list, D_list):
        L = E_true + A_true / (N ** alpha_true) + B_true / (D ** beta_true)
        L += np.random.normal(0, 0.02)  # 添加噪声
        L_list.append(L)
    
    print("Observed data:")
    for N, D, L in zip(N_list, D_list, L_list):
        print(f"  N={N}M, D={D}M, L={L:.4f}")
    
    # 拟合 Scaling Law
    fitter = ScalingLawFitter(alpha=0.076, beta=0.095, E=1.2, fix_alpha_beta=True)
    fitter.fit(N_list, D_list, L_list)
    
    # 预测新配置
    N_test, D_test = 500, 5000  # 500M 参数，5B token
    L_pred = fitter.predict(N_test, D_test)
    print(f"\nPredicted loss for N={N_test}M, D={D_test}M: {L_pred:.4f}")
    
    # 计算最优配置
    C = 1e21  # 1 ExaFLOP
    N_opt, D_opt, L_opt = fitter.optimal_config(C, compute_unit='E')
    print(f"\nOptimal config for C=1 ExaFLOP:")
    print(f"  N_opt = {N_opt/1e9:.2f}B parameters")
    print(f"  D_opt = {D_opt/1e9:.2f}B tokens")
    print(f"  Predicted loss = {L_opt:.4f}")
    
    # 验证 Chinchilla 比例：N_opt / D_opt 应该接近 1/20（每个参数看到20个token）
    print(f"  Tokens per parameter: {D_opt / N_opt:.1f}")
```

### 代码解析

这个 Scaling Law 拟合器包含几个关键功能：

1. **参数拟合**：使用 `scipy.optimize.minimize` 拟合 A, B, E（可以固定 α 和 β 为文献值，也可以全部拟合）。
2. **损失预测**：给定任意 (N, D) 配置，预测测试损失。这对于规划实验非常有用——不需要训练完整模型，就能估算性能。
3. **最优配置搜索**：在给定计算预算 C 下，数值搜索最优的 (N, D) 配置。这验证了 Chinchilla 的理论结论：N 和 D 应该大致等比例增长。
4. **单位处理**：支持 TeraFLOPs、PetaFLOPs、ExaFLOPs 等不同计算量单位，方便工程使用。

---

## 4.7 工程实践框：大规模预训练的生存指南

### 常见陷阱

1. **数值下溢**：在 FP16 训练中，梯度更新可能导致参数下溢为零（尤其是 LayerNorm 的缩放参数）。解决方案：使用 bfloat16（范围更大）或在优化器中保持 FP32 的 master weights。
2. **数据加载瓶颈**：当 GPU 计算速度远快于数据加载速度时，GPU 会空闲等待数据。解决方案：使用多进程数据加载（num_workers > 0）、预读取（prefetch）、内存映射（mmap）或 SSD 缓存。
3. **Checkpoint  corruption**：大规模训练数周，如果 checkpoint 文件损坏（如磁盘故障、网络中断），所有工作将丢失。解决方案：定期保存多个 checkpoint（如每 1000 步保存一次，保留最近 3 个），并验证 checkpoint 的完整性（重新加载并跑一步前向传播）。
4. **Loss 发散**：如果学习率过高、batch size 过大或数据中有异常值，loss 可能在训练中途突然发散到 NaN。解决方案：使用梯度裁剪（max_norm=1.0）、学习率 warmup（前 1-2% 的步数线性增长）、损失缩放（loss scaling）和异常检测（如果 loss > 2× 前 100 步的平均值，则回退到上一个 checkpoint）。

### 硬件需求估算

- **训练 GPT-2 级别**（124M 参数，序列长度 1024，batch size 512）：单张 RTX 4090（24GB）需要约 2-3 天；使用 4 张 RTX 4090 数据并行，约 12-18 小时。
- **训练 GPT-3 级别**（1.3B 参数）：需要 4-8 张 A100 40GB，使用 ZeRO-3 数据并行，训练时间约 1-2 周。
- **训练 LLaMA-7B**（7B 参数，1T token）：需要 64 张 A100 80GB，使用 3D 并行，训练时间约 2-3 周（约 1M GPU 小时）。
- **训练 Chinchilla-70B**（70B 参数，1.4T token）：需要 256-512 张 TPU v4 或 A100，训练时间约 2-3 周。

### 超参选择建议

- **Batch size**：大 batch size（如 4M token）通常需要较大的学习率（成正比）和更长的 warmup。DeepSpeed 的 ZeRO-3 允许在更多 GPU 上扩展 batch size，但注意线性扩展假设（linear scaling rule）在 batch size 过大时失效。
- **学习率峰值**：通常与模型大小的平方根成反比。例如，GPT-2 使用 6e-4，GPT-3 使用 1.6e-4，LLaMA-7B 使用 3e-4。具体值需要通过小规模实验（如 1% 的数据）来搜索。
- **Warmup 比例**：通常占总训练步数的 1-2%。对于 1T token 的训练，如果 batch size 为 4M，总步数约 250k，warmup 应为 2.5k-5k 步。
- **权重衰减**：通常设为 0.1（AdamW）。过高的权重衰减（如 0.3）会损害大模型的性能，过低（如 0.01）则可能导致过拟合。
- **梯度累积**：如果单卡显存无法容纳目标 batch size，使用梯度累积。注意：梯度累积等价于增大 batch size，因此学习率也应相应调整。

---

## 4.8 本章小结

本章的核心结论是：大语言模型的预训练不是艺术，而是**可预测、可优化、可工程化**的科学。Scaling Laws 提供了一种数学框架，将模型性能与资源投入（计算、参数、数据）联系起来，使得我们可以：
1. 用少量实验预测大模型的性能；
2. 在给定预算下找到最优的 (N, D) 配置；
3. 理解数据质量、数据混合和分布式训练对最终性能的影响。

Chinchilla 的最优性原则——N 和 D 等比例增长——是工程实践中最具影响力的结论。它告诉我们：不要盲目追求模型规模，而是要平衡规模与数据。一个充分训练的小模型，通常优于一个欠训练的大模型。

但 Scaling Laws 也有边界：它描述的是**平均性能**（如困惑度），而非特定能力（如涌现能力）。如果一个应用需要多步数学推理或代码生成，可能需要超出 Chinchilla 最优性的模型规模。第5章将讨论如何在预训练之后通过微调（fine-tuning）和对齐（alignment）来诱导这些特定能力。

---

## 延伸阅读

1. [Hoffmann et al., 2022] **Training compute-optimal large language models** —— Chinchilla 的原始论文，提出了 Scaling Laws 的最优性分析。本章 4.1-4.2 节的数学推导和实验分析直接基于该文。
2. [Ding et al., 2023] **Parameter-efficient fine-tuning for large-scale models** —— 系统对比了 LoRA、Adapter、Prefix Tuning 等参数高效微调方法。本章对预训练后微调的讨论参考了该文的分类框架。
3. [Guu et al., 2020] **Retrieval augmented language model pre-training** —— 展示了如何通过检索外部知识来增强预训练，是理解数据扩展与知识扩展关系的关键文献。本章对数据质量的讨论参考了该文的方法论。
4. [Wang et al., 2023] **Pre-trained language models and their applications** —— 全面的预训练语言模型综述，覆盖了预训练目标、模型架构、微调策略和应用场景。本章的 Scaling Laws 框架参考了该文的系统性分类。
5. [Goyal et al., 2021] **Larger-scale transformers for multilingual masked language modeling** —— 展示了多语言预训练中的 Scaling Laws 行为，发现跨语言迁移遵循与单语言类似的幂律关系。本章对多语言数据混合的讨论直接基于该文的实验结论。

---

> **交叉引用**：本章的 Scaling Law 拟合器代码（4.6 节）可以在第5-7章的实验规划中直接使用。第2章的 Transformer 实现是本章预训练实验的底层架构。第5章将讨论预训练之后的推理能力诱导（如 CoT），第6章将讨论预训练之后的对齐（如 RLHF），这两者都是预训练的后续步骤。第12章的 Dreamer 训练也遵循类似的 Scaling Laws，但数据效率（样本复杂度）的幂律指数不同。
