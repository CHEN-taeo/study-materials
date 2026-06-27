# 第2章：Transformer 的演化与极限（从 BERT/GPT 到 Mamba/MoE）

## 核心思想

Transformer 的自注意力机制（Self-Attention）是当代大语言模型的基石，但它的 O(n²) 计算复杂度和内存墙瓶颈（Memory Wall）在长序列场景下成为致命限制。Mamba 用选择性状态空间模型（Selective State Space Model）将序列建模降为线性复杂度，混合专家（Mixture of Experts, MoE）通过稀疏激活将参数规模扩展至万亿级别——这些演化不是对 Transformer 的否定，而是对其核心假设的重新权衡。

---

## 历史脉络：从循环到注意力的范式转移

在 Transformer 之前，序列建模的标准答案是循环神经网络（RNN）。RNN 的理论框架优雅：维护一个隐藏状态 h_t，在每一步将当前输入 x_t 和上一状态 h_{t-1} 结合，生成新的状态 h_t 和输出 y_t。这一设计直接对应了人类处理序列的直觉：读句子时，我们大脑中有一个"上下文状态"，不断被新单词更新。但 RNN 有一个致命的理论缺陷：它的状态转移是**局部迭代**的——h_t 只能通过 h_{t-1} 间接影响 h_{t+10}。当序列变长时，梯度需要通过连续的矩阵乘法反向传播，这导致了著名的梯度消失和梯度爆炸问题。LSTM 和 GRU 通过门控机制（输入门、遗忘门、输出门）部分缓解了这一问题，但只是缓解，而非解决。2017 年，即使是最先进的 LSTM 语言模型，其有效上下文长度也很难超过 100 个 token。

注意力机制（Attention Mechanism）的提出改变了这一局面。Bahdanau 等人在 2014 年为神经机器翻译引入注意力时，其核心动机是：RNN 的瓶颈状态（bottleneck state）强迫模型将源语言句子的全部信息压缩到一个固定维度的向量中，这对长句是不公平的。注意力允许解码器在每一步**直接查看**源句子的所有位置，并通过加权平均计算上下文向量。Vaswani 等人在 2017 年的关键洞见是：如果注意力已经足够强大，为什么我们还需要 RNN？Transformer 完全抛弃了循环结构，将注意力作为唯一的序列建模机制。这一决策的代价是计算量的增加（从 RNN 的 O(n) 到自注意力的 O(n²)），但收益是长程依赖的直接建模能力和并行计算效率。后续的 BERT 和 GPT 分别将这一架构推向了理解任务和生成任务的巅峰，但二次复杂度始终是一个悬而未决的工程问题。FlashAttention 通过 IO-aware 的内存优化将实际计算时间降低了一个数量级，而 Mamba 和 MoE 则从架构层面挑战了 Transformer 的二次假设——它们分别用线性复杂度和稀疏激活开辟了新的扩展维度。

---

## 2.1 从循环到注意力：为什么 RNN 失败了

### 2.1.1 RNN 的数学困境

RNN 的前向传播公式简洁：

h_t = tanh(W_h h_{t-1} + W_x x_t + b)

在反向传播时，损失函数对 h_{t-1} 的梯度需要通过链式法则传递：

∂L/∂h_{t-1} = ∂L/∂h_t · ∂h_t/∂h_{t-1} = ∂L/∂h_t · diag(1 - tanh²(·)) · W_h

当时间步长 T 很大时，梯度需要连乘 T 次 W_h。如果 W_h 的最大奇异值大于 1，梯度会指数爆炸；如果小于 1，梯度会指数消失。LSTM 通过引入细胞状态 c_t 和门控机制，将梯度传播路径从主路径（h_t）分离到一条"高速公路"（c_t），但这条高速公路的畅通是有条件的——它依赖于遗忘门接近 1。在实践中，当序列超过几百个时间步时，即使 LSTM 也难以保持稳定的梯度流。

### 2.1.2 注意力的直觉与数学

注意力机制打破了 RNN 的局部迭代限制。在 Bahdanau 注意力中，上下文向量 c_t 是源句子所有隐藏状态的加权平均：

c_t = Σ_j α_{tj} h_j

其中 α_{tj} 是注意力权重，由解码器当前状态 s_t 和源状态 h_j 的相似度计算：

α_{tj} = softmax(score(s_t, h_j))

score(s_t, h_j) = s_t^T W h_j

这一设计的革命性在于：无论 j 和 t 之间的距离有多远，注意力权重 α_{tj} 都是**直接计算**的——不需要通过中间状态传播。这彻底消除了长程依赖的梯度衰减问题。

但 Bahdanau 注意力仍然是一个辅助模块，依附于 RNN 编码器-解码器框架。Vaswani 等人的关键创新是：如果注意力本身就能完成序列建模，为什么不把 RNN 完全去掉？

---

## 2.2 Transformer 的数学本质：自注意力的三大假设

### 2.2.1 Scaled Dot-Product Attention

Transformer 的自注意力（Self-Attention）将输入序列 X ∈ R^{n×d} 通过三个线性投影映射为 Query (Q)、Key (K)、Value (V)：

Q = XW_Q,  K = XW_K,  V = XW_V

注意力输出是：

Attention(Q, K, V) = softmax(QK^T / √d_k) V

这里的 softmax 操作在序列维度上进行，每个输出位置 i 对输入序列的所有位置 j 计算权重。缩放因子 √d_k 是为了防止当维度 d_k 较大时，点积的数值过大导致 softmax 进入饱和区（梯度极小）。

### 2.2.2 为什么不是简单的"相似度匹配"

对自注意力的常见误解是："Query 就像搜索，Key 就像索引，Value 就像内容。"这个类比有启发性，但不完整。更准确的理解是：

- **Query** 代表当前 token 的"需求"——我（当前词）需要什么信息？
- **Key** 代表每个 token 的"供给"——我（每个词）能提供什么信息？
- **Value** 代表每个 token 的"内容"——我（每个词）的实际语义内容是什么？

QK^T 的物理意义是**需求与供给的匹配程度**：对于当前位置 i，计算它与所有位置 j 的匹配分数，然后按分数加权平均 V。这不是简单的信息检索，而是**动态的特征聚合**：每个位置的输出都是所有输入位置的加权组合，权重由当前位置的需求决定。

### 2.2.3 位置编码：Transformer 的隐形假设

自注意力本身是**置换不变**的（permutation invariant）：如果重新排列输入序列，注意力输出的集合不变（只是对应位置的输出被重新排列）。这意味着 Transformer 在没有任何额外机制的情况下，无法区分"第1个词"和"第10个词"的位置差异。

原始 Transformer 使用正弦/余弦位置编码：

PE(pos, 2i) = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

这种编码的巧妙之处在于：通过不同频率的正弦波，模型可以学习相对位置关系——任意两个位置 pos 和 pos+k 的编码之间的点积只取决于 k（相对距离），而不取决于 pos 的绝对值。后续工作（如 RoPE、ALiBi）进一步优化了位置编码，使得外推（extrapolation）到训练时未见的序列长度成为可能。第4章将讨论位置编码对长上下文 LLM 的影响。

### 2.2.4 多头注意力：并行子空间假设

Transformer 不是只计算一次注意力，而是计算 h 次（通常 h=8 或 16），每次使用不同的投影矩阵，然后将结果拼接：

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

多头设计的动机是：不同的注意力头可以关注不同的关系模式。实证研究表明，某些头专门关注语法关系（如主语-动词一致），某些头关注共指关系（代词与先行词），某些头关注局部 n-gram 模式。第6章的可解释性研究将深入讨论注意力头的专业化分工。

---

## 2.3 BERT vs GPT：编码器与解码器的分野

### 2.3.1 BERT 的双向编码器

BERT（Bidirectional Encoder Representations from Transformers）使用 Transformer 的**编码器**堆叠 [Devlin et al., 2018]。编码器使用双向自注意力：每个位置都可以 attending 到所有位置（包括前面的和后面的）。BERT 的预训练任务是掩码语言建模（Masked Language Modeling, MLM）——随机掩码输入中的某些 token，让模型根据上下文预测被掩码的词。这一设计使得 BERT 在文本理解任务（如情感分析、问答、命名实体识别）上表现卓越，因为理解任务通常需要双向上下文。

但 BERT 有一个根本限制：它不能生成文本。MLM 目标不是自回归的，模型在训练时已经"看到"了未来的 token（被掩码的词的上下文），因此它无法被直接用于逐词生成。要将 BERT 用于生成，需要额外的解码模块（如 GPT 风格的解码器），这破坏了预训练-微调的一致性。

### 2.3.2 GPT 的自回归解码器

GPT 系列使用 Transformer 的**解码器**堆叠。解码器使用**因果自注意力**（causal self-attention）：每个位置 i 只能 attending 到位置 ≤ i 的 token，通过掩码（masking）上三角矩阵实现。GPT 的预训练目标是标准的自回归语言建模：给定前缀，预测下一个 token。这一设计天然适合生成任务，因为训练目标和推理目标完全一致。

GPT 的局限性在于：它不能充分利用双向上下文。在理解任务中，BERT 的双向编码通常优于 GPT 的单向编码。这也是 GPT-3 和 GPT-4 在 few-shot 场景下才展现出理解能力的原因——它们不是通过架构的双向性，而是通过大规模预训练隐含地学习到了双向关系的近似。

### 2.3.3 编码器-解码器的统一：T5 与 BART

T5 和 BART 采用编码器-解码器架构，试图统一理解和生成。T5 将所有 NLP 任务框架为"文本到文本"转换，使用 span corruption 预训练目标（掩码连续的 token 片段，让解码器生成被掩码的内容）。BART 则使用更激进的破坏策略（token 删除、填充、置换、旋转）。这些模型在翻译和摘要任务上表现优异，但参数量相同时，纯解码器模型（如 GPT）在生成任务上仍具有优势。工程界的共识是：如果你的任务以生成为主，GPT 风格的解码器是更高效的架构选择；如果以理解为主，BERT 或 T5 更合适；如果两者都需要，GPT-4 级别的纯解码器模型在足够大时可以通过 prompt 工程实现双向理解。

---

## 2.4 效率革命：FlashAttention 与内存墙

### 2.4.1 注意力的内存瓶颈

标准注意力计算的时间复杂度是 O(n²·d)，空间复杂度也是 O(n²)（需要存储 n×n 的注意力矩阵）。在训练时，为了反向传播，还需要存储中间激活值，这使得内存开销进一步翻倍。对于序列长度 n=4096 和批量大小 32，仅注意力矩阵就需要 32 × 4096² × 4 bytes ≈ 2 GB 内存。当 n=32768（如 Claude 的长上下文）时，这一开销变为 128 GB——即使是 A100 80GB 也吃不消。

### 2.4.2 FlashAttention 的 IO-Aware 优化

FlashAttention [Dao et al., 2022] 的核心洞见是：GPU 的内存层次结构（HBM → SRAM → 寄存器）中，HBM（高带宽内存）到 SRAM（共享内存/缓存）的读写带宽是计算瓶颈。标准注意力算法需要将 n×n 的注意力矩阵完整地 materialize（物化）到 HBM 中，这导致大量的 IO 开销。FlashAttention 通过**分块计算**（tiling）和**在线 softmax**（online softmax），将注意力计算分解为可以在 SRAM 中完成的小块，避免了对完整注意力矩阵的存储需求。

具体来说，FlashAttention 将 Q、K、V 矩阵分成小块（block size B），每次只加载 B×d 的块到 SRAM，计算局部的注意力分数，然后更新输出累加器。在线 softmax 技术允许在不存储完整 softmax 分母的情况下，增量地更新 softmax 输出。这使得 FlashAttention 的时间复杂度仍然是 O(n²·d)，但实际运行速度提升了 2-8 倍，内存复杂度从 O(n²) 降低到 O(n)。

### 2.4.3 FlashAttention 的局限与扩展

FlashAttention 不改变注意力的二次复杂度——它只优化了常数因子。对于 n > 100k 的极长序列，二次复杂度仍然不可接受。后续工作（FlashAttention-2、FlashAttention-3）进一步优化了并行性和硬件利用率，但本质上仍是二次的。这是 Mamba 等替代架构出现的技术背景：当工程优化耗尽后，我们需要改变复杂度本身。

---

## 2.5 超越 Transformer：Mamba 与 MoE

### 2.5.1 Mamba：选择性状态空间模型

Mamba [Gu & Dao, 2023] 的核心动机是：RNN 的线性复杂度和 Transformer 的表达能力能否兼得？状态空间模型（State Space Model, SSM）提供了一个可能的答案。经典 SSM 将序列映射定义为线性时不变系统：

h'(t) = Ah(t) + Bx(t)
y(t) = Ch(t) + Dx(t)

在离散化后，这转化为递归更新：

h_k = Ā h_{k-1} + B̄ x_k
y_k = C h_k + D x_k

其中 Ā 和 B̄ 是离散化后的参数。如果 A 是良态的（如对角矩阵），这个递归可以在 O(n·d) 时间内完成，因为每个时间步的计算不依赖于其他时间步。

但经典 SSM 有一个问题：它是**时不变**的（time-invariant）——参数 A, B, C 不随输入变化。这意味着模型无法选择性地关注或忽略某些输入 token。Mamba 的关键创新是**选择性 SSM**（Selective SSM）：让 B、C 和步长 Δ 成为输入 x 的函数：

B = Linear_B(x),  C = Linear_C(x),  Δ = Linear_Δ(x)

这使得模型可以根据输入内容**选择性地**更新状态——对于无关的 token，B 可以接近零，状态几乎不变；对于关键 token，B 可以很大，状态大幅更新。这种选择性机制赋予了 Mamba 类似注意力"关注特定位置"的能力，但保持了线性复杂度。

### 2.5.2 Mamba 的架构与实现

Mamba 的完整架构包含一个输入投影、一个选择性 SSM 核心和一个输出投影。与 Transformer 不同，Mamba 不需要位置编码——状态空间的递归结构天然编码了时序信息。Mamba 的并行训练通过**并行扫描**（parallel scan）算法实现：虽然前向传播是递归的，但可以通过关联扫描（associative scan）在 log(n) 并行步内计算所有时间步的状态。

实验表明，Mamba 在语言建模（困惑度）和长序列任务上匹配甚至超过同等规模的 Transformer，但训练速度更快、内存占用更少。然而，Mamba 并非 Transformer 的完全替代品：在需要复杂全局推理的任务（如多步数学推理、长程依赖的代码生成）上，Transformer 的二次注意力仍然表现出优势。工程上的共识是：Mamba 适合**长序列、低延迟**的场景（如实时音频处理、长文档摘要），而 Transformer 适合**需要复杂全局交互**的场景。

### 2.5.3 Mamba-2：SSM 与注意力的统一视角

Mamba-2 [Dao & Gu, 2024] 提出了一个关键洞见：状态空间模型和注意力机制可以被统一在同一个数学框架下。具体而言，Mamba-2 将 SSM 中的选择性状态转移重新解释为**结构化矩阵乘法**（structured matrix multiplication），并证明自注意力是这种结构化矩阵的一个特例。这一发现的意义在于：

1. **理论澄清**：Mamba 的线性复杂度和 Transformer 的二次注意力不再是两种完全不同的机制，而是同一连续谱上的两个端点。selective SSM 可以看作是一种**带结构的线性注意力**（structured linear attention），它在保持线性复杂度的同时，仍然具备一定的位置选择性。
2. **工程收益**：Mamba-2 的并行化比原始 Mamba 更高效，因为它可以通过块级矩阵分解（block decomposition）在更大的 tile 上并行计算，而不是逐 token 的并行扫描。
3. **统一视角**：这为后续设计"混合架构"（hybrid architecture）提供了理论基础——在需要长程记忆的位置用 SSM，在需要全局注意的位置用标准注意力，而不是非此即彼。

但需要注意，Mamba-2 仍然无法解决所有问题。在需要精确长程依赖的任务（如引用解析、长文档中的事实追踪）上，Transformer 的二次注意力仍然更可靠。Mamba-2 的价值在于把替代架构从"对立"关系重新定位为"互补"关系。

### 2.5.4 混合专家（MoE）：参数规模的稀疏扩展

MoE 不是 Transformer 的替代，而是 Transformer 的扩展。MoE 的核心思想是：与其用一个巨大的密集网络处理所有输入，不如用多个"专家"（通常是小型前馈网络），并通过一个门控网络（Gating Network）为每个输入选择性地激活少数专家。

在 MoE 层中，输入 x 首先通过门控网络计算路由分数：

g(x) = TopK(softmax(W_g · x), k)

然后只激活 k 个专家（通常是 k=1 或 2），输出是这些专家输出的加权组合：

y = Σ_{i∈TopK} g_i(x) · E_i(x)

由于每个 token 只激活少数专家，MoE 的**计算量**与同等大小的密集模型相近，但**参数量**可以是密集模型的数十倍。Mixtral 8×7B 就是一个典型例子：它有 8 个 7B 参数的专家，总参数量 47B，但每个 token 只激活 2 个专家，实际计算量约等于 12B 参数模型。

MoE 的工程挑战在于**负载均衡**（load balancing）：如果门控网络总是选择少数几个专家，其他专家将不被训练，导致参数浪费。Switch Transformer 通过辅助损失函数（auxiliary loss）惩罚专家使用的不均衡，GShard 和 GLaM 进一步扩展了 MoE 到分布式训练场景。

### 2.5.5 国产 MoE 的关键创新：DeepSeek-MoE 与 Qwen-MoE

2024 年的 MoE 工程实践出现了重要创新，主要来自国内团队，这些设计正在影响全球 MoE 模型的工程标准。

**DeepSeek-MoE [Dai et al., 2024]** 提出了两个关键设计：

1. **共享专家（Shared Expert）**：在多个专家之外引入一个始终被激活的共享专家。这个设计基于一个观察：总有一些"基础能力"（如语法、基本推理）是所有输入都需要的，与其让每个 token 单独路由到这些基础专家，不如固定一个共享专家来处理通用知识，从而释放其他专家去学习更专业化的模式。
2. **细粒度专家（Fine-grained Expert）**：将大专家拆分为多个小专家，每个专家只负责一个更窄的语义子空间。通过更细粒度的路由，模型可以更精确地激活相关知识。DeepSeek-MoE 的实验表明，在相同参数量下，细粒度专家 + 共享专家的组合比传统粗粒度 MoE 有更好的性能。

**Qwen-MoE [Qwen team, 2024]** 则进一步优化了训练和推理效率：

1. **专家分片与通信优化**：通过 expert parallelism 的精细调度，减少 GPU 之间的专家路由通信开销；
2. **Fine-grained expert segmentation**：类似 DeepSeek 的细粒度设计，但结合了 Qwen 系列已有的模型架构优化；
3. **训练稳定性**：在路由函数上引入辅助损失和约束，防止训练初期的专家崩溃（expert collapse）问题。

这些国产 MoE 设计的共同点是：不再把 MoE 仅仅当作"参数放大器"，而是把它当作**专家分工机制**。这要求设计者不仅考虑负载均衡，还要考虑专家的专业化程度和基础能力的共享机制。

MoE 的第二个挑战是**推理效率**：虽然计算量低，但需要在 GPU 间路由激活的专家，通信开销可能抵消计算收益。DeepSeek-MoE 和 Qwen-MoE 通过共享专家和更细粒度的路由策略，在一定程度上缓解了这个问题。第4章将讨论 MoE 与 Scaling Laws 的交互关系。

---

## 2.6 代码：从零实现 Multi-Head Self-Attention

下面的代码实现了一个完整的 Multi-Head Self-Attention 模块，包含位置编码、因果掩码和可选的 FlashAttention 风格内存优化。这个实现可以直接插入到任何 Transformer 架构中。

```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """
    标准 Multi-Head Self-Attention 实现。
    支持：因果掩码、可选的 FlashAttention 风格内存优化、可配置的头数。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 causal: bool = False, use_flash: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal
        self.use_flash = use_flash  # 如果 PyTorch >= 2.0，使用内置 scaled_dot_product_attention
        
        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        mask: [batch, seq_len, seq_len] 或 None
        返回: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影到 Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头: [batch, num_heads, seq_len, d_head]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        if self.use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 内置 FlashAttention 优化
            attn_mask = None
            if self.causal:
                attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.causal and attn_mask is None
            )
        else:
            # 手动实现注意力
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, heads, seq, seq]
            
            # 因果掩码
            if self.causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # 可选的外部 mask
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V)  # [batch, heads, seq, d_head]
        
        # 合并多头: [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        return out

class TransformerBlock(nn.Module):
    """标准 Transformer Block：Attention + FFN + LayerNorm + Residual"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout, causal)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 架构
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ============ 演示 ============
if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size, seq_len, d_model = 2, 16, 64
    num_heads = 8
    
    # 输入嵌入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 位置编码
    pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512)
    x = pos_enc(x)
    
    # Transformer Block
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=256, causal=True)
    
    # 前向传播
    output = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    
    # 因果性检查：位置 i 的输出不应受位置 i+1 之后的影响
    # 修改最后一个 token 的输入，检查第一个 token 的输出是否变化
    x_modified = x.clone()
    x_modified[:, -1, :] += 10.0
    output_modified = block(x_modified)
    diff = (output[:, 0, :] - output_modified[:, 0, :]).abs().max()
    print(f"Causality check (first token unchanged by last token): {diff.item():.6f}")
```

### 代码解析

这个实现包含几个关键细节：

1. **多头重塑**：通过 `view` 和 `transpose` 将 `[batch, seq, d_model]` 转换为 `[batch, heads, seq, d_head]`，使得每个头可以独立计算注意力。
2. **因果掩码**：使用 `torch.triu` 生成上三角掩码，确保位置 i 只能 attending 到位置 ≤ i。
3. **FlashAttention 回退**：如果 PyTorch ≥ 2.0，自动使用 `scaled_dot_product_attention`（底层调用 FlashAttention 内核）；否则回退到手动实现。
4. **Pre-Norm 架构**：在应用 attention 和 FFN 之前先做 LayerNorm，这是现代 Transformer 的标准做法（比原始论文的 Post-Norm 更稳定）。
5. **因果性验证**：通过修改最后一个 token 的输入，验证第一个 token 的输出是否不变，从而确认因果性正确实现。

---

## 2.7 工程实践框：Transformer 训练与部署的实战要点

### 常见陷阱

1. **位置编码的数值不稳定**：正弦位置编码在超长序列（>4096）外推时可能出现数值问题，因为 sin/cos 函数在高频区域的数值精度下降。解决方案：使用 RoPE 或 ALiBi 替代正弦编码，或在外推时对位置编码进行插值（如 LLaMA 的 Position Interpolation）。
2. **多头注意力的头数选择**：d_model 必须能被 num_heads 整除。常见配置是 d_model=768, num_heads=12（d_head=64）或 d_model=1024, num_heads=16。头数过多会导致每个头的维度过低（d_head < 32），表达能力不足；头数过少则无法捕捉多样化的关系模式。
3. **FFN 的扩展比**：d_ff（前馈网络隐藏维度）通常设置为 4×d_model（如 d_model=768, d_ff=3072）。降低扩展比（如 2×）可以减少参数量，但可能损害模型的非线性表达能力。

### 硬件需求估算

- **训练一个 GPT-2 级别模型**（124M 参数，序列长度 1024）：单张 RTX 4090（24GB）可以训练，但 batch size 受限（约 8-16）。使用梯度累积（gradient accumulation）可以模拟更大的 batch size。
- **训练一个 GPT-3 级别模型**（1.3B 参数）：需要 4-8 张 A100 40GB，使用 ZeRO-3 数据并行。
- **训练一个 MoE 模型**（如 Mixtral 8×7B）：虽然每个 token 的计算量约 12B，但总参数量 47B 需要大量显存存储。推理时需要 2-4 张 A100 80GB，或使用 4-bit 量化降至单卡。
- **FlashAttention 的内存节省**：对于序列长度 8192，FlashAttention 相比标准注意力节省约 8-10 倍内存。这意味着在 24GB 显存上，标准注意力的最大序列长度约 2048，而 FlashAttention 可达 8192。

### 超参选择建议

- **学习率调度**：Transformer 对学习率非常敏感。通常使用 Warmup + Cosine Decay：前 2000-4000 步线性 warmup 到峰值（如 6e-4），然后余弦衰减到最小值（峰值的 10%）。峰值学习率与模型大小成反比：模型越大，峰值学习率越低。
- **Dropout 率**：预训练大模型（>1B）通常使用较低 dropout（0.0-0.1），因为数据量足够大时过拟合风险低。微调时可以提高 dropout（0.1-0.3）防止过拟合。
- **梯度裁剪**：Transformer 训练极易出现梯度爆炸，必须使用梯度裁剪（max_norm=1.0）。这是稳定训练的必要条件，而非可选优化。
- **混合精度训练**：使用 torch.cuda.amp 或 bfloat16 可以将内存占用减半、训练速度提升 1.5-2 倍。但注意：Softmax 和 LayerNorm 应保持 fp32 精度，否则数值不稳定。

---

## 2.8 本章小结

本章的核心论点可以概括为：Transformer 不是序列建模的终极答案，而是当前扩展性（scalability）和表达能力之间的最优权衡。自注意力机制通过二次复杂度换取了长程依赖的直接建模，这一 trade-off 在短序列（<4k）上是值得的，但在长序列上催生了 FlashAttention 等工程优化和 Mamba 等架构替代。BERT 和 GPT 的编码器-解码器分野，本质上是双向理解与单向生成之间的任务适配选择，而非架构的绝对优劣。MoE 则从另一个维度——参数规模而非计算复杂度——扩展了 Transformer 的能力边界。

理解这些演化的关键，不在于记住每个架构的具体参数，而在于掌握一个分析框架：**任何序列建模架构都在处理三个竞争目标——表达能力（能建模什么依赖关系）、计算效率（训练/推理的时间和空间成本）、扩展性（随着规模增长，性能如何变化）**。Transformer 在表达能力上优于 RNN，在计算效率上劣于 Mamba，在扩展性上优于早期架构。第3章将把这一分析框架推广到跨模态表征，展示为什么文本、视觉和物理世界的建模最终都收敛到了注意力或注意力变体。

---

## 延伸阅读

1. [Radford et al., 2018] **Improving language understanding by generative pre-training** —— GPT-1 的原始论文，首次证明了生成式预训练在理解任务上的有效性。本章对 GPT 路线的分析基于该文的架构选择。
2. [Dao et al., 2022] **FlashAttention: Fast and memory-efficient exact attention with IO-awareness** —— FlashAttention 的技术报告，详细推导了 IO-aware 分块算法。本章 2.4 节的内存分析大量参考了该文的实验数据。
3. [Ding et al., 2023] **Parameter-efficient fine-tuning for large-scale models** —— 系统对比了 LoRA、Adapter、Prefix Tuning 等参数高效微调方法，对 Transformer 的后续应用有直接指导意义。第4章和第6章将引用该文的实验结论。
4. [Amatriain et al., 2023] **Transformer models: an introduction and catalog** —— 最全面的 Transformer 变体分类目录，覆盖了 BERT、GPT、T5、BART、Switch Transformer 等数十种架构。本章的架构对比框架参考了该文的分类体系。
5. [Hunyuan-TurboS, 2025] **Mamba-Transformer synergy: architecture and performance analysis** —— 2025 年的最新工作，系统对比了 Mamba 与 Transformer 在语言建模、长序列和推理任务上的 trade-off。本章 2.5 节的分析参考了该文的实验结论。

---

> **交叉引用**：本章的 Transformer 实现代码（2.6 节）将在第4章的预训练实验中被复用和扩展。第3章将讨论如何把自注意力机制从文本推广到视觉和物理状态。第8章的 ReAct 实现将使用本章的因果注意力模块来处理 Agent 的对话历史。第12章的 Dreamer RSSM 将与本章的注意力架构形成对比，展示为什么循环结构更适合时序动力学建模。
