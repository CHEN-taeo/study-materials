# 第3章：表征学习——文本、视觉、世界动态的共享数学结构

## 核心思想

文本的上下文嵌入、视觉的 CNN/ViT 特征、世界模型的潜在状态，本质都是同一个数学问题的三种实例化：将高维观测映射到低维潜在空间，在潜在空间中保持语义相似性为几何距离。从 Word2Vec 的分布式假设到 CLIP 的跨模态对齐，再到 JEPA 的联合嵌入预测，这些技术的共性在于"压缩与预测"——不是简单地存储数据，而是学习数据的生成结构和动态规律。

---

## 历史脉络：从词向量到统一潜在空间的四十年

表征学习（Representation Learning）的源头可以追溯到认知科学中的"概念空间"理论。1980 年代，心理学家 Gardenfors 提出，人类概念可以通过高维空间中的几何区域表示，语义相似性对应于空间中的距离。这一思想在 2000 年代被引入机器学习：Word2Vec（2013）证明了词向量可以通过简单的预测任务（预测上下文词）从大规模语料中学习，且语义关系（如"国王 - 男人 + 女人 ≈ 女王"）表现为向量算术。但 Word2Vec 的表征是**静态的**——每个词只有一个向量，不考虑语境。

Transformer 的出现改变了这一点。BERT 和 GPT 的上下文嵌入是**动态的函数**——同一个词在不同句子中获得不同向量。这意味着表征从"词是什么"升级为"词在当前语境中扮演什么角色"。2020 年，CLIP 将这一思想推向跨模态：通过对比学习，图像和描述它的文本被映射到同一个潜在空间。2021 年，Dreamer 的 RSSM 证明物理世界的动力学状态也可以被压缩为低维潜在向量，并在潜在空间中进行预测和规划。2023 年，LeCun 的 JEPA 框架提出了一个更激进的主张：表征学习的目标不应该是重建输入（如 VAE），也不应该是区分正负样本（如 CLIP），而应该是**预测压缩后的表征**——即学习"在这个状态下，下一个状态的表征是什么"。

这一演化线索中蕴含着多种不同的目标函数：Word2Vec 的上下文预测学习语义关系；BERT/GPT 的动态上下文嵌入学习语境函数；CLIP 的对比学习学习跨模态对齐；Dreamer 的 RSSM 学习像素重建和奖励预测；JEPA 学习压缩表征之间的预测。它们并非回答同一个问题的线性进化，而是在表征学习的不同维度上探索：重建、对比、预测、跨模态、动态性。把它们强行串成"从描述性到预测性"的单线叙事，会误导读者忽略这些工作的根本差异。真正贯穿这些工作的主题是"压缩与结构"——不是简单地存储数据，而是学习数据中的统计结构、语义结构和动态规律。至于哪一种结构更重要、哪一种目标函数更通用，仍然是一个开放问题。本章将沿着这一线索，揭示文本、视觉和物理世界表征的共享数学基础，同时为它们之间的本质差异保留空间，并为第7章的多模态融合和第14章的协同与集成奠定基础。

---

## 3.1 分布式表征的语义几何

### 3.1.1 Word2Vec 的分布式假设

Word2Vec 的核心假设是**分布式假设**（Distributional Hypothesis）："上下文相似的词，语义也相似"。Skip-gram 模型通过预测给定词的上下文词来学习词向量：

L = -Σ_{t} Σ_{-c≤j≤c, j≠0} log P(w_{t+j} | w_t)

其中 P(w_O | w_I) = exp(v_{w_O}'^T · v_{w_I}) / Σ_w exp(v_w'^T · v_{w_I})

这里的 v_{w_I} 是输入词嵌入，v_{w_O}' 是输出词嵌入。在训练完成后，输入嵌入 v_w 被用作词的表征向量。

Word2Vec 的惊人发现是：这些向量不仅编码了语义相似性，还编码了**语义关系**的线性结构。例如，"king - man + woman ≈ queen"的向量算术在嵌入空间中近似成立。这一现象的数学解释是：Word2Vec 的隐式矩阵分解等价于 PMI 矩阵的降维，而语义关系在这种低维投影中近似保持线性。

但 Word2Vec 有两个根本性限制：
1. **静态性**：每个词只有一个向量，无法处理多义词（如"bank"可以是银行或河岸）；
2. **局部性**：上下文窗口通常只有 5-10 个词，无法捕捉长程依赖和全局语义结构。

### 3.1.2 从静态到动态：上下文嵌入

ELMo（2018）首次提出上下文嵌入：用双向 LSTM 为每个词生成一个依赖于整个句子的向量。但 LSTM 的梯度问题限制了上下文的质量。Transformer 的上下文嵌入彻底解决了这一问题：通过自注意力机制，每个位置的嵌入可以"看到"整个序列的所有位置，并根据相关性加权聚合信息。

BERT 的上下文嵌入是**深层表征**：输入经过 12-24 层 Transformer 后，每一层都产生一个中间表征。研究表明，BERT 的底层（1-4 层）编码词法信息（词性、形态），中层（5-8 层）编码句法信息（依存关系、短语结构），顶层（9-12 层）编码语义信息（语义角色、实体关系）。这种分层结构不是人为设计的，而是预训练任务（MLM）的自然涌现。

GPT 的上下文嵌入则有所不同：由于因果掩码的存在，位置 i 的嵌入只能 attending 到位置 ≤ i 的 token。这使得 GPT 的表征是**前缀条件的**（prefix-conditioned）——给定前缀，预测下一个 token 的表征。这种单向性在生成任务中是优势，但在理解任务中是劣势。

---

## 3.2 文本表征的动态性：为什么上下文嵌入比静态词向量更强大

### 3.2.1 多义词的消解

上下文嵌入解决静态词向量的核心问题：多义词。在 Word2Vec 中，"bank"只有一个向量，是"银行"和"河岸"的某种平均。但在 BERT 中，"I went to the bank to deposit money"和"The river bank was covered in flowers"中的"bank"会获得完全不同的向量，因为它们的上下文（"deposit money" vs "river"）引导注意力机制选择了不同的语义维度。

这种动态性不仅是工程优势，更是**语义理论**的进步。传统语义学认为词义是词典式的（lexical）——每个词有一个固定的定义。上下文嵌入则支持**用法论**（Use Theory）语义观：词的意义不是固定的，而是由其使用方式（上下文）动态决定的。这一观点与 Wittgenstein 的"语言游戏"理论不谋而合。

### 3.2.2 深层表征的可分解性

BERT 的深层表征还有一个有趣的性质：**可分解性**。通过 probing（探测）实验，研究者发现：
- 在 BERT 的第 3 层，线性分类器可以从表征中准确预测词性（POS tagging）；
- 在第 7 层，线性分类器可以预测依存关系（dependency parsing）；
- 在第 11 层，线性分类器可以预测语义角色（semantic role labeling）。

这意味着 BERT 的表征空间是**层次化组织**的：低层编码低级语言特征，高层编码高级语义特征。这种层次性不是人为设计的，而是预训练任务（从简单预测到复杂推理）的渐进学习结果。这与人类语言习得的顺序（先词法，后句法，再语义）惊人地一致。

---

## 3.3 视觉表征：从 CNN 到 Vision Transformer

### 3.3.1 CNN 的归纳偏置

在 Transformer 之前，计算机视觉的标准是卷积神经网络（CNN）。CNN 的核心假设是**局部性**和**平移不变性**：图像的特征在局部区域（如 3×3 或 5×5 的感受野）内提取，且同一特征在不同位置共享参数（卷积核）。这种归纳偏置（inductive bias）使得 CNN 在图像分类、检测和分割任务上取得了巨大成功。

但 CNN 的局部性假设也是其局限。当需要建模全局关系（如"图中的猫在狗的左边"）时，CNN 必须通过多层卷积逐步扩大感受野，这在深层网络中效率低下且信息丢失严重。此外，CNN 的卷积核权重是位置共享的，这意味着模型无法天然地学习位置相关的模式（如"人脸总是在图像上方"）。

### 3.3.2 Vision Transformer：把图像当作词序列

Vision Transformer（ViT）的核心思想是：将图像分割为固定大小的 patch（如 16×16 像素），将每个 patch 展平并线性投影为向量，然后像处理词序列一样用 Transformer 处理这些 patch 向量。ViT 的输入序列是：

z_0 = [x_class; x_p^1 E; x_p^2 E; ... ; x_p^N E] + E_pos

其中 x_p^i 是第 i 个 patch，E 是投影矩阵，x_class 是分类 token，E_pos 是位置嵌入。

ViT 的优势在于：自注意力机制可以直接建模任意两个 patch 之间的关系，无论它们在图像中的空间距离有多远。这消除了 CNN 的局部性限制，使得模型可以天然地学习全局语义关系（如"天空在上方"、"地面在下方"）。

但 ViT 也有一个代价：由于缺乏 CNN 的局部性归纳偏置，ViT 需要**更多数据**才能达到与 CNN 相当的性能。ViT-B/16 在 ImageNet-21k 上预训练后才能在 ImageNet-1k 上超过 ResNet-50。这一数据效率问题催生了混合架构（如 Swin Transformer、ConvNeXt），试图结合 CNN 的局部性和 Transformer 的全局性。第7章将讨论这些混合架构在多模态 LLM 中的应用。

---

## 3.4 世界动态的表征：RSSM 与潜在状态空间

### 3.4.1 为什么需要状态压缩？

在强化学习中，Agent 的观测通常是高维的（如 84×84×3 的 Atari 游戏帧，或 224×224×3 的机器人相机图像）。直接在原始像素上训练策略是无效率的——像素中的大多数信息（如纹理、光照变化）与决策无关，真正重要的是物体的位置、速度和关系。

World Model 的核心思想是：**学习一个压缩函数，将高维观测映射到低维潜在状态，然后在潜在状态上训练策略**。这类似于人类认知：我们不会在视网膜级别的像素上思考，而是在"物体-位置-关系"的抽象表征上思考。

### 3.4.2 RSSM 的数学结构

Dreamer 的 RSSM（Recurrent State Space Model）定义了一个潜在状态 z_t，它是确定性部分 h_t 和随机部分 s_t 的拼接：

h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
s_t ~ p(s_t | h_t)
z_t = [h_t; s_t]

其中 h_t 是确定性记忆（通过 GRU 或 LSTM 更新），s_t 是随机状态（通过变分推断学习）。观测 x_t 和奖励 r_t 的生成模型是：

x_t ~ p(x_t | z_t)
r_t ~ p(r_t | z_t)

RSSM 的关键创新是**在潜在空间中进行想象性推演**：给定当前状态 z_t，模型可以通过采样未来的动作 a_t, a_{t+1}, ... 来生成想象的轨迹 z_{t+1}, z_{t+2}, ...。这些想象的轨迹用于训练策略和价值函数，而不需要与真实环境交互。这使得 Dreamer 的样本效率比无模型 RL 高出数个数量级。

但 RSSM 的潜在状态是**纯数值的**——它是一个 32 维或 256 维的向量，不包含任何语义标签。这与 LLM 的文本嵌入形成鲜明对比：文本嵌入有天然的语义可解释性（如"国王"向量与"女王"向量的差接近"男性-女性"），而 RSSM 的潜在向量对人类是黑箱的。

---

## 3.5 跨模态对齐：CLIP 与对比学习的数学本质

### 3.5.1 对比学习的目标函数

CLIP（Contrastive Language-Image Pre-training）通过对比学习将文本和图像映射到同一潜在空间。给定一批图像-文本对 {(x_i, t_i)}，CLIP 的目标是最大化正样本对（匹配的图像和文本）的相似度，最小化负样本对（不匹配的图像和文本）的相似度：

L = -1/N Σ_{i=1}^N [log(exp(sim(x_i, t_i)/τ) / Σ_j exp(sim(x_i, t_j)/τ)) + log(exp(sim(x_i, t_i)/τ) / Σ_j exp(sim(x_j, t_i)/τ))]

其中 sim(x, t) = (E_image(x)^T · E_text(t)) / (||E_image(x)|| · ||E_text(t)||) 是余弦相似度，τ 是温度参数。

这个对称的目标函数迫使图像编码器和文本编码器学习"对齐的"表征：如果图像 x_i 和文本 t_i 描述同一内容，它们的嵌入向量在潜在空间中应该彼此靠近；如果不匹配，应该彼此远离。

### 3.5.2 对比学习的隐性假设

对比学习有一个隐性假设：**语义信息可以通过"对比"来提取**。具体来说，模型通过区分"这是什么"和"这不是什么"来学习内容。这一假设的局限性在于：对比学习只编码了**相对关系**（A 比 B 更接近 C），而没有直接编码**预测关系**（如果 A 发生，B 会怎样）。这恰恰是 JEPA 试图解决的问题。

---

## 3.6 JEPA：从"对比什么"到"预测什么"

### 3.6.1 预测性表征 vs 对比性表征

LeCun 在提出 JEPA 时指出，对比学习（如 CLIP）和生成学习（如 VAE）都不是表征学习的终极目标。对比学习的问题是：它只学习判别边界，不学习数据的内在结构。生成学习的问题是：它强迫模型重建输入的每个像素，这浪费了大量计算在无关紧要的细节上（如纹理、光照）。

JEPA 的目标函数是：**预测被掩码部分的表征，而不是重建被掩码部分的像素**。具体来说，给定图像 x，将其分为可见部分 x_v 和掩码部分 x_m。JEPA 的编码器 E 将 x_v 编码为表征 s_v，然后预测器 P 根据 s_v 预测掩码部分的表征：

ŝ_m = P(s_v, context)

目标是最小化预测表征 ŝ_m 与目标表征 s_m（由目标编码器 E_target 对 x_m 编码得到）之间的距离：

L = ||ŝ_m - s_m||²

注意：这里的目标不是让 ŝ_m 等于 x_m 的像素值，而是让 ŝ_m 等于 x_m 的**表征**。这意味着 JEPA 不需要生成像素，只需要在表征空间中预测"掩码部分的内容应该由什么向量表示"。

### 3.6.2 JEPA 的联合嵌入架构

JEPA 的另一个关键设计是**联合嵌入**（Joint Embedding）：预测器 P 和编码器 E 共享同一个潜在空间，而不是各自学习不同的空间。这与早期的方法（如 Cross-View Coding）不同，后者让不同视角的表征在各自的空间中学习，然后通过学习映射来对齐。联合嵌入保证了预测和目标从一开始就处于同一空间，避免了映射误差。

JEPA 的预测性目标还有一个深层优势：它天然编码了**因果结构**。如果模型能够预测"在状态下 A，下一步的表征是什么"，那么它实际上学习了一个隐式的动力学模型。这正是 World Model 所需要的。第13章将深入讨论 JEPA 在视频世界建模中的应用。

---

## 3.7 代码：双塔 CLIP 风格对比学习框架

下面的代码实现了一个简化的双塔 CLIP 模型，展示如何将文本和图像映射到共享的潜在空间。这个实现可以直接扩展到完整的 CLIP 训练。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImageEncoder(nn.Module):
    """
    简化版图像编码器：使用小型 CNN 提取特征，然后投影到共享空间。
    真实 CLIP 使用 ViT 或 ResNet，这里用轻量 CNN 以便教学。
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Linear(128, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, H, W]
        features = self.conv(x).flatten(1)  # [batch, 128]
        embed = self.projection(features)     # [batch, embed_dim]
        embed = F.normalize(self.norm(embed), dim=-1)  # L2 归一化
        return embed

class TextEncoder(nn.Module):
    """
    简化版文本编码器：使用可学习嵌入 + Transformer，投影到共享空间。
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 d_model: int = 128, num_heads: int = 4, max_len: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 2, batch_first=True),
            num_layers=2
        )
        self.projection = nn.Linear(d_model, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [batch, seq_len]
        seq_len = x.size(1)
        h = self.token_embed(x) + self.pos_embed[:, :seq_len, :]
        
        if mask is not None:
            # mask: True 表示有效位置
            mask = ~mask  # Transformer 使用 True 表示 mask
        
        h = self.transformer(h, src_key_padding_mask=mask)
        # 使用 [CLS] 位置（第一个 token）的表征
        features = h[:, 0, :]  # [batch, d_model]
        embed = self.projection(features)  # [batch, embed_dim]
        embed = F.normalize(self.norm(embed), dim=-1)
        return embed

class CLIPModel(nn.Module):
    """
    双塔 CLIP 模型：图像编码器 + 文本编码器 + 对比损失。
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
    
    def forward(self, images: torch.Tensor, text: torch.Tensor, 
                text_mask: torch.Tensor = None) -> dict:
        """
        images: [batch, 3, H, W]
        text: [batch, seq_len]
        text_mask: [batch, seq_len], True 表示有效 token
        """
        image_embeds = self.image_encoder(images)  # [batch, embed_dim]
        text_embeds = self.text_encoder(text, text_mask)  # [batch, embed_dim]
        
        # 计算相似度矩阵
        logits = torch.matmul(image_embeds, text_embeds.T)  # [batch, batch]
        logits = logits * torch.exp(self.temperature)
        
        # 对比损失（对称）
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)      # 图像→文本
        loss_t2i = F.cross_entropy(logits.T, labels)    # 文本→图像
        loss = (loss_i2t + loss_t2i) / 2
        
        return {
            'loss': loss,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits': logits,
            'temperature': torch.exp(self.temperature).item()
        }

# ============ 演示 ============
if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size = 4
    vocab_size = 1000
    embed_dim = 64
    
    # 随机数据
    images = torch.randn(batch_size, 3, 64, 64)
    text = torch.randint(0, vocab_size, (batch_size, 16))
    text_mask = torch.ones(batch_size, 16).bool()
    
    # 模型
    model = CLIPModel(vocab_size, embed_dim)
    
    # 前向传播
    output = model(images, text, text_mask)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Image embeds shape: {output['image_embeds'].shape}")
    print(f"Text embeds shape: {output['text_embeds'].shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Learned temperature: {output['temperature']:.4f}")
    
    # 验证跨模态检索：用第一个图像查询最相似的文本
    similarities = torch.matmul(output['image_embeds'][0:1], output['text_embeds'].T)
    best_match = similarities.argmax(dim=-1).item()
    print(f"Best text match for image 0: text {best_match}")
    
    # 训练一步
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    output['loss'].backward()
    optimizer.step()
    print("Training step completed.")
```

### 代码解析

这个 CLIP 实现包含几个关键教学点：

1. **L2 归一化**：在投影到共享空间后，对嵌入进行 L2 归一化。这是对比学习的标准做法，确保相似度计算不受向量模长影响。
2. **可学习温度**：温度参数 τ 控制分布的锐度。τ 太小（分布太尖锐）时训练不稳定，τ 太大（分布太平滑）时对比信号弱。将 τ 设为可学习参数（初始化为 0.07）让模型自动找到最优锐度。
3. **对称损失**：同时计算图像→文本和文本→图像的交叉熵，确保两个编码器相互对齐。
4. **轻量架构**：图像编码器使用小型 CNN 而非 ResNet/ViT，文本编码器使用 2 层 Transformer，这使得代码可以在 CPU 上快速运行，适合教学实验。

---

## 3.8 工程实践框：表征学习的训练与部署要点

### 常见陷阱

1. **表征崩溃（Representation Collapse）**：对比学习中最常见的问题是模型将所有样本映射到同一个向量，导致对比损失为零但表征毫无信息量。解决方案：使用 stop-gradient（如 BYOL、SimSiam）、添加预测器（predictor）或使用非对称架构（如 CLIP 的双塔）。
2. **批次大小依赖**：对比学习的性能严重依赖批次大小——批次越大，负样本越多，对比信号越强。在单卡 GPU 上训练 CLIP 时，batch size 通常只有 32-64，效果远不及大规模分布式训练（batch size 32768）。解决方案：使用记忆库（memory bank）或动量编码器（momentum encoder）来模拟大批次。
3. **模态不平衡**：在多模态训练中，某一模态（通常是文本）可能主导损失，导致另一模态的表征学习不足。解决方案：使用模态特定的学习率、梯度裁剪或平衡采样策略。

### 硬件需求估算

- **训练简化 CLIP**（如上面的代码，batch size 64，图像 64×64）：单张 RTX 4090 可训练，显存占用约 4-6 GB；
- **训练完整 CLIP**（ViT-B/32，batch size 32768）：需要 256-512 张 V100/A100，分布式训练数天。OpenAI 的原始 CLIP 使用了 256 张 V100 训练 12 天；
- **训练 RSSM World Model**（Atari 环境，图像 64×64）：单张 RTX 4090 可处理，显存占用约 8-12 GB，训练时间数小时到一天；
- **推理跨模态模型**：将 CLIP 用于图像检索时，需要预先计算所有图像的嵌入并存储在向量数据库中（如 FAISS、Milvus）。百万级图像的嵌入存储需要约 2-4 GB 内存（128 维 float32 向量）。

### 超参选择建议

- **温度参数 τ**：通常初始化为 0.07-0.1。如果训练初期 loss 下降过快但检索性能差，说明 τ 太小（分布太尖锐），应增大 τ；如果 loss 一直很高，说明 τ 太大，应减小 τ。
- **嵌入维度**：跨模态对齐的嵌入维度通常在 128-512 之间。维度太小（<64）无法编码足够信息，维度太大（>1024）增加计算开销但收益递减。
- **投影头**：在编码器后添加一个非线性投影头（如 MLP）通常能提升对比学习性能，但在下游任务中使用表征时应该去掉投影头，使用编码器的原始输出（SimCLR 的实验发现）。
- **数据增强**：图像侧的数据增强（随机裁剪、颜色抖动、高斯模糊）是 CLIP 训练的关键。文本侧的数据增强通常不需要，因为文本的语义稳定性高于图像。

---

## 3.9 本章小结

本章揭示了文本、视觉和物理世界表征的共享数学结构：它们都是将高维观测映射到低维潜在空间的**压缩函数**，而压缩的质量取决于目标函数的设计。Word2Vec 通过局部共现预测压缩词汇语义，BERT 通过全局掩码预测压缩上下文语义，CLIP 通过跨模态对比压缩对齐语义，RSSM 通过状态转移预测压缩物理动力学，JEPA 通过表征预测压缩视觉结构。

这些技术的共性可以概括为三个层次：
1. **描述性表征**（Word2Vec, CNN）：学习"这是什么"；
2. **上下文性表征**（BERT, ViT）：学习"这在当前语境中是什么"；
3. **预测性表征**（RSSM, JEPA）：学习"如果发生什么，这将会变成什么"。

从描述到上下文到预测，表征学习的能力逐步增强。第4章将讨论如何在大规模上预训练这些表征（Scaling Laws），第7章将展示如何将文本和视觉表征对齐到统一空间（多模态 LLM），第14章将展示如何用 World Model 的预测性表征增强 Agent 的因果推理。

---

## 延伸阅读

1. [Guu et al., 2020] **Retrieval augmented language model pre-training** —— 展示了如何通过检索外部知识来增强语言模型的表征，是 RAG 技术的理论基础。本章对表征外部化的讨论参考了该文的方法论。
2. [Lin et al., 2024] **VILA: On pre-training for visual language models** —— 系统分析了视觉语言模型的预训练策略，对比了交错式（interleaved）和对比式（contrastive）预训练的优劣。本章 3.5 节的跨模态对齐分析参考了该文的实验结论。
3. [Yin et al., 2024] **A survey on multimodal large language models** —— 最全面的多模态 LLM 综述，覆盖了视觉-语言、音频-语言、视频-语言等方向。本章对视觉表征的演化分析参考了该文的架构分类。
4. [Huang et al., 2025] **Vision-R1: Incentivizing reasoning in multimodal LLMs** —— 展示了如何在多模态模型中诱导推理能力，通过强化学习（GRPO）提升视觉问答的准确率。本章对表征与推理关系的讨论与该文的技术路线直接相关。
5. [Zhao et al., 2026] **Edge general intelligence through world models and agentic AI** —— 提出了世界模型与 Agent AI 在边缘设备上融合的技术路线，展示了表征统一性在工程落地中的具体方案。本章的"表征统一性"方法论直接受到该文的启发。

---

> **交叉引用**：本章的 CLIP 实现代码（3.7 节）将在第7章被扩展为完整的视觉语言模型。第2章的 Transformer 模块是本章文本编码器的底层组件。第12章的 RSSM 实现是本章 3.4 节理论分析的直接延续。第13章的 JEPA 深入讨论将基于本章 3.6 节的预测性表征框架展开。第14章的三者融合将综合运用本章的所有表征技术。
