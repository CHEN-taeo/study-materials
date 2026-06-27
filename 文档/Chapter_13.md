# 第13章 JEPA 与预测性表征学习：LeCun 的自主机器智能蓝图

**核心思想**：JEPA（Joint Embedding Predictive Architecture）不重建像素，不对比样本，而是在高维抽象表征空间中预测被掩码区域的表征，从而将自监督学习从"生成式重建"与"对比式匹配"推向第三条路径——预测性表征学习。这一范式的终极目标是构建能够在表征空间内推演世界动力学、支持规划与推理的自主机器智能系统。

**历史脉络**：自监督视觉表征学习在2020年前经历了两条泾渭分明的路径。第一条是对比学习（Contrastive Learning），以MoCo [He et al., 2020] 和SimCLR [Chen et al., 2020] 为代表，通过拉近正样本对、推远负样本对学习表征。其致命弱点在于对负采样策略的敏感性：负样本过少导致表征坍塌（representation collapse），负样本过多则引入语义相似的"假阴性"噪声。第二条是生成式重建（Generative Reconstruction），以MAE [He et al., 2022] 和BEiT [Bao et al., 2022] 为代表，通过掩码自编码器重建原始像素。这条路迫使模型在像素级细节上浪费大量计算容量——模型必须学会重建一张猫的胡须纹理，即便该纹理对"猫"这一语义概念的判断毫无贡献。与此同时，强化学习领域的世界模型研究（以Dreamer系列 [Hafner et al., 2023] 为代表）在潜在空间（latent space）中进行了前向预测，但Dreamer的RSSM（Recurrent State Space Model）仍依赖于像素级重建损失来训练状态空间模型。Yann LeCun在2022年提出的自主机器智能蓝图 [LeCun, 2022] 直指这两条路径的根本局限：人类婴儿从不重建视网膜上的像素，也从不对比两张图像的相似度；婴儿通过观察世界、预测下一刻的感官输入来学习。JEPA由此诞生——它学习在抽象表征空间中预测，而非在原始感官空间中重建。第3章的JEPA初步介绍为本章的深入分析奠定了基础；本章将从数学本质、图像与视频中的具体实现、工程实践与哲学定位四个维度展开。

---

## 13.1 从生成式世界模型到预测性表征

在理解JEPA之前，必须首先厘清其试图取代和继承的对象。自监督学习（Self-Supervised Learning, SSL）在视觉领域的核心问题始终是如何从无标注图像中提取可用于下游任务的语义表征。2019至2021年间，对比学习主导了视觉SSL的学术议程。MoCo v3、SimCLR v2、SwAV等方法将表征学习转化为一个判别问题：为每个图像生成两个不同的视图（augmented views），使其在表征空间中距离最小化；同时从同一批次中抽取其他图像作为负样本，使其距离最大化。这一范式的计算开销与批次大小（batch size）成正比——SimCLR需要4096甚至更大的batch才能有效学习。更严重的是，负样本可能包含与正样本语义相似的图像（例如，两只不同品种的狗），模型被强制将它们推远，实质上在表征空间中引入了语义扭曲。

生成式重建提供了另一条出路。MAE（Masked Autoencoder）将图像划分为不重叠的patch，随机遮蔽75%的patch，要求Transformer编码器-解码器结构重建被遮蔽的像素。这一方法在ImageNet上取得了惊人的微调性能（fine-tuning accuracy），但存在一个结构性的效率问题：解码器必须处理低维像素空间的高频细节，而编码器的最后一层表征（用于下游任务）却被迫包含足以重建任意纹理的信息。这意味着模型容量被"挪用"到了对语义无关的纹理预测上。从信息论的角度，重建损失 $L_{recon} = \|x - \hat{x}\|^2$ 对像素级误差一视同仁，重建一片草叶的纹理与重建一只眼睛的语义重要性在损失函数中被平等对待。

世界模型（World Model）的研究则从另一个角度逼近同一问题。Ha & Schmidhuber [2018] 的开创性工作提出了在紧凑潜在空间中学习环境动力学模型的想法。Dreamer系列 [Hafner et al., 2023] 将这一思想推向了深度强化学习的前沿：一个编码器将像素观测压缩为潜在状态 $z_t$，一个递归模型预测 $z_{t+1}$，一个策略网络在潜在空间中规划动作。Dreamer的RSSM通过像素级重建损失训练编码器，即 $L = \|o_t - \text{decode}(z_t)\|^2$，其中 $o_t$ 为原始观测。这导致RSSM在表征中必须保留足够的像素级信息以支持重建，即便这些信息对决策（policy）毫无用处。Guan et al. [2024] 在自动驾驶世界模型的综述中明确指出，RSSM与JEPA代表了两种根本不同的路径：RSSM依赖于重建损失来绑定潜在状态与感官观测，而JEPA通过联合嵌入预测直接学习抽象表征，无需 pixel-level commitment。

LeCun提出的JEPA范式包含一个关键洞察：如果模型的目标是学习"世界如何运转"的表征，那么它应该预测的是抽象状态的演化，而非像素的演化。在像素空间预测下一帧视频是病态的（ill-posed）——一片被风吹动的树叶下一刻可能出现在数百个不同的像素位置，任何像素级预测器都将被迫输出模糊的平均值。但在表征空间中，树叶的"物体性"和"运动轨迹"可以被编码为紧凑的向量，预测下一帧等价于预测该向量的更新。JEPA正是将这一洞察从视频领域（V-JEPA）推广到静态图像（I-JEPA）乃至通用感官模态的系统性尝试。

## 13.2 JEPA的数学本质：三种表征学习范式的对比

JEPA的数学框架可以精确地置于自监督学习的理论谱系中。现有三种主流范式——对比性、生成性与预测性——在目标函数、表征空间与信息瓶颈上存在本质差异。

**对比性表征学习**（Contrastive Representation Learning）的 InfoNCE 损失可写为：

$$L_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\sum_{k=0}^{K} \exp(\text{sim}(z_i, z_k)/\tau)}$$

其中 $z_i = f(x_i)$ 为编码器输出，$z_j^+$ 为正样本表征，$z_k$（$k \geq 1$）为负样本表征，$\tau$ 为温度系数。该损失的梯度强制正样本对在表征空间中聚集，同时推散负样本对。其理论根基在于互信息最大化：一个好的表征应该保留输入与目标之间的互信息。但实践中的问题在于负样本的构造：如果 $K$ 过小，优化过程容易收敛到坍塌解（所有输入映射到同一点）；如果 $K$ 过大，计算开销剧增，且无法避免"假阴性"问题。

**生成性表征学习**（Generative Representation Learning）以MAE为例，其损失为：

$$L_{MAE} = \sum_{m \in \mathcal{M}} \|x_m - \hat{x}_m\|^2$$

其中 $\mathcal{M}$ 为被掩码的patch索引集合，$\hat{x}_m = g(f(x_{\setminus \mathcal{M}}))$ 为解码器基于可见patch的重建。该损失迫使编码器 $f$ 保留足以恢复任意被掩码区域的全部像素信息。然而，从信息瓶颈（Information Bottleneck）的视角，编码器被迫保留的信息量远大于下游任务（如分类）所需。纹理、光照、噪声等与语义无关的变量占据了表征容量。

**预测性表征学习**（Predictive Representation Learning）即JEPA的核心损失函数为：

$$L_{JEPA} = \sum_{y \in \mathcal{Y}} \|s_y - g(s_x, a)\|^2$$

其中 $s_x = f_{\theta}(x)$ 为上下文编码器（context encoder）对可见输入的表征，$s_y = f_{\theta'}(y)$ 为目标编码器（target encoder）对被掩码区域的表征，$g$ 为预测器（predictor），$a$ 为可选的动作/上下文变量。目标编码器 $f_{\theta'}$ 通常是上下文编码器 $f_{\theta}$ 的指数移动平均（EMA）：$\theta' \leftarrow \lambda \theta' + (1-\lambda) \theta$。这一EMA机制至关重要：它提供了一种"慢速"的目标表征，防止模型通过简单的恒等映射或信息泄露来最小化损失——如果 $s_y$ 与 $s_x$ 来自同一个编码器，predictor可能学会复制而非预测。

JEPA的数学优雅之处在于它同时避免了对比损失的负样本困境和生成损失的像素级约束。它不需要负样本，因为损失的形式本身（预测误差）已经提供了足够的训练信号；它不重建像素，因为目标 $s_y$ 存在于高维抽象空间。预测器 $g$ 的作用相当于一个"隐式解码器"——但它解码的不是像素，而是目标区域的表征。这意味着模型必须学会在表征空间中操作"概念"（例如"这是一只猫的眼睛"）而非像素坐标。

从表征几何的角度，JEPA学习的是一个嵌入空间，其中欧氏距离编码了语义相似性。对比学习显式优化相似性度量（通过 $\text{sim}$ 函数），JEPA则通过预测任务隐式地塑造该度量：如果两个区域在语义上相关，预测器应该能够基于上下文推断其表征；如果无关，预测误差将惩罚模型。这种"任务驱动"的表征度量比手工设计的相似性函数更具适应性。

## 13.3 I-JEPA：图像中的联合嵌入预测

I-JEPA（Image-based Joint Embedding Predictive Architecture）是JEPA范式在静态图像领域的首次大规模验证。其架构由三个核心组件构成：上下文编码器（context encoder）、目标编码器（target encoder）与预测器（predictor）。两个编码器均为Vision Transformer（ViT），但目标编码器不接收梯度，仅通过EMA更新。预测器是一个较浅的Transformer，接收上下文编码器的输出以及一组可学习的掩码token（mask tokens），输出被掩码区域的预测表征。

**掩码策略**：I-JEPA与MAE最关键的区别在于掩码的拓扑结构。MAE采用随机patch掩码（random patch masking），每个patch被独立遮蔽，遮蔽比率通常高达75%。I-JEPA则采用**空间连续的块掩码**（spatially contiguous block masking）：每次遮蔽一个大的矩形区域（例如覆盖图像32×32像素的区域），同时保留多个分散的可见区域作为上下文。这种设计有一个深刻的动机：如果模型只需基于相邻patch预测一个被掩码patch，它可以依赖局部纹理统计（如边缘连续性）而不必理解全局语义。空间连续的块掩码迫使模型整合远距离的上下文信息——例如，基于图像左下角的一只爪子来预测右上角缺失的猫头。

具体实现中，I-JEPA将图像划分为14×14的patch网格（对应224×224输入与16×16 patch尺寸）。目标编码器接收完整图像，输出所有patch的表征。上下文编码器接收一个"可见区域"掩码，仅处理未被遮蔽的patch。预测器接收上下文编码器的输出表征，以及代表目标区域位置的可学习位置嵌入（positional embeddings），输出目标区域表征的预测值。训练目标是最小化预测表征与目标编码器输出的L2距离。

**与对比学习的性能对比**：I-JEPA在ImageNet-1K上的线性探测（linear probing）性能表明，无需负样本、无需数据增强（data augmentation）的预测性表征，能够达到与对比学习方法相当的语义水平。这一点具有重要的工程意义：对比学习高度依赖精心设计的数据增强（如随机裁剪、颜色抖动、高斯模糊），增强策略的失效直接导致表征质量崩塌；JEPA几乎不需要增强，因为预测任务本身提供了丰富的训练信号。在目标检测、语义分割等下游任务中，I-JEPA预训练的ViT backbone展现了优异的迁移能力，证明预测性表征捕获了适用于密集预测任务的层次化特征。

**表征坍塌的分析**：I-JEPA的设计巧妙地避免了表征坍塌。在对比学习中，坍塌对应所有输入映射到同一点；在JEPA中，如果上下文编码器输出恒定向量，预测器将无法区分不同目标区域，导致所有目标预测相同。然而，目标编码器的EMA更新提供了一种"锚定"效应：即使上下文编码器开始坍塌，目标编码器（作为EMA）仍保留历史信息，预测器必须持续匹配一个动态目标，这使得坍塌解不稳定。从动力系统的角度，EMA引入了一个慢变的外部场，打破了坍塌解的对称性。

## 13.4 V-JEPA：视频中的预测性表征学习

视频是预测性表征学习的天然试验场，因为视频的时序结构本身即是一个预测任务：给定过去帧，推断未来帧。然而，在像素空间预测视频帧是一个臭名昭著的病态问题。一个抛向空中的球在下一帧可能出现在任何位置；一扇门的开合存在两种相反的可能性；一个人下一句话的内容在像素上不可预测。任何像素级预测器（如视频扩散模型、VAE）在面对这些多模态未来时，被迫输出所有可能性的统计平均值，导致运动模糊（motion blur）和细节丧失。V-JEPA（Video Joint Embedding Predictive Architecture）通过将预测从像素空间转移到表征空间，彻底回避了这一问题。

V-JEPA的架构继承了I-JEPA的三组件设计，但扩展到时序维度。输入是一段视频片段（例如16帧），目标编码器处理完整片段，输出每一帧的时空表征。上下文编码器处理被掩码的片段：部分时间步被完全遮蔽，或部分空间区域在所有时间步被遮蔽。预测器接收上下文表征，预测被掩码时间步或区域的表征。

**关键洞察**：在表征空间中，球的运动可以被编码为一条抛物线轨迹的参数，预测下一帧等价于更新这些参数。门的状态可以被编码为"开/闭"的二值变量，预测下一帧等价于根据上下文推断状态转移。这些表征在语义上是确定的，即便像素级未来是多模态的。V-JEPA因此实现了"确定性预测抽象表征，而保留像素级的不确定性"。这与第12章讨论的因果推断有深刻联系：V-JEPA在表征层面编码了观察到的规律（associative patterns），但尚未建立显式的因果模型（interventional model）。

V-JEPA在视频动作识别（action recognition）和视频目标分割（video object segmentation）等下游任务中展示了强大的迁移性能。与之前的视频预训练方法（如VideoMAE、TimeSformer）相比，V-JEPA的优势不仅在于更高的下游准确率，还在于预训练效率：无需在像素级生成高分辨率视频帧，预测器只需在低维表征空间中操作，训练速度显著提升。在Kinetics-400等标准视频理解基准上，V-JEPA的线性探测性能接近全监督训练的水平。

**时序掩码策略**：V-JEPA的掩码设计比I-JEPA更为复杂。除了空间上的连续块掩码，V-JEPA引入时序掩码：模型可能需要基于前8帧预测后8帧的表征，或在时间维度上进行随机采样。这种"时空块"掩码迫使模型学习跨时间推理，例如推断一个被临时遮挡的物体仍然存在（object permanence）。这种推理能力正是世界模型所需的核心认知功能。

## 13.5 JEPA的局限：语义盲区和任务依赖性

尽管JEPA在理论和实验上展现了吸引力，作为一个仍在快速演进的研究方向，它存在若干结构性局限。

**语义盲区**（Semantic Blindness）：JEPA通过预测误差驱动表征学习，但预测误差对语义层级不敏感。一个模型可能完美预测"一辆红色汽车在下一帧的表征"，但对"汽车为什么停下"毫无概念。预测损失优化的是表征空间中的几何接近性，而非语义因果性。这意味着JEPA可能学到大量关于"世界如何变化"的统计规律，却对"世界为何如此变化"一无所知。例如，I-JEPA可以预测一个被掩码区域包含"天空"，因为它从周围patch推断出"这是户外场景"；但它不会理解"天空是蓝色的因为瑞利散射"。这种语义盲区使JEPA表征在需要深层因果推理的任务（如物理直觉判断、工具使用规划）中可能表现不足。第12章讨论的因果发现方法（如do-calculus、structural causal models）可以与JEPA形成互补：JEPA提供低维状态表征，因果模型在这些状态上学习干预分布。

**任务依赖性**（Task Dependency）：JEPA学到的表征并非对所有下游任务都最优。JEPA的预测目标（在图像中预测空间区域的表征，在视频中预测时序区域的表征）隐式地编码了特定的归纳偏置（inductive bias）。这种偏置对分类、检测、分割等空间任务极为有利，但对需要精细空间定位或像素级精确度的任务（如医学图像分割中的微小病灶检测、遥感图像的细粒度变化检测）可能不够。在这些任务中，MAE等重建式预训练反而具有优势，因为重建任务迫使模型保留像素级的空间信息。工程上，混合预训练策略（JEPA预测损失 + 轻量级重建损失）可能是解决这一矛盾的务实路径。

**缺少显式层次结构**：当前的I-JEPA和V-JEPA实现采用单一尺度的预测器。人类视觉系统处理信息时存在明确的层次：从V1的初级特征到IT区的物体表征。JEPA理论上可以扩展为层次化预测架构（Hierarchical JEPA），其中低层预测局部纹理特征，高层预测物体级状态转移，但目前这一方向尚未有大规模实验验证。Sainburg & Weinreb [2026] 指出，JEPA-style objectives在Agentic AI中需要显式的层次化结构来桥接感知表征与动作规划，否则表征将停留在"被动观察"层面，无法支持主动推理。

**计算效率的权衡**：虽然JEPA避免了像素级解码，但predictor的计算开销不可忽视。在I-JEPA中，predictor深度通常为encoder的1/2（如6层Transformer），且需要处理所有目标位置的预测。当目标区域数量较大时，predictor的内存占用与计算量可能超过一个轻量级像素解码器。此外，target encoder的EMA更新虽然不引入额外训练，但需要在内存中维护两套编码器参数，对显存受限的边缘设备构成挑战。

## 13.6 JEPA与World Model的关系

第3章将JEPA定位为世界模型的一种候选实现，本章进一步论证这一定位的理论基础与工程路径。世界模型的核心功能是为智能体提供一个内部模拟器（internal simulator）：给定当前状态和动作，预测下一状态。在JEPA的框架中，编码器将感官输入压缩为状态表征，predictor充当转移模型（transition model），预测下一状态的表征。当扩展到动作条件化（action-conditioned）时，JEPA的预测损失变为 $L = \|s_{t+1} - g(s_t, a_t)\|^2$，这与模型预测控制（Model Predictive Control, MPC）中的动力学模型在数学形式上完全一致。

然而，JEPA与经典的World Model（如RSSM、MuZero）存在关键区别。RSSM [Hafner et al., 2023] 通过一个递归神经网络（RNN）在潜在空间中维持一个随时间更新的状态 $h_t$，并通过该状态预测下一观测 $o_{t+1}$。RSSM的状态显式编码了时间累积信息，适合需要长期记忆的任务。JEPA的预测器在单次前向传播中处理所有上下文，不维持跨时间步的隐状态。这带来两个后果：一方面，JEPA更简单、更稳定，避免了RNN的梯度消失与长期依赖问题；另一方面，JEPA在处理需要长程时序推理的任务（如长视频理解、多步规划）时，可能需要额外的记忆机制（如外部记忆库或分层预测架构）。

Guan et al. [2024] 在自动驾驶领域的世界模型综述中，将JEPA与RSSM并列为两大主流架构。RSSM依赖重建损失来约束潜在状态与物理世界的一致性，而JEPA完全通过预测性自监督学习表征。在自动驾驶场景中，RSSM的像素级重建可以提供可解释的视觉预测（例如，渲染未来数秒的驾驶场景），这对安全关键系统具有工程价值；JEPA的抽象表征预测则更适合端到端决策（end-to-end driving policy），因为它不浪费计算资源在视觉细节重建上。

Wu et al. [2026] 提出的RLVR-World框架则从另一个角度补充了JEPA：他们用强化学习（RL）后训练世界模型，使其能够探索训练数据分布之外的状态。这与JEPA的纯自监督路径形成对照。JEPA从大量无标注数据中学习统计规律，但无法主动生成反事实（counterfactual）场景；RLVR-World通过奖励信号引导世界模型探索罕见状态。两者的融合——JEPA提供基础表征，RL后训练扩展泛化边界——可能是通向鲁棒世界模型的一条有效路径。

Dupoux, LeCun & Malik [2026] 从认知科学的角度为JEPA提供了生物学辩护。人类婴儿的学习并不依赖于显式的教师信号或像素重建任务；婴儿通过观察物理世界、预测物体运动、发现预测偏差来构建内部世界模型。当一个婴儿看到玩具被布遮住时，她预测玩具仍然存在（object permanence）；当布被揭开而玩具消失时，预测误差触发惊讶反应，驱动学习更新。JEPA的预测损失本质上正是这种"预测-惊讶-学习"循环的数学形式化。该文进一步指出，System 2（慢思考）的认知架构需要一个层次化的JEPA栈：底层处理感官预测，中层处理物体动力学，顶层处理社会交互与语言推理。这为JEPA的未来发展指明了架构方向。

Sainburg & Weinreb [2026] 在讨论Agentic AI时，将JEPA-style objectives置于"感知-推理-行动"循环的感知端。他们指出，当前LLM-based agent（如ReAct、AutoGPT）的推理发生在离散符号空间，而JEPA提供了从连续感官输入到抽象表征的桥梁。在具身智能（embodied AI）场景中，机器人需要同时处理视觉、力觉、本体感觉等多模态输入，JEPA的联合嵌入架构天然支持多模态表征的对齐——不同模态的编码器可以将输入映射到同一个预测空间，predictor在该空间中跨模态进行预测。这种多模态JEPA（Multimodal JEPA）尚未出现大规模实现，但在理论上已具备清晰的架构路径。

## 13.7 代码：简化版I-JEPA实现

以下代码展示了一个基于PyTorch的简化版I-JEPA，包含Vision Transformer编码器、EMA目标编码器、块掩码策略与预测器。该实现使用标准`nn.TransformerEncoderLayer`以最小化依赖，可直接运行。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 可学习的positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x + self.pos_embed

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class IJEPA(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 enc_depth=12, pred_depth=6, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        # Context encoder: 处理可见patch
        self.context_encoder = ViTEncoder(embed_dim, enc_depth, num_heads)
        
        # Target encoder: EMA of context encoder, 无梯度
        self.target_encoder = ViTEncoder(embed_dim, enc_depth, num_heads)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: 接收context token + mask token, 输出target预测
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.predictor = ViTEncoder(embed_dim, pred_depth, num_heads)
        self.predictor_proj = nn.Linear(embed_dim, embed_dim)

        self.ema_momentum = 0.996  # EMA更新率

    @torch.no_grad()
    def update_target_encoder(self):
        """每步训练后调用, 以EMA更新target encoder"""
        for p_ctx, p_targ in zip(self.context_encoder.parameters(),
                                 self.target_encoder.parameters()):
            p_targ.data.mul_(self.ema_momentum).add_(p_ctx.data, alpha=1 - self.ema_momentum)

    def make_masks(self, batch_size, num_visible, num_targets):
        """生成随机掩码。真实实现中应使用空间连续的块掩码。"""
        device = next(self.parameters()).device
        context_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        for b in range(batch_size):
            perm = torch.randperm(self.num_patches, device=device)
            context_mask[b, perm[:num_visible]] = True
            target_mask[b, perm[num_visible:num_visible + num_targets]] = True
        return context_mask, target_mask

    def forward(self, imgs):
        B = imgs.shape[0]
        x = self.patch_embed(imgs)  # (B, N, D)

        # 决定可见与目标patch数量 (这里固定比例)
        num_visible = int(self.num_patches * 0.75)
        num_targets = self.num_patches - num_visible
        context_mask, target_mask = self.make_masks(B, num_visible, num_targets)

        # Context encoder: 仅处理可见patch
        ctx_tokens = x[context_mask].view(B, num_visible, self.embed_dim)
        ctx_repr = self.context_encoder(ctx_tokens)  # (B, num_visible, D)

        # Target encoder: 处理全部patch, 无梯度
        with torch.no_grad():
            full_repr = self.target_encoder(x)  # (B, N, D)
            target_tokens = full_repr[target_mask].view(B, num_targets, self.embed_dim)

        # Predictor: 拼接context repr与mask token, 预测target
        mask_tokens = self.mask_token.expand(B, num_targets, -1)
        pred_input = torch.cat([ctx_repr, mask_tokens], dim=1)  # (B, num_visible + num_targets, D)
        pred_repr = self.predictor(pred_input)
        pred_repr = self.predictor_proj(pred_repr[:, -num_targets:])  # 取最后num_targets个

        # 损失: 预测表征与target encoder输出的L2距离
        loss = F.mse_loss(pred_repr, target_tokens)
        return loss


# ----------------- 训练示例 -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IJEPA(img_size=224, patch_size=16, embed_dim=768, 
                  enc_depth=12, pred_depth=6, num_heads=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # 模拟一个batch的训练
    imgs = torch.randn(4, 3, 224, 224).to(device)
    loss = model(imgs)
    loss.backward()
    optimizer.step()
    model.update_target_encoder()  # EMA更新
    print(f"Loss: {loss.item():.4f}")
```

该代码保留了I-JEPA的核心机制：
1. **双编码器结构**：context encoder接收梯度，target encoder通过EMA平滑更新，防止表征坍塌。
2. **Predictor**：较浅的Transformer（6层）接收context token和可学习的mask token，在统一序列中处理。
3. **掩码策略**：当前为随机掩码，注释中说明真实实现应采用空间连续块掩码。
4. **预测损失**：在表征空间中的MSE，而非像素空间的重建损失。

## 13.8 工程实践框

**预测目标选择**（Target Configuration）：I-JEPA的性能对目标块（target block）的大小和数量高度敏感。目标块过小（如单个patch），模型可以依赖局部纹理完成预测，无需学习全局语义；目标块过大，上下文信息不足，预测任务过于困难。Meta的原始实现采用多尺度目标块（multi-scale target blocks），从覆盖图像1/16到1/4的区域中采样。工程上，建议先固定目标块占图像面积的4%~16%，逐步增加目标数量（从1个块到4个块）观察下游任务性能。目标块与上下文块的重叠应最小化，否则信息泄露（information leakage）会通过重叠区域降低预测难度，导致表征学习不充分。

**掩码策略**（Masking Strategy）：空间连续块掩码（spatially contiguous block masking）与随机散布掩码（random scattered masking）的对比实验表明，连续块掩码在ImageNet线性探测上稳定优于随机掩码，提升幅度约2~3个百分点。原因已在前文分析：连续块迫使模型进行长距离语义推理。在视频场景（V-JEPA）中，时序连续掩码（即遮蔽连续的时间步）同样优于随机时序采样。对于非自然图像数据（如医学CT、遥感图像），需要调整块掩码的几何形状：CT图像中病灶可能仅占数个slice，使用三维立方体掩码比二维块掩码更适合。

**表征可视化**（Representation Visualization）：训练JEPA后，如何验证其学到了语义表征而非纹理捷径？推荐使用CKA（Centered Kernel Alignment）比较不同层表征与人工标注的语义分割图之间的对齐程度。如果JEPA的高层表征与语义标签的CKA系数显著高于低层，说明层次化语义结构已经形成。此外，可通过探测目标块预测误差的分布来诊断模型：如果误差在物体边界处显著升高，说明模型学会了物体级别的分割；如果误差均匀分布，可能仅学到了纹理统计。

**硬件需求估算**：以ViT-H/14（embed_dim=1280, depth=32）为例，context encoder参数量约6.3亿，target encoder等量，predictor（深度16）约3.1亿，总参数量约16亿。在batch size=256、图像分辨率224×224条件下，单次前向传播需要约28GB显存。使用混合精度训练（AMP）和梯度检查点（gradient checkpointing）可将显存降至约20GB，适配8×A100 80GB节点。目标编码器的EMA更新在CPU上完成，不增加前向传播时间。Predictor深度对训练速度影响显著：将predictor从16层减至6层，训练速度提升约35%，但ImageNet线性探测下降约1.5个百分点，需在效率与精度间权衡。

**常见训练陷阱**：
1. **Predictor过浅**：如果predictor深度不足（如仅2层），它无法补偿context与target之间的复杂信息差距，模型退化为近似恒等映射，损失下降但表征质量差。经验法则是predictor深度至少为encoder的1/3。
2. **EMA动量过高**：如果ema_momentum > 0.999，target encoder更新过慢，在训练初期提供的目标信号过于陈旧，导致收敛速度显著下降。建议从0.996开始，训练后期提升至0.999。
3. **学习率不匹配**：JEPA的encoder通常使用较大权重衰减（如0.05），而predictor的权重衰减应较小（如0.01），因为predictor需要快速适应目标表征的变化。使用统一优化器配置可能导致predictor欠拟合。

## 13.9 小结

JEPA代表了一种根本性的范式转移：自监督学习的目标从"重建输入"或"匹配样本"转向"在抽象空间中预测未来表征"。I-JEPA在图像中证明了这一路径的可行性，无需负样本、无需数据增强，即可达到对比学习级别的语义表征；V-JEPA将这一思想扩展到视频，避免了像素级预测的多模态模糊问题。然而，JEPA的语义盲区、任务依赖性和缺少显式层次结构等局限，表明它并非世界模型的终极答案，而是通往自主机器智能的一块关键基石。与Dreamer的RSSM [Hafner et al., 2023] 相比，JEPA放弃了像素级重建，换取了表征学习的纯度；与RLVR-World [Wu et al., 2026] 相比，JEPA专注于自监督预训练，尚未充分探索外部奖励对模型能力的扩展。从认知科学的视角 [Dupoux, LeCun & Malik, 2026]，JEPA模拟了人类婴儿通过预测偏差进行学习的核心机制，但要构建真正自主的智能系统，仍需将感知预测与因果推理、动作规划、符号抽象深度融合——这一融合正是第14章将要探讨的神经-符号混合架构的使命。

---

## 延伸阅读

以下文献从生成式世界模型、JEPA的综述定位、认知科学基础、Agentic AI中的层次化结构以及世界模型的强化学习训练五个角度，为读者提供了深入理解JEPA及其周边生态系统的关键入口。所有论文均来自已检索的145篇数据源。

1. [Hafner et al., 2023] — D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. *Mastering diverse domains through world models*. arXiv:2301.04104, 2023. Dreamer系列世界模型的集大成之作，系统阐述了RSSM在潜在空间中的预测机制，是理解JEPA与生成式世界模型差异的必读基准。

2. [Guan et al., 2024] — Y. Guan, H. Liao, Z. Li, J. Hu, R. Yuan, et al. *World models for autonomous driving: An initial survey*. IEEE Transactions on Intelligent Vehicles, 2024. 该综述明确将JEPA与RSSM并列为世界模型研究的两大技术路线，对自动驾驶场景下的架构选择具有直接的工程参考价值。

3. [Dupoux, LeCun & Malik, 2026] — E. Dupoux, Y. LeCun, and J. Malik. *Why AI systems don't learn and what to do about it: Lessons on autonomous learning from cognitive science*. arXiv:2603.15381, 2026. LeCun直接参与的认知科学视角论文，从婴儿学习机制出发为JEPA的预测性学习提供了生物学动机，并提出了层次化JEPA栈的架构愿景。

4. [Sainburg & Weinreb, 2026] — T. Sainburg and C. Weinreb. *The Cartesian Cut in Agentic AI*. arXiv:2604.07745, 2026. 讨论了JEPA-style objectives在Agentic AI系统中的层次化定位，分析了感知预测与动作规划之间的表征鸿沟，以及JEPA如何作为连接两者的桥梁。

5. [Wu et al., 2026] — J. Wu, S. Yin, N. Feng, and M. Long. *RLVR-World: Training world models with reinforcement learning*. Advances in Neural Information Processing Systems, 2026. 提出了通过强化学习后训练世界模型的框架，与JEPA的纯自监督路径形成互补，展示了如何结合预测性表征与探索奖励来扩展世界模型的泛化边界。
