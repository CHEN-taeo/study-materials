# 第15章 前沿与未来：具身智能、科学发现与AI for Robotics

**核心思想**：Agent、LLM与World Model的交汇点不在数字世界，而在物理世界——当Agent获得传感器与执行器，当World Model学会预测物理定律，人工智能才能真正从"理解世界"迈向"改变世界"。

**历史脉络**：1986年Brooks提出"无需表征的智能"（subsumption architecture），认为昆虫级别的行为无需内部世界模型；1990年代Dyna架构证明，若智能体拥有准确的world model，则可在想象中进行规划（Sutton, 1991）；2010年代，DeepMind的Atari agent（Mnih et al., 2015）首次展示了从像素到控制的端到端强化学习，但始终困于仿真环境。2018-2020年，Domain Randomization（Tobin et al., 2017）与Sim-to-Real transfer方法试图弥合仿真与现实的鸿沟，却依赖大量手工设计的随机化策略。2021年后，第12章的Dreamer系列（Hafner et al., 2020, 2023）将world model引入连续控制，MuZero（Schrittwieser et al., 2020）在围棋、国际象棋、Atari上证明了基于模型的规划可以超越无模型方法。2023-2024年，多模态大语言模型（MLLM）与World Model的结合催生了新一代具身智能系统：不再将感知、推理、行动割裂为三个独立模块，而是让语言模型直接生成运动策略，或用world model为语言模型提供物理约束。然而，纯文本LLM的替代方案——例如直接将机器人传感器数据映射到电机指令的端到端模仿学习——在小规模数据下表现尚可，但面对新环境、新物体时泛化失败，这证明了"无模型"路径的局限性，也解释了为何需要World Model作为中间表征。

---

## 15.1 从纯文本到物理世界的跨越

前三部分的教材已经构建了一个清晰的层次：第1-5章的LLM教会了机器如何理解语言；第6-10章的Agent架构让LLM获得了工具使用、记忆和规划能力；第11-14章的World Model（特别是第12章的Dreamer系列）赋予智能体在潜在空间中预测未来、进行反事实推理的能力。但所有这些能力，在数字世界（互联网、数据库、API）中运行得再好，终究只是"键盘上的智能"。

具身智能（Embodied AI）的核心命题是：智能的本质必须从与物理环境的交互中涌现。Rodney Brooks在1986年的经典论文中批判了当时AI研究对符号推理的迷恋，指出六足机器人不需要先构建环境的几何模型再规划步态——感知与运动之间的直接耦合（perception-action loop）本身就是智能。这一观点在2010年代被深度学习重新诠释：不是放弃表征，而是让表征从数据中学习。第12章的DreamerV3（Hafner et al., 2023）的RSSM架构在latent space中编码了图像序列的动态规律，本质上是让机器人"想象"自己行动的后果。但Dreamer训练在MuJoCo仿真中，现实中的机器人面对的问题是：仿真中的摩擦力、光照、布料形变与真实世界存在系统性偏差，这被称为**Sim-to-Real gap**。

跨越Sim-to-Real gap的尝试经历了三代范式。第一代是**Domain Randomization**（Tobin et al., 2017）：在仿真中随机化纹理、光照、相机位姿，训练出一个对变化鲁棒的控制器。这种方法在抓取任务上有效，但随机化的维度需要人工设计，过度随机化会使任务变得无解。第二代是**系统辨识与自适应控制**（Yu et al., 2017）：在真实环境中在线估计仿真参数，缩小模型误差。问题在于，许多物理参数（如物体表面的摩擦系数、关节间隙）难以从少量交互中准确估计。第三代是**Residual RL**（Johannink et al., 2019）：将仿真中学到的策略作为基础控制器，然后在现实中用RL微调一个残差策略，只修正基础策略的误差部分。这种方法的优势在于：残差策略的动作空间远小于完整策略，真实环境中的样本效率大幅提升。例如，Johannink et al.在X-Wing插孔任务中，仿真策略将插销对准孔洞，残差策略在真实环境中修正最后几毫米的接触力——这是Sim-to-Real中最困难的部分，因为接触动力学（contact dynamics）在仿真中几乎不可能精确建模。

与此同时，科学发现（AI for Science）代表了另一个方向。与机器人不同，科学Agent的"身体"不是机械臂，而是实验室设备（移液机器人、扫描电镜、化学反应釜）或计算仪器（分子动力学模拟器、有限元求解器）。2020年后，LLM被用于假设生成（Swanson, 2023）、实验设计（Burger et al., 2020）和文献综述（Boiko et al., 2023）。但纯文本LLM缺乏对物理约束的理解：它可能提出一个"理论上合理"的化学反应，却忽略了该反应在室温下需要数月才能完成。因此，将World Model引入科学Agent成为关键：在提出实验之前，先在分子动力学模拟器或有限元模型中"预演"实验结果。这正是第14章讨论的World Model在科学发现中的延伸——不是预测下一帧游戏画面，而是预测蛋白质折叠构象或材料应力分布。

---

## 15.2 具身智能：有物理身体的Agent

### 15.2.1 感知-运动循环与World Model

具身智能系统的基础结构仍然遵循第12章的感知-模型-规划-行动（Sense-Model-Plan-Act）循环，但每个环节都面临物理约束。在纯文本Agent中，"感知"是读取API返回的JSON；在具身Agent中，感知是RGB-D相机、激光雷达、触觉传感器的原始信号。这些高维、连续、噪声数据不能直接输入LLM的token序列。

当前的解决方案通常采用**分层表征**（hierarchical representation）：底层是视觉编码器（如CLIP ViT、DINOv2）将图像压缩为语义向量；中层是World Model（如RSSM或JEPA）在latent space中建模动态；顶层是LLM或决策模块，在压缩后的表征空间中进行推理。这种分层的动机来自信息论：原始图像（224×224×3 ≈ 150k维）包含大量与决策无关的信息（如像素级的噪声），而latent space（如RSSM的32维或256维）只保留预测未来所必需的状态信息。

一个具体的例子是RT-2（Brohan et al., 2023），它将机器人控制指令表示为自然语言token（如"将红色积木移到左边"），然后将视觉输入和语言指令共同编码为VLM的输入，输出离散的电机动作token。然而，RT-2的局限在于：它没有显式的World Model，无法预测"如果移动红色积木，绿色积木会不会倒塌"。为了弥补这一点，Socratic Model（Zeng et al., 2022）将视觉描述、物理常识和规划任务分别交给不同的预训练模型，通过语言作为"胶水"进行协调。但多模型之间的信息损失和延迟，使这种方法难以满足实时控制的需求（通常需要50-100Hz的控制频率）。

更系统的方案是将第12章的Dreamer架构直接扩展到真实机器人。DreamerV3在latent space中训练一个Recurrent State-Space Model (RSSM)，将图像序列映射为随机状态变量，然后基于预测的未来奖励进行策略优化。在机器人任务中，这种模型的优势是：latent dynamics学习的是仿真/真实环境的物理规律，而非特定视觉纹理。这意味着，如果latent state真正编码了物体位置、速度、质量等物理属性，那么从仿真到真实的迁移就转化为latent space中的分布偏移问题——而分布偏移可以通过少量真实数据微调来修正。Hafner et al., 2023在多种连续控制任务中验证了DreamerV3的泛化能力，包括需要精确接触的机械臂操作，但原始论文仍局限于仿真环境。后续工作如SafeDreamer（Huang et al., 2024）将安全约束（如碰撞避免）显式纳入world model的预测框架，使用Lagrangian methods在潜在空间中优化保守策略，这是真实机器人部署的前提。

### 15.2.2 Sim-to-Real Gap的本质

Sim-to-Real gap不是简单的"仿真不够精确"，而是三个层次差异的叠加：

**视觉差异（Visual Domain Gap）**：仿真中的纹理、光照、阴影与真实环境不同。即使是照片级真实渲染（如NVIDIA Isaac Sim），仍然与真实相机在动态范围、运动模糊、噪声特性上存在差异。RL-CycleGAN（Rao et al., 2020）提出了一种RL-aware的图像翻译方法：不是将仿真图像"美化"为真实风格，而是保留对RL策略有用的视觉特征（如物体边缘、深度线索），同时随机化与任务无关的背景。实验证明，这种RL-aware的翻译比标准CycleGAN在真实机器人抓取中的成功率提高了15-20%。

**动力学差异（Dynamics Domain Gap）**：刚体仿真器（MuJoCo, PyBullet）假设物体是刚性、质量均匀分布的，但真实物体存在柔性、形变、迟滞效应。例如，抓取一个装满水的瓶子时，瓶子的重心随倾斜角度变化；抓取一块布料时，接触点的摩擦力分布是非均匀的。这些效应无法通过简单的参数随机化覆盖。

**接触动力学差异（Contact Dynamics Gap）**：这是Sim-to-Real中最棘手的部分。仿真中的碰撞检测通常使用罚函数法（penalty method）或约束求解（LCP），需要人工调整接触刚度参数。真实世界中的接触涉及表面粗糙度、微观形变、粘弹性，参数空间极其复杂。Residual RL（Johannink et al., 2019）的洞见在于：与其在仿真中精确建模接触，不如让仿真策略负责"大致正确"的运动，然后在真实环境中用少量交互学习一个残差修正。数学上，真实动作 $a_{real} = a_{sim} + a_{residual}$，其中 $a_{sim}$ 是仿真策略输出的基础动作，$a_{residual}$ 是一个小幅度扰动，通常被约束在 $[-0.1, 0.1]$ 的范围内（相对于动作空间的归一化值）。

### 15.2.3 多模态感知与语言接地

具身智能的感知不仅限于视觉。触觉传感器（如GelSight、BioTac）提供接触力分布、滑动检测、纹理信息；本体感知（proprioception）提供关节角度、扭矩、速度；听觉感知可以检测物体碰撞、马达异常。如何将这些异构信号融合到统一的决策框架中，是2024-2025年的核心研究问题。

当前的融合策略有三种。第一种是**早期融合**（early fusion）：将所有传感器数据拼接为一个大向量，输入端到端神经网络。优点是简单，缺点是不同传感器的采样频率和噪声特性差异巨大（相机30Hz，触觉传感器1000Hz），强行同步会导致信息损失。第二种是**中期融合**（mid-level fusion）：每种模态独立编码为latent vector，然后在latent space中拼接。例如，Any mal机器人的视觉用ResNet编码，关节状态用MLP编码，两者拼接后输入策略网络。第三种是**晚期融合**（late fusion）：不同模态独立决策，然后通过投票或加权平均输出最终动作。这种方法在单模态失效时具有鲁棒性，但无法捕捉模态间的交互关系（如"看到杯子倾斜"和"感觉到滑动"同时发生时应立即调整抓握力）。

语言在具身智能中扮演了"接地"（grounding）的角色。纯强化学习的策略是黑箱：神经网络从像素到扭矩的映射无法解释。引入语言后，策略可以生成自然语言描述当前状态和计划（如"杯子已经倾斜，需要增大握力"），这不仅提高了可解释性，还允许人类通过语言纠正策略（如"不要碰杯子边缘，握住杯身"）。SayCan（Ahn et al., 2022）和Inner Monologue（Huang et al., 2022）是这一方向的代表：它们将LLM作为高级规划器，低级动作由预训练的策略执行，LLM在每一步评估"当前状态下，哪个技能是可执行的"。然而，这种分层架构的延迟较高（LLM推理通常需要100ms-1s），难以满足高速运动控制的需求。

---

## 15.3 科学发现：Agent作为虚拟实验室

### 15.3.1 从文献综述到假设生成

传统科学发现的过程是：科学家阅读文献→提出假设→设计实验→收集数据→分析结果→修正假设。这个循环中，LLM可以加速前两个环节，但后三个环节需要与物理世界交互。2023年后，AI for Science的主流范式从"用LLM读论文"转向"用Agent运行实验"。

以材料发现为例。传统的计算材料学使用密度泛函理论（DFT）计算候选材料的能带结构，但DFT计算成本高昂（一个结构需要数小时到数天的GPU时间）。如果有一个World Model可以快速预测DFT结果（即学习从晶体结构到能带的映射），Agent就可以在"想象"中筛选数万种候选材料，只将最有希望的几种送交真实DFT计算。这正是GNoME（Google DeepMind, 2023）的核心思想：训练一个图神经网络作为world model，预测晶体结构的稳定性，然后将其作为奖励函数指导材料生成。

化学合成领域也出现了类似范式。ChemCrow（Bran et al., 2023）和SynAgents（Zheng et al., 2024）将LLM与化学工具（如RDKit、Schrödinger）结合，Agent可以：1）用LLM提出合成路线；2）用分子模拟器验证该路线是否可行（如检查反应条件是否温和、副产物是否可控）；3）在真实实验室中由移液机器人执行。这里的World Model不是视频预测模型，而是分子动力学模拟器或量子化学计算工具——它们提供了物理定律的近似预测。

### 15.3.2 实验设计的主动学习

科学Agent的关键能力是**主动学习**（active learning）：在有限的实验预算下，选择信息量最大的实验。传统实验设计（Design of Experiments, DoE）使用高斯过程（Gaussian Process）建模输入-输出关系，然后基于采集函数（如Expected Improvement）选择下一个实验点。但高斯过程在高维空间（如蛋白质序列空间、化学反应条件空间）中扩展性差。

2024年的突破是将LLM与贝叶斯优化结合。LLM的知识先验（如"含有苯环的分子通常有较高的疏水性"）可以作为高斯过程的均值函数，大幅减少需要的实验次数。更重要的是，LLM可以提出非标准的实验设计：例如，在药物筛选中，传统方法逐个测试化合物；LLM-Agent可以建议先测试一组结构 diverse 的化合物以最大化信息增益，或者建议先测试一个已知活性化合物的衍生物以验证假设。

一个具体的例子是蛋白质工程中的定向进化（directed evolution）。传统方法对蛋白质序列进行随机突变，筛选出活性提高的变体。而使用World Model（如AlphaFold或ESM-2的结构预测）可以预测突变后的结构变化，从而评估哪些突变可能改善活性。Agent的决策循环是：1）ESM-2预测突变体结构；2）计算活性位点几何变化；3）选择最有希望的突变进行真实实验（表达蛋白并测量活性）。这种方法将突变筛选从随机搜索转变为有信息指导的搜索，实验效率提升10-100倍。

### 15.3.3 数据整合与因果发现

现代科学研究面临的数据碎片化问题：基因组数据在NCBI，蛋白质结构在PDB，代谢通路在KEGG，临床数据在各自医院的防火墙内。Agent需要跨越这些数据库进行整合。LLM的自然语言理解能力使其可以解析不同数据库的异构schema，但跨库查询的可验证性是一个问题——LLM可能"幻觉"出不存在的数据库字段或错误的连接条件。

更深层的问题是因果发现。相关性不等于因果性：A和B同时变化，可能是因为A导致B，B导致A，或存在共同原因C。传统因果推断方法（如PC算法、do-calculus）需要明确的变量定义和干预实验。在科学Agent中，World Model提供了"反事实推理"的能力：如果World Model可以预测"如果改变X，Y会如何变化"，那么Agent就可以在没有实际干预的情况下评估因果假设。例如，在药物发现中，World Model可以预测"如果抑制基因A，代谢物B的浓度会上升还是下降"，从而帮助Agent设计基因敲除实验来验证因果链路。

---

## 15.4 AI for Robotics：从仿真到真实世界

### 15.4.1 领域随机化与数据增强

第15.2节已经讨论了Domain Randomization的基本思想，但2023-2024年的进展使其更加系统化。传统Domain Randomization在视觉层面随机化纹理和光照，但在动力学层面仍使用固定的物理参数。新的方法如**随机化物理引擎本身**（Murthy et al., 2023）：训练策略时，同时运行多个不同物理引擎（MuJoCo, PyBullet, Isaac Gym）的仿真实例，策略必须在这些不一致的仿真中都能成功。这迫使策略学习对引擎假设不敏感的表征——本质上是学习更抽象的物理规律（如"物体不会穿模"、"重力向下"），而非特定的动力学参数。

另一项进展是**可微分仿真**（differentiable simulation）。如果仿真器是可微的，策略的梯度可以直接通过仿真器反向传播到物理参数，从而用梯度下降优化Domain Randomization的分布。但这要求仿真器能精确计算接触力的梯度，目前只在简单场景（如刚体碰撞、软体形变）中可行。

### 15.4.2 残差RL与自适应迁移

Residual RL（Johannink et al., 2019）的数学框架可以形式化为：

给定仿真策略 $\pi_{sim}(s)$，真实环境中的策略为 $\pi_{real}(s) = \pi_{sim}(s) + \pi_{res}(s)$，其中 $\pi_{res}$ 是残差策略。训练 $\pi_{res}$ 时，动作空间被约束为 $\|a_{res}\| \leq \epsilon$，确保残差只做小幅度修正。这种约束的动机是：仿真策略已经解决了大部分问题（如将机械臂移动到目标附近），真实环境只修正最后一小部分（如接触力的精确控制）。

Residual RL的成功依赖于两个假设：1）仿真策略在真实环境中至少不会崩溃（即不会导致危险状态）；2）残差策略的动作空间足够小，使得样本效率高到可以用真实数据训练。如果仿真策略在真实环境中完全失败（如机器人一启动就摔倒），Residual RL无法恢复。这时需要更保守的初始化方法，如**课程学习**（curriculum learning）：先在真实环境中训练简单任务（如保持站立），然后逐步增加难度。

自适应迁移（adaptive transfer）是Residual RL的扩展。不是训练一个固定的残差策略，而是让策略在线估计环境参数的变化。例如，机器人在地毯上行走时摩擦系数高，在瓷砖上行走时摩擦系数低。策略可以维护一个关于环境参数的贝叶斯后验，每一步根据观测更新后验，然后选择最优动作。这种方法类似于第13章的在线适应，但应用于物理参数而非任务目标。

### 15.4.3 视觉泛化与跨本体迁移

机器人策略面临的一个极端挑战是**跨本体迁移**（cross-embodiment transfer）：在一个机器人（如7-DoF Franka机械臂）上训练的策略，能否直接用于另一个机器人（如6-DoF UR10）？本体差异包括关节数量、连杆长度、工作空间、力矩限制。传统方法需要重新训练，但2024年的研究表明，如果策略在latent space中操作（而非直接在电机空间操作），跨本体迁移是可行的。

Learning to manipulate anywhere（Yuan et al., 2024）提出了一种视觉泛化框架：策略不是输出关节角度，而是输出末端执行器（end-effector）的6D位姿（3D位置 + 3D方向）和夹爪开合度。然后使用一个逆运动学（IK）求解器将6D位姿映射到具体机器人的关节角度。这种方法将"任务规划"（如何移动末端执行器）与"运动学执行"（如何将末端执行器位置映射到关节角度）解耦。不同机器人共享同一个任务策略，只需替换各自的IK求解器。实验表明，这种框架在模拟到真实、跨机器人、跨场景的抓取任务中实现了零样本（zero-shot）泛化。

视觉泛化的另一维度是**跨场景泛化**。机器人在实验室的白桌上训练抓取，需要在家庭的木桌上、厨房的台面上也能工作。关键是让视觉表征对背景不敏感。DINOv2（Oquab et al., 2023）和SAM（Kirillov et al., 2023）等自监督视觉模型提供了物体分割的通用表征，将这些表征作为策略的输入，可以使策略聚焦于物体本身而非背景。Yuan et al., 2024的框架正是基于这种视觉表征：策略网络接收DINOv2编码的物体特征，而非原始像素，从而在不同视觉环境下保持稳定性能。

---

## 15.5 三个关键障碍：数据瓶颈、泛化瓶颈、安全瓶颈

### 15.5.1 数据瓶颈

机器人数据比文本数据稀缺6-9个数量级。GPT-4训练使用了约13万亿token的文本数据，而最大的机器人数据集（如Open X-Embodiment，Padalkar et al., 2023）包含约100万条机器人轨迹，每条轨迹约100-1000步，总计约10亿帧图像。如果按数据量换算，机器人数据仅相当于GPT-4的0.001%。

更根本的问题是数据的异构性：不同机器人有不同的传感器配置、动作空间、任务定义。将Franka机械臂的抓取数据与四足机器人的行走数据合并，需要解决本体对齐（embodiment alignment）问题。当前的主流策略是**数据共享与标准化**：使用统一的观测空间（如RGB图像 + 本体感知）和动作空间（如末端执行器位姿），然后训练一个大型多机器人策略。但这种方法的局限是：低质量数据（如抖动、失败轨迹）会污染训练集，而筛选高质量数据需要人工标注。

### 15.5.2 泛化瓶颈

即使有了足够数据，机器人策略的泛化能力仍然远低于人类。人类可以从一个示范中学会"抓取杯子"，然后泛化到不同形状、大小、材质的杯子。而当前策略通常需要数千次示范才能掌握一个特定物体的抓取。

泛化瓶颈的根源在于表征的抽象层次。人类在抓取杯子时，表征的是"把手是可抓握的部分"、"杯口朝上"等语义概念；而当前策略的表征可能是"像素X到像素Y的映射"或"latent space中的某个区域"。多模态LLM（如GPT-4V、LLaVA）提供了提升表征抽象层次的可能：它们可以生成"杯子的把手在左侧"这样的语言描述，策略基于语言描述而非像素进行决策。但这又引入了延迟问题——LLM推理时间远大于控制频率需求。

另一种路径是**世界模型中的因果表征学习**。如果World Model学会了"杯子的把手决定了抓握位置"这一因果结构，那么策略就可以基于因果变量（如把手的位置）进行决策，而非基于像素级的相关特征（如杯子的颜色）。这类似于第12章讨论的JEPA架构中的分层预测：高层预测语义级别的变化，低层预测像素级别的变化。

### 15.5.3 安全瓶颈

真实机器人部署的最大障碍是安全。仿真中的策略失败只会导致reward降低；真实环境中的失败可能导致财产损失或人员伤害。当前的安全保障方法有三类：

**硬安全（Hard Safety）**：通过物理机制（如速度限制、关节力矩限制、紧急停止按钮）确保即使策略出错，机器人的运动范围也在安全区域内。这些约束不依赖AI，是工业机器人的标准配置，但会严重限制策略的灵活性。

**软安全（Soft Safety）**：在训练过程中加入约束，使策略学会避免危险状态。SafeDreamer（Huang et al., 2024）将约束强化学习（Constrained RL）与World Model结合，在latent space中预测"代价"（如碰撞概率、关节超限程度），然后使用Lagrangian multiplier在优化reward的同时满足安全约束。数学上，优化目标变为：

$$\max_{\pi} J_R(\pi) \quad \text{s.t.} \quad J_C(\pi) \leq \epsilon$$

其中 $J_R$ 是期望回报，$J_C$ 是期望代价。SafeDreamer的洞见是：在latent space中预测代价比在原始图像空间中更容易，因为latent state已经编码了与物理状态相关的压缩信息。

**验证安全（Verified Safety）**：使用形式化方法（formal methods）证明策略在特定条件下永远不会进入危险状态。例如，通过Reachability Analysis计算策略作用下系统状态可达集合，然后验证该集合与安全区域不相交。这种方法计算成本极高，目前只适用于低维系统（如2-3关节的机械臂），难以扩展到高维机器人。

---

## 15.6 代码：基于Dreamer的极简机器人导航任务

本节提供一个基于DreamerV3思想的简化世界模型实现，用于机器人在迷宫环境中的导航。这不是完整的DreamerV3（缺少完整的RSSM分布建模和多种优化技巧），但保留了核心思想：1）从图像中学习latent dynamics；2）在latent space中训练策略和价值函数；3）使用想象轨迹（imagined trajectories）进行规划。

```python
"""
Simplified Dreamer-style Robot Navigation
基于World Model的潜在空间强化学习，用于2D迷宫导航。
与第12章的Dreamer不同，此版本使用确定性latent dynamics，
便于教学和快速实验。在真实机器人部署前，应替换为完整的RSSM和actor-critic架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ============================================
# 1. 环境：简化的2D迷宫（可替换为真实机器人接口）
# ============================================

class SimpleMaze:
    """
    2D迷宫环境：agent在网格中移动，目标是从起点到达终点。
    观测为局部视野（3x3 around agent），奖励为距离目标的负值。
    """
    def __init__(self, size=10):
        self.size = size
        self.reset()
    
    def reset(self):
        self.agent_pos = np.array([1, 1], dtype=np.int32)
        self.goal_pos = np.array([self.size - 2, self.size - 2], dtype=np.int32)
        # 固定障碍物
        self.obstacles = [(3, 3), (3, 4), (4, 3), (5, 5), (5, 6), (6, 5)]
        return self._get_obs()
    
    def _get_obs(self):
        # 返回局部3x3视野，归一化到[0, 1]
        obs = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                x = self.agent_pos[0] + i - 1
                y = self.agent_pos[1] + j - 1
                if 0 <= x < self.size and 0 <= y < self.size:
                    if (x, y) in self.obstacles:
                        obs[i, j] = 1.0  # 墙
                    elif np.array_equal([x, y], self.goal_pos):
                        obs[i, j] = 0.5  # 目标
        return obs.flatten()  # 9维向量
    
    def step(self, action):
        # action: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_pos = self.agent_pos + np.array(moves[action])
        
        # 碰撞检测
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size 
            and tuple(new_pos) not in self.obstacles):
            self.agent_pos = new_pos
        
        # 奖励：距离目标的负欧氏距离 + 到达奖励
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.1 * dist
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward += 10.0
        
        return self._get_obs(), reward, done, {}


# ============================================
# 2. World Model：Encoder -> Latent Dynamics -> Decoder
# ============================================

class WorldModel(nn.Module):
    """
    简化世界模型：
    - Encoder: 将观测压缩为latent state z
    - Dynamics: 预测 z_{t+1} = f(z_t, a_t)
    - Decoder: 从z重构观测（用于训练监督）
    - Reward Predictor: 从z预测reward
    """
    def __init__(self, obs_dim=9, action_dim=4, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Encoder: obs -> z
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Dynamics: z_t, a_t -> z_{t+1}
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder: z -> obs_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
            nn.Sigmoid()  # 输出在[0,1]
        )
        
        # Reward predictor: z -> r_hat
        self.reward_pred = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def encode(self, obs):
        return self.encoder(obs)
    
    def predict_next(self, z, action):
        """给定当前latent state和action，预测下一latent state"""
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        za = torch.cat([z, action_onehot], dim=-1)
        return self.dynamics(za)
    
    def decode(self, z):
        return self.decoder(z)
    
    def predict_reward(self, z):
        return self.reward_pred(z).squeeze(-1)
    
    def forward(self, obs, action):
        z = self.encode(obs)
        z_next = self.predict_next(z, action)
        obs_hat = self.decode(z_next)
        reward_hat = self.predict_reward(z_next)
        return obs_hat, reward_hat, z, z_next


# ============================================
# 3. Actor-Critic in Latent Space
# ============================================

class LatentActorCritic(nn.Module):
    """在World Model的latent space中运行的策略和价值网络"""
    def __init__(self, latent_dim=32, action_dim=4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def get_action(self, z, deterministic=False):
        logits = self.actor(z)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()
    
    def get_value(self, z):
        return self.critic(z).squeeze(-1)


# ============================================
# 4. 训练循环：World Model + Actor-Critic联合训练
# ============================================

class DreamerLiteAgent:
    """
    简化版Dreamer Agent：
    1. 收集真实环境轨迹
    2. 用真实数据训练World Model（重构 + 预测reward）
    3. 用World Model生成想象轨迹，训练Actor-Critic
    """
    def __init__(self, obs_dim=9, action_dim=4, latent_dim=32, 
                 gamma=0.99, lr=3e-4, device='cpu'):
        self.device = device
        self.gamma = gamma
        
        self.world_model = WorldModel(obs_dim, action_dim, latent_dim).to(device)
        self.ac = LatentActorCritic(latent_dim, action_dim).to(device)
        
        self.optimizer_wm = torch.optim.Adam(
            self.world_model.parameters(), lr=lr
        )
        self.optimizer_ac = torch.optim.Adam(
            self.ac.parameters(), lr=lr
        )
        
        self.buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.imagine_horizon = 15  # 想象轨迹长度
    
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def _sample_batch(self):
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_obs)).to(self.device),
            torch.FloatTensor(dones).to(self.device)
        )
    
    def train_world_model(self):
        if len(self.buffer) < self.batch_size:
            return {}
        
        obs, actions, rewards, next_obs, _ = self._sample_batch()
        
        obs_hat, reward_hat, z, z_next = self.world_model(obs, actions)
        
        # 损失1：观测重构
        recon_loss = F.mse_loss(obs_hat, next_obs)
        # 损失2：reward预测
        reward_loss = F.mse_loss(reward_hat, rewards)
        
        loss = recon_loss + reward_loss
        
        self.optimizer_wm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.optimizer_wm.step()
        
        return {'recon_loss': recon_loss.item(), 'reward_loss': reward_loss.item()}
    
    def train_actor_critic(self):
        """使用想象轨迹训练Actor-Critic（Dreamer的核心）"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        obs, _, _, _, _ = self._sample_batch()
        
        with torch.no_grad():
            z = self.world_model.encode(obs)
        
        # 想象轨迹：从真实latent state出发，用World Model和当前策略 rollout
        imagined_zs = [z]
        imagined_rewards = []
        
        for t in range(self.imagine_horizon):
            action = self.ac.get_action(imagined_zs[-1])
            z_next = self.world_model.predict_next(imagined_zs[-1], action)
            r_hat = self.world_model.predict_reward(z_next)
            imagined_zs.append(z_next)
            imagined_rewards.append(r_hat)
        
        # 计算TD(lambda) returns
        values = [self.ac.get_value(z) for z in imagined_zs]
        returns = []
        g = values[-1]
        for r, v in zip(reversed(imagined_rewards), reversed(values[:-1])):
            g = r + self.gamma * g
            returns.insert(0, g)
        
        # Actor损失：最大化优势（return - value）
        # Critic损失：拟合return
        z_stack = torch.stack(imagined_zs[:-1])
        returns = torch.stack(returns)
        
        logits = self.ac.actor(z_stack.view(-1, z_stack.size(-1)))
        actions_imagined = torch.stack([self.ac.get_action(z_i) for z_i in imagined_zs[:-1]]).view(-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions_imagined.unsqueeze(1)).squeeze(1)
        
        values_pred = self.ac.get_value(z_stack.view(-1, z_stack.size(-1)))
        advantages = (returns.view(-1) - values_pred).detach()
        
        actor_loss = -(selected_log_probs * advantages).mean()
        critic_loss = F.mse_loss(values_pred, returns.view(-1).detach())
        
        loss = actor_loss + critic_loss
        
        self.optimizer_ac.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
        self.optimizer_ac.step()
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
    
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            z = self.world_model.encode(obs_t)
            action = self.ac.get_action(z, deterministic=deterministic)
        return action.item()


# ============================================
# 5. 主训练循环
# ============================================

def train_robot_navigation():
    env = SimpleMaze(size=10)
    agent = DreamerLiteAgent(
        obs_dim=9, action_dim=4, latent_dim=32, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    num_episodes = 2000
    log_interval = 200
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        # 每回合训练World Model和Actor-Critic
        if len(agent.buffer) >= agent.batch_size:
            for _ in range(5):  # 每次环境交互后更新5次
                wm_loss = agent.train_world_model()
                ac_loss = agent.train_actor_critic()
        
        if episode % log_interval == 0:
            print(f"Episode {episode}, Steps: {steps}, Reward: {episode_reward:.2f}, "
                  f"Buffer: {len(agent.buffer)}")
    
    # 测试
    obs = env.reset()
    trajectory = [env.agent_pos.copy()]
    for _ in range(50):
        action = agent.select_action(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        trajectory.append(env.agent_pos.copy())
        if done:
            break
    print("Test trajectory:", trajectory)
    return agent


if __name__ == "__main__":
    agent = train_robot_navigation()
```

**代码说明**：
- `SimpleMaze` 环境是2D网格，可替换为真实机器人接口（如PyBullet中的TurtleBot或Franka机械臂）。
- `WorldModel` 实现了从观测到latent state的压缩、latent dynamics预测、以及reward预测。注意此版本使用确定性latent state；完整Dreamer使用RSSM，其中latent state分解为确定性路径（ recurrent state）和随机变量（stochastic state），以捕捉环境中的不确定性。
- `LatentActorCritic` 不在像素空间而在latent space中运行，这意味着一旦World Model训练完成，策略训练不依赖真实环境——可以在想象（imagination）中进行，这正是Dreamer样本效率高的原因。
- `train_actor_critic` 中的想象轨迹是Dreamer的核心：用当前World Model和当前策略从真实latent state出发，rollout未来15步，计算TD returns，然后优化策略。这使得agent在"脑内模拟"中学习，大幅减少真实环境交互次数。
- 在真实机器人部署时，应将`obs_dim`替换为相机图像维度（如64×64×3=12288），并在Encoder中加入卷积层；同时需要增加 domain randomization 或 residual policy 模块以应对Sim-to-Real gap。

---

## 15.7 工程实践框：Sim-to-Real迁移、真实机器人安全协议、数据收集成本

### 训练时的常见坑

**坑1：World Model在仿真中过拟合，导致latent dynamics无法泛化到真实环境。** 症状：仿真中想象轨迹完美，真实机器人第一步就崩溃。诊断：检查World Model在真实数据（哪怕只有10条轨迹）上的重构误差。如果真实数据的重构误差比仿真数据高一个数量级，说明latent space没有学到与物理相关的表征，而是过拟合了仿真纹理。修复：增加Encoder的容量，使用更强的数据增强（如随机裁剪、颜色抖动），或引入 domain adversarial training 让latent space对视觉域不敏感。

**坑2：Residual RL的残差策略饱和，导致修正幅度过大。** 症状：仿真策略在真实环境中一开始工作，但残差策略逐渐覆盖基础策略，最终行为不稳定。诊断：监控残差动作的L2范数，如果超过预设阈值（如0.1）的步数比例持续上升，说明残差策略在"抢戏"。修复：在残差策略的损失函数中加入L2正则化，或使用动作掩码（action mask）将残差约束在基础动作的邻域内。

**坑3：想象轨迹发散（imagination divergence）。** 症状：Dreamer-style agent在训练初期表现正常，但后期世界模型的预测误差累积，想象轨迹越来越偏离真实动态，导致策略在想象中学会的行为在现实中失败。诊断：在训练日志中绘制"想象return vs 真实return"的散点图，如果相关系数低于0.5，说明想象不可靠。修复：缩短想象horizon（从50步降到15步），增加World Model的训练步数，或使用 ensemble of world models 对想象轨迹进行不确定性估计，只使用高置信度的想象数据训练策略。

**坑4：触觉/力传感器数据与视觉数据不同步。** 症状：策略在视觉上看起来正确，但接触力异常。诊断：检查传感器数据的时间戳，如果触觉数据比视觉数据延迟50ms以上，策略在接触瞬间使用的是"接触前"的视觉观测。修复：使用时间对齐的数据缓冲区，或将历史观测堆叠（stack）为输入（如过去4帧），让策略能感知到时间序列的变化。

### 超参选择

| 超参数 | 仿真训练 | Sim-to-Real训练 | 说明 |
|--------|----------|----------------|------|
| 想象horizon | 50-100 | 5-15 | 真实环境中世界模型误差累积更快，缩短horizon可减少噪声 |
| 残差动作边界 $\epsilon$ | N/A | 0.05-0.15 | 真实机器人通常用关节位置的归一化动作空间；残差幅度取决于关节精度 |
| World Model学习率 | 3e-4 | 1e-4 | 真实数据噪声大，降低学习率提高稳定性 |
| Actor-Critic更新比例 | 1:1 | 1:2 | 真实环境中价值估计更困难，增加critic更新频率 |
| Batch size | 256 | 64 | 真实数据少，小batch避免过拟合；增大数据增强强度补偿 |
| 折扣因子 $\gamma$ | 0.99 | 0.95-0.98 | 真实机器人任务通常更短视，因为长期预测不可靠 |

### 硬件需求估算

**仿真训练**：训练一个针对7-DoF机械臂的DreamerV3风格策略，在MuJoCo环境中（观测为64×64 RGB，控制频率20Hz，每回合1000步）：
- World Model（CNN Encoder + RSSM + Decoder）：在单张RTX 4090上，训练100万环境步约需8-12小时。
- 如果使用分布式仿真（如Isaac Gym，支持 thousands of parallel envs），同样数据量可在1-2小时内完成，但需要更大的GPU显存（建议24GB+）。
- 内存：32GB RAM足够；如果使用图像replay buffer，100万帧64×64×3的float32图像约需14.6GB，应存储为uint8（压缩至3.7GB）。

**真实机器人训练**：
- Residual RL在真实机器人上的数据效率通常为每任务50-500次真实交互（每次交互约10秒）。如果实验是连续运行，单任务训练时间为8分钟到1.5小时。但需要人工重置环境和监控安全。
- 如果使用遥操作收集模仿学习数据（如用VR手柄控制机械臂），收集100条高质量轨迹（每条10-20秒）需要熟练操作员2-4小时。成本约为操作员时薪加上机器人损耗（约$20-50/小时）。
- 大规模多机器人数据收集（如Open X-Embodiment项目）需要数十台机器人和全职工程师团队，成本在百万美元级别。

**安全协议（真实机器人部署前的必检清单）**：
1. 速度限制：关节速度不超过最大安全速度的50%（如Franka Panda的额定关节速度为2.5 rad/s，部署时限制为1.25 rad/s）。
2. 工作空间限制：在软件中定义虚拟边界（virtual walls），如果机器人末端执行器超出预设区域，立即触发零力矩模式（zero-torque mode）。
3. 接触力检测：使用机器人内置的力矩传感器监控末端执行器的力/力矩。如果超过阈值（如20N），立即停止并回溯。
4. 急停机制：物理急停按钮（hardware E-stop）必须在操作员触手可及的位置；软件急停（在检测到异常时切断电机电源）的延迟必须低于50ms。
5. 人在回路（human-in-the-loop）：前50次真实部署中，必须有人类监督员随时接管控制；策略只被允许在监督员按下"允许"按钮后执行动作。
6. 夜间/无人测试：绝不允许在无人值守的情况下运行未经充分验证的策略。即使是成熟策略，也应在隔离区域（如安全笼）中测试。

---

## 15.8 落地场景：端到端自动驾驶、人形机器人与VLA模型

前面的章节聚焦于机械臂和导航小车等实验室场景。但2024-2025年，三个更大的落地场景正在重塑“World Model + Agent”的工业实践。本节简要讨论这三个场景及其对全书框架的启示。

### 15.8.1 端到端自动驾驶：World Model 最大的落地场景

自动驾驶是World Model概念的最大规模工业应用。Tesla FSD V12+（2024年起）采用端到端神经网络架构，将摄像头输入直接映射为控制输出，不再依赖人工编写的规则模块。Waymo的第五代系统也在向端到端方向迁移。Uber ATG的研究表明， Learned World Model 可以在内部预测其他车辆和行人的未来轨迹，用于规划。

关键观察：自动驾驶领域的实践并不遵循“LLM + Agent + World Model 三者显式集成”的架构。Tesla 的做法更接近“端到端 VLA”——视觉输入直接映射到控制输出，World Model 是隐式习得的（模型的中间层表征编码了环境动力学），而非显式的潜在空间预测器。这直接挑战了第1章的“三者必须集成”叙事：如果端到端 VLA 可以解决问题，显式 World Model 是否还有必要？

一种观点认为，显式 World Model 在可解释性、可验证性和安全关键场景下仍有价值——你可以检查 World Model 的预测是否合理，但无法检查端到端模型的隐藏层。另一种观点认为，随着模型规模增大，端到端模型的隐式 World Model 会越来越准确，显式分解的工程收益将消失。这个争论在2025年仍未解决。

### 15.8.2 人形机器人 LLM Policy

2024-2025年，人形机器人领域出现了一波“LLM as robot policy”的浪潮。Figure 02（Figure AI, 2024）直接接入 OpenAI 的多模态模型进行语音交互和任务规划。Tesla Optimus Gen 2 展示了基于神经网络的全身控制。Unitree H1/G1 通过开源生态降低了人形机器人的硬件门槛。

这些系统的架构与本书前面讨论的“Dreamer + Residual RL”路线有显著区别。Figure 02 的架构更像“LLM 做高层规划 + 传统控制器做低层执行”——LLM 负责理解语言指令、分解任务、选择技能，而低层的行走、抓取由专门的策略网络或传统控制器完成。这是一种松耦合集成，而非统一架构。

对全书的启示：人形机器人的工业实践表明，在当前技术阶段，松耦合集成（LLM 做规划 + 专用策略做执行）比紧耦合集成（统一表征空间）更实用。第14章讨论的统一架构方案目前仍处于研究阶段。

### 15.8.3 VLA 模型：跳过显式 World Model 的端到端路径

视觉-语言-行动模型（Vision-Language-Action Model, VLA）是2023-2025年机器人领域最重要的新路线。RT-2 [Brohan et al., 2023] 首次证明，一个在互联网文本和图像上预训练的 VLM 可以通过机器人轨迹数据微调，直接输出机器人动作。OpenVLA [Kim et al., 2024] 将这一思路开源化。π0 [Black et al., 2024] 进一步用 flow matching 替代自回归生成作为动作解码器，在灵巧操作任务上取得了 SOTA 性能。

VLA 的核心思想是：不显式建模 World Model，而是通过大规模多模态预训练让模型隐式习得物理直觉。这与 Dreamer 路线（显式学习潜在动力学）和第1章的“三者集成”叙事形成了直接张力。

关键问题：VLA 路线是否意味着显式 World Model 的终结？目前的证据不支持如此强的结论：
1. VLA 在数据充足的任务上表现优异，但在数据稀缺的新任务上泛化能力有限——而 World Model 的想象性 rollout 可以在少量真实数据上生成合成训练数据。
2. VLA 的隐式 World Model 无法被人类检查和验证，这在安全关键场景下是缺陷——而显式 World Model 的预测可以被审计。
3. VLA 的动作生成是端到端的，难以注入人类常识约束——而 LLM + World Model 架构可以通过 LLM 的语义层注入“不要碰红色按钮”这样的约束。

结论：VLA 和显式 World Model 是互补而非替代关系。在不同任务、不同数据条件、不同安全要求下，两种路径各有优势。第14章的集成方案和本节的 VLA 路线应被视为一个谱系的两端，而非非此即彼的选择。

---

## 15.9 小结与全书总结

本章讨论了三条前沿路径：具身智能让Agent从数字世界进入物理世界，科学发现让Agent成为虚拟实验室中的研究员，AI for Robotics将强化学习、World Model和机器人学融合为统一的工程系统。这三条路径共享一个核心挑战：从仿真到真实的迁移。第12章的Dreamer提供了样本效率的基础，Residual RL和Domain Randomization提供了迁移的工具，而多模态LLM为机器人提供了语义级的推理能力。但当前技术仍然存在数据瓶颈（机器人数据比文本数据少6-9个数量级）、泛化瓶颈（策略难以从一种物体泛化到另一种）和安全瓶颈（真实部署需要多重保障机制）。

回顾全书，我们构建了一个完整的认知框架：
- **第1-5章**（LLM）解决了"理解"的问题——语言、知识、推理；
- **第6-10章**（Agent）解决了"组织"的问题——记忆、工具使用、规划、多Agent协作；
- **第11-14章**（World Model）解决了"预测"的问题——从图像中学习物理规律，在想象中规划未来；
- **第15章**（前沿）展示了"行动"的可能——将理解、组织、预测的能力转化为对物理世界的干预。

Agent·LLM·World Model的交汇点，正是人工智能从"知识压缩"走向"世界干预"的转折点。LLM压缩了人类的知识，World Model压缩了物理的规律，Agent将它们组织为可执行的策略。当这三个系统真正融合——当机器人能用自然语言解释自己的行为，能在想象中预演实验，能从失败中快速学习——我们将拥有不仅是"智能"而且是"智慧"的系统：理解世界，预测未来，并负责任地改变它。

---

## 延伸阅读

1. **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023).** Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*.
   - DreamerV3的完整论文，提出了RSSM的改进版本，在超过150种连续控制任务中验证了World Model的样本效率和泛化能力。第12章的核心参考文献，也是本章15.6代码的理论基础。

2. **Johannink, T., Bahl, S., Nair, A., Luo, J., et al. (2019).** Residual reinforcement learning for robot control. *IEEE International Conference on Robotics and Automation (ICRA)*.
   - Residual RL的奠基工作，展示了如何将仿真策略与真实环境的残差修正结合，特别适用于接触动力学难以建模的机器人操作任务。直接支撑第15.4.2节的讨论。

3. **Rao, K., Harris, C., Irpan, A., Levine, S., et al. (2020).** RL-CycleGAN: Reinforcement learning aware simulation-to-real. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
   - 提出了一种对RL友好的图像域自适应方法，在保留任务相关视觉特征的同时消除仿真与真实的视觉差异，是Domain Randomization之外的重要Sim-to-Real路径。

4. **Yuan, Z., Wei, T., Cheng, S., Zhang, G., Chen, Y., et al. (2024).** Learning to manipulate anywhere: A visual generalizable framework for reinforcement learning. *arXiv preprint arXiv:2407.15815*.
   - 提出了跨场景、跨机器人的视觉泛化框架，通过将策略输出从关节空间提升到末端执行器位姿空间，实现了零样本Sim-to-Real和跨本体迁移。支撑第15.4.3节的讨论。

5. **Huang, W., Ji, J., Xia, C., Zhang, B., et al. (2024).** SafeDreamer: Safe reinforcement learning with world models. *International Conference on Learning Representations (ICLR)*.
   - 将约束强化学习引入World Model框架，在latent space中显式建模安全代价，为真实机器人部署提供了理论保障和实用算法。支撑第15.5.3节的安全讨论。

---

*本章完*
