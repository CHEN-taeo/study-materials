# 第5章：推理与思维链（CoT）——从 prompt 工程到长程推理的涌现

## 核心思想

大语言模型的推理能力不是预训练的自然产物，而是"生成中间步骤"（intermediate reasoning steps）这一结构化解码策略所诱导的涌现行为。Chain-of-Thought prompting（CoT）的本质不是让模型"更聪明"，而是通过改变输出格式（从直接答案变为逐步推理）激活了预训练模型中已有的、但被直接解码所抑制的符号操作能力。理解 CoT 的边界——它在哪些任务上有效、为什么有效、以及它生成的推理链是否忠实反映了模型的内部推理——是设计可靠 Agent 系统的必要前提。

---

## 历史脉络：从直接答案到逐步推理的范式转移

在 GPT-3 之前，解决数学或逻辑问题的标准方法是训练专门的符号系统（如定理证明器、约束求解器）或微调任务特定的神经网络（如seq2seq模型）。这些方法的问题是：它们需要大量标注数据，且泛化能力有限。GPT-3 的 few-shot learning 展示了另一种可能性：通过 prompt 中的示例，模型可以直接学习任务的输入-输出映射，无需梯度更新。

但 few-shot 学习在复杂推理任务上表现不佳。例如，当 GPT-3 被问到"一个农场有 37 只羊，又买了 15 只，现在有多少只？"时，它可能直接回答"52"（正确），也可能回答"47"或"37"（错误）。这种错误率随着问题复杂度（如多步算术、符号推理）急剧上升。2022 年，Wei et al. [Chain-of-Thought prompting elicits reasoning in large language models, 2022] 提出了一个简单但革命性的方法：在 prompt 中不仅提供输入-输出对，还提供**中间推理步骤**。例如：

Q: 一个农场有 37 只羊，又买了 15 只，现在有多少只？
A: 首先，农场原有 37 只羊。然后，又买了 15 只。所以总数是 37 + 15 = 52。答案是 52。

加入这种逐步推理的示例后，GPT-3 在 GSM8K（小学数学问题数据集）上的准确率从约 10% 提升到约 40%。这一提升不是通过微调实现的，而是通过**改变 prompt 格式**实现的——模型参数没有更新，只是解码策略改变了。

后续研究迅速扩展了 CoT 的框架。Self-Consistency [Wang et al., 2022] 发现，对同一个问题采样多个推理链，然后取多数投票的答案，可以进一步提升准确率。Tree of Thoughts [Yao et al., 2023] 将线性推理链扩展为树状搜索，允许模型在推理过程中回溯和评估不同路径。2024-2025 年，Long CoT（长程思维链）成为新的研究热点，模型通过生成数百甚至数千个推理步骤来解决复杂问题。但与此同时，研究也开始质疑 CoT 的忠实性：模型生成的推理链是否真实反映了它的内部推理过程？Sprague et al. [2025] 发现，在某些情况下，模型在推理链中"编造"了不存在的推理步骤，但答案仍然正确——这说明 CoT 可能不是真正的推理，而是答案的"事后合理化"。

---

## 5.1 推理的涌现：为什么大模型突然能解数学题

### 5.1.1 涌现能力的定义与争议

"涌现"（Emergence）在 AI 中指的是：当模型规模超过某个阈值后，某些能力突然"出现"——这些能力在小模型中不存在，在大模型中却显著存在。Wei et al. [2022] 的实验表明，CoT 的效果在模型参数超过约 100B 时才显著，而在 10B 以下的模型中几乎无效。这引发了一个问题：推理能力是模型规模的连续函数，还是存在一个真正的"相变"（phase transition）？

后续研究对此存在分歧。一些研究 [Schaeffer et al., 2023] 认为，所谓的"涌现"只是度量方式造成的假象：如果改用连续度量（如困惑度）而非二元度量（如准确率），能力的增长是平滑的。另一些研究 [Merrill & Sabharwal, 2023] 则从计算复杂性角度证明，Transformer 的推理能力受限于电路复杂性，某些推理任务（如需要指数级深度电路的任务）不可能通过简单扩大规模来解决。

### 5.1.2 预训练中的隐含推理

无论涌现是否真实，CoT 的效果表明：大语言模型在预训练过程中已经隐含地学习了推理模式。这些模式不是显式编码的（预训练数据中没有"推理规则"），而是从海量文本中的**解释性段落**中统计学习到的。例如，数学教科书、编程教程、科学论文中充满了"逐步推导"的文本，模型通过预测这些文本，学会了生成推理链的格式和结构。

但这只是格式学习，还是真正的推理能力？这是一个关键问题。如果 CoT 只是格式模仿，那么模型在训练时见过的任务类型上表现好，在全新类型的推理任务上表现差。如果 CoT 是真实推理，那么模型应该能够泛化到训练时未见的推理模式。目前证据支持中间立场：CoT 在训练分布内的任务上表现优异，在分布外任务上表现下降，但下降速度比直接回答慢——这表明 CoT 至少部分激活了模型的符号操作能力。

---

## 5.2 Chain-of-Thought：中间步骤的力量

### 5.2.1 CoT 的数学直觉

从信息论的角度，CoT 可以看作是一种**结构化编码**：它将问题的解从单个答案 y 扩展为序列 (r_1, r_2, ..., r_k, y)，其中 r_i 是中间推理步骤。这种编码有两个优势：

1. **分解复杂度**：将复杂问题分解为简单子问题，每个子问题只需要局部推理。例如，多步算术 "(37 + 15) × 2" 可以分解为 "先算 37+15=52，再算 52×2=104"。
2. **增加监督信号**：每个中间步骤 r_i 都提供了额外的监督信号，即使最终答案 y 错误，中间步骤也可能部分正确。这类似于课程学习（curriculum learning）——从简单到复杂逐步学习。

### 5.2.2 Zero-shot CoT：不需要示例的推理

原始 CoT 需要 few-shot 示例（即 prompt 中包含几个带推理步骤的示例）。Kojima et al. [2022] 发现，即使不提供示例，只需在问题后追加一句 "Let's think step by step"，模型也能生成推理链。这称为 Zero-shot CoT。

Zero-shot CoT 的发现具有重要意义：它表明模型已经内化了"逐步推理"的格式，不需要外部示例来触发。但 Zero-shot CoT 的效果通常不如 few-shot CoT——提示语的质量（如 "Let's think step by step" vs "Let's work through this carefully"）对性能有显著影响。这提示我们：CoT 的效果不仅取决于模型能力，还取决于**提示工程**（prompt engineering）的质量。

---

## 5.3 Self-Consistency：从单次推理到集合推理

### 5.3.1 多数投票的数学原理

Self-Consistency [Wang et al., 2022] 的核心思想是：对同一个问题，使用温度采样（temperature sampling）生成多个推理链，然后取答案的多数投票。数学上，如果单个推理链的正确概率是 p，生成 k 个独立链后取多数投票，正确概率约为：

P_correct ≈ Σ_{i=⌈k/2⌉}^k C(k,i) p^i (1-p)^{k-i}

当 p > 0.5 时，随着 k 增加，P_correct 趋近于 1（大数定律）。当 p < 0.5 时，多数投票反而降低正确率。因此，Self-Consistency 只在模型已经"勉强正确"的情况下有效——如果模型完全不理解问题，生成更多链只是生成更多错误。

### 5.3.2 推理路径的多样性

Self-Consistency 的效果依赖于**推理路径的多样性**。如果所有采样链都遵循相同的推理模式，那么多数投票只是重复了同一个错误。为了增加多样性，可以：
- 使用较高的温度（如 T=0.7-1.0）来增加采样的随机性；
- 使用不同的 prompt 变体（如 "Let's think step by step" vs "Let's solve this carefully"）；
- 使用不同的分解策略（如从问题结尾倒推 vs 从开头顺推）。

但多样性也有代价：当温度过高时，推理链可能出现逻辑错误或不相关的步骤。实验表明，k=5-10 的采样次数是 sweet spot——进一步增加 k 的收益递减，而计算成本线性增长。

---

## 5.4 Tree of Thoughts：从线性链到分支搜索

### 5.4.1 线性 CoT 的局限

标准 CoT 是线性的：一旦生成一个推理步骤，就不能回溯修改。这类似于人类解决问题时"一条路走到黑"的策略。但对于需要探索多种可能性的问题（如谜题、规划、创造性任务），线性策略是低效的。

Tree of Thoughts（ToT） [Yao et al., 2023] 将推理扩展为树状搜索：
1. **分解**：将问题分解为若干思考步骤；
2. **生成**：对每个步骤生成多个候选想法；
3. **评估**：用 LLM 或启发式函数评估每个候选想法的质量；
4. **搜索**：使用 BFS（广度优先搜索）或 DFS（深度优先搜索）探索最有希望的路径。

ToT 在需要创造性探索的任务（如 24 点游戏、创意写作）上显著优于线性 CoT。但 ToT 的计算成本也更高：如果每个步骤生成 k 个候选，深度为 d，那么总节点数是 O(k^d)。这使得 ToT 在实时应用中难以部署。

---

## 5.5 长程 CoT：从 Few-shot 到 Long CoT

### 5.5.1 为什么需要长程推理？

标准 CoT 通常只有 5-15 个推理步骤。但对于复杂问题（如数学竞赛题、代码调试、科学推理），人类可能需要数百个步骤。Long CoT 的目标是让模型生成这种长程推理链。

2024-2025 年的研究表明，通过**强化学习**（如 GRPO、PPO）诱导模型生成更长的推理链，可以显著提升复杂任务的性能。例如，DeepSeek-R1 使用 GRPO（Group Relative Policy Optimization）训练模型生成数千个 token 的推理过程，在数学竞赛（AIME）和代码竞赛（Codeforces）上达到了接近人类顶尖选手的水平。

### 5.5.2 Long CoT 的训练方法

Long CoT 的训练通常涉及以下步骤：
1. **冷启动**：使用少量高质量的 Long CoT 数据（如人工标注的详细推导）进行 SFT（Supervised Fine-Tuning）；
2. **强化学习诱导**：使用 RL 奖励模型来鼓励模型生成长推理链。奖励不仅基于最终答案的正确性，还基于推理链的完整性（如是否包含所有必要步骤）；
3. **拒绝采样**：从 RL 训练后的模型中采样大量推理链，筛选出正确的链，用于进一步的 SFT；
4. **迭代**：重复步骤 2-3，直到性能收敛。

这种训练方法的一个关键发现是：模型在 RL 训练过程中会自发地发展出新的推理策略（如自我验证、回溯、子目标分解），这些策略不是显式编程的，而是 reward 信号诱导的涌现行为。这与人类学习数学的过程类似：通过大量练习和反馈，学生逐渐发展出高效的解题策略。

### 5.5.3 两条技术路线：Test-Time Compute Scaling vs RL 诱导

2024-2025 年的长程推理研究实际上沿两条不同路线展开，理解它们的区别对设计 Agent 系统至关重要。

**路线一：Test-Time Compute Scaling（o1 路线）**。OpenAI 于 2024 年 9 月发布的 o1 模型（以及后续的 o3、o4-mini）采用的核心策略是：在推理阶段（而非训练阶段）大幅增加计算投入。o1 在生成最终答案前，会在隐藏的思维链（hidden chain-of-thought）中进行大量内部推理——搜索多条路径、验证中间结果、回溯错误方向。这意味着模型的推理能力不再仅由参数规模决定，还由推理时允许的计算量决定。o1 的关键特征是：推理过程对用户不可见（或仅提供摘要），且推理的“长度”由模型自主决定，而非由 prompt 控制。从工程角度看，这是将“思考时间”从训练阶段转移到推理阶段，用更多的 inference-time FLOPs 换取更高的准确率。

**路线二：RL 诱导长推理链（R1 路线）**。DeepSeek 于 2025 年初发布的 DeepSeek-R1 采用的策略是：通过强化学习直接训练模型生成显式的、可见的长推理链。R1 使用 GRPO（Group Relative Policy Optimization）作为优化算法——与标准 PPO 相比，GRPO 用组内相对优势（同一问题的多个采样之间的相对排名）代替全局 value network 的 baseline，减少了对独立 value 模型的需求。R1 的推理链是可见的、可审查的，这使得它的推理过程比 o1 更透明，但也意味着用户需要消耗更多 output token。

**两条路线的关键差异**：

1. **推理可见性**：o1 的思维链是隐藏的，R1 的推理链是可见的。这影响了可审查性和调试难度——R1 的推理过程可以被人类检查和验证，而 o1 只能信任最终输出。
2. **计算控制点**：o1 在推理时动态决定“思考多久”，R1 的推理链长度主要由训练时的 RL 信号塑造。前者更像“给模型更多时间思考”，后者更像“训练模型养成深入思考的习惯”。
3. **泛化机制**：o1 的 test-time scaling 假设模型在预训练中已经习得了推理能力，只需在推理时“释放”出来；R1 的 RL 诱导假设模型需要通过 RL 信号“学会”如何生成长推理链。两者的假设不同，适用的任务可能也不同。
4. **工程集成**：对于 Agent 系统，R1 路线更容易集成——推理链是显式文本，可以被 Agent 的后续步骤解析和使用。o1 路线的隐藏思维链对 Agent 不可见，Agent 只能使用最终答案，无法利用中间推理步骤。

目前（2025 年中）两条路线各有优势：o1 在通用推理任务上表现更强，R1 在数学和代码竞赛上接近 o1 且推理过程透明。两条路线是否会收敛——例如通过 RL 训练模型同时掌握可见推理和隐藏推理——是 2025-2026 年的关键研究方向。

---

## 5.6 CoT 的边界与忠实性

### 5.6.1 任务依赖：CoT 不是万能的

CoT 并非对所有任务都有效。研究表明，CoT 在以下任务上效果显著：
- 数学推理（算术、代数、几何）
- 符号推理（逻辑谜题、代码生成）
- 多步决策（规划、调度）

但在以下任务上效果有限或无效：
- 常识推理（如"为什么天空是蓝的？"）——常识问题不需要逐步推理，直接回答更高效；
- 知识检索（如"法国的首都是什么？"）——这类问题不需要推理，只需要记忆；
- 模式识别（如"这个序列的下一个数字是什么？"）——模式识别通常是直觉性的，难以分解为步骤。

Sprague et al. [2024] 的系统性分析表明，CoT 的效果与任务的**可分解性**（decomposability）高度相关：如果任务可以自然地分解为子任务，CoT 有效；如果任务是整体性的，CoT 无效甚至有害。

### 5.6.2 忠实性：推理链是否可信？

CoT 的一个深层问题是**忠实性**（faithfulness）：模型生成的推理链是否真实反映了它的内部推理过程？还是仅仅是答案的"事后合理化"（post-hoc rationalization）？

Turpin et al. [2023] 的实验表明，当 prompt 中的偏见信息被巧妙地插入时（如将选项顺序与正确答案关联），模型在 CoT 中经常引用这些偏见信息，即使它们与正确答案无关。这说明模型在生成推理链时，可能受到各种启发式的影响，而非严格的逻辑推理。

更近期的研究 [Lanham et al., 2023; Uesato et al., 2022] 发现，LLM 的推理链可能与其内部真实推理过程不一致：模型可能先通过某种内部机制得出答案，然后生成的推理链只是将这个答案包装成逻辑形式。如果这一结论成立，那么 CoT 的可靠性将大打折扣——我们无法通过检查推理链来验证答案的正确性。

### 5.6.3 工程启示：如何处理 CoT 的不忠实性

对于 Agent 系统的设计者，CoT 的不忠实性提出了严峻挑战。如果一个 Agent 基于 CoT 做出决策，但 CoT 是不可信的，那么决策的可靠性如何保证？目前行业中的应对策略包括：
- **验证器**：使用外部验证器（如代码执行器、数学求解器）检查 CoT 中的每一步，而非只检查最终答案；
- **工具增强**：在 CoT 中强制使用工具（如计算器、搜索引擎）来获取中间结果，减少模型"编造"的可能性；
- **多轮验证**：让模型生成 CoT 后，再用另一个模型或同一个模型检查 CoT 的逻辑一致性。

---

## 5.7 代码：CoT + Self-Consistency 数学问题求解器

下面的代码实现了一个基于 CoT 和 Self-Consistency 的数学问题求解器。它模拟了 LLM 的推理过程（使用规则系统模拟 LLM API），展示了如何生成多个推理链并取多数投票。

```python
import random
import re
from collections import Counter

class MockLLM:
    """
    模拟 LLM 的推理能力。在真实系统中，这里应替换为 GPT-4/LLaMA 的 API 调用。
    使用规则系统模拟不同难度下的推理准确率。
    """
    def __init__(self, base_accuracy: float = 0.6):
        self.base_accuracy = base_accuracy
    
    def generate_cot(self, problem: str, temperature: float = 0.7) -> str:
        """
        生成 CoT 推理链和答案。模拟 LLM 的推理过程。
        """
        # 解析简单算术问题（如 "37 + 15 = ?"）
        match = re.match(r'What is (\d+)\s*([+\-*/])\s*(\d+)\?\s*', problem)
        if match:
            a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
            
            # 根据 base_accuracy 决定是否正确
            if random.random() < self.base_accuracy:
                # 正确推理
                if op == '+':
                    result = a + b
                    steps = f"Step 1: Identify the operation: {a} + {b}.\n"
                    steps += f"Step 2: Add the units: {a % 10} + {b % 10} = {(a % 10 + b % 10) % 10}, carry {(a % 10 + b % 10) // 10}.\n"
                    steps += f"Step 3: Add the tens: {a // 10} + {b // 10} + carry = {result // 10}.\n"
                    steps += f"Final answer: {result}"
                elif op == '-':
                    result = a - b
                    steps = f"Step 1: Identify the operation: {a} - {b}.\n"
                    steps += f"Step 2: Subtract directly: {a} - {b} = {result}.\n"
                    steps += f"Final answer: {result}"
                elif op == '*':
                    result = a * b
                    steps = f"Step 1: Identify the operation: {a} × {b}.\n"
                    steps += f"Step 2: Compute product: {a} × {b} = {result}.\n"
                    steps += f"Final answer: {result}"
                else:
                    result = a / b if b != 0 else 0
                    steps = f"Step 1: Identify the operation: {a} / {b}.\n"
                    steps += f"Step 2: Compute division: {a} / {b} = {result:.2f}.\n"
                    steps += f"Final answer: {result:.2f}"
            else:
                # 错误推理（常见错误）
                errors = [a + b + 1, a + b - 1, a * b, a - b]
                result = random.choice(errors)
                steps = f"Step 1: I see the numbers {a} and {b}.\n"
                steps += f"Step 2: Let me compute quickly: {a} {op} {b} = {result}.\n"
                steps += f"Final answer: {result}"
            
            return steps
        
        # 多步问题（如 "37 + 15, then multiply by 2"）
        multi_match = re.match(r'What is \((\d+)\s*([+\-*/])\s*(\d+)\)\s*\*\s*(\d+)\?\s*', problem)
        if multi_match:
            a, op, b, c = int(multi_match.group(1)), multi_match.group(2), int(multi_match.group(3)), int(multi_match.group(4))
            if op == '+':
                intermediate = a + b
            elif op == '-':
                intermediate = a - b
            elif op == '*':
                intermediate = a * b
            else:
                intermediate = a / b if b != 0 else 0
            
            if random.random() < self.base_accuracy - 0.1:  # 多步问题准确率稍低
                result = intermediate * c
                steps = f"Step 1: First compute {a} {op} {b} = {intermediate}.\n"
                steps += f"Step 2: Then multiply by {c}: {intermediate} × {c} = {result}.\n"
                steps += f"Final answer: {result}"
            else:
                result = random.choice([intermediate + c, intermediate * c + 1, a * c])
                steps = f"Step 1: Compute {a} {op} {b} = {intermediate}.\n"
                steps += f"Step 2: Multiply by {c}: {intermediate} × {c} = {result}.\n"
                steps += f"Final answer: {result}"
            return steps
        
        return "I cannot solve this problem."
    
    def extract_answer(self, cot_text: str) -> str:
        """从 CoT 中提取最终答案。"""
        # 查找 "Final answer:" 或 "Answer:" 后的数字
        patterns = [
            r'Final answer:\s*([\d.]+)',
            r'Answer:\s*([\d.]+)',
            r'=\s*([\d.]+)\s*$'
        ]
        for pattern in patterns:
            match = re.search(pattern, cot_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "N/A"

class CoTSolver:
    """
    CoT + Self-Consistency 求解器。
    """
    def __init__(self, llm: MockLLM, num_samples: int = 5, temperature: float = 0.7):
        self.llm = llm
        self.num_samples = num_samples
        self.temperature = temperature
    
    def solve(self, problem: str) -> dict:
        """
        使用 CoT + Self-Consistency 求解问题。
        返回: {'answer': str, 'confidence': float, 'all_chains': list}
        """
        chains = []
        answers = []
        
        for _ in range(self.num_samples):
            random.seed()  # 重置随机种子以获取多样性
            chain = self.llm.generate_cot(problem, self.temperature)
            answer = self.llm.extract_answer(chain)
            chains.append(chain)
            answers.append(answer)
        
        # 多数投票（只考虑数值答案）
        valid_answers = [a for a in answers if a != "N/A"]
        if not valid_answers:
            return {'answer': "N/A", 'confidence': 0.0, 'all_chains': chains}
        
        counter = Counter(valid_answers)
        most_common = counter.most_common(1)[0]
        answer, count = most_common
        confidence = count / len(valid_answers)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'all_chains': chains,
            'vote_distribution': dict(counter)
        }

# ============ 演示 ============
if __name__ == "__main__":
    # 创建模拟 LLM（基础准确率 60%）
    llm = MockLLM(base_accuracy=0.6)
    solver = CoTSolver(llm, num_samples=7, temperature=0.8)
    
    problems = [
        "What is 37 + 15?",
        "What is (37 + 15) * 2?"
    ]
    
    for problem in problems:
        print(f"\n{'='*50}")
        print(f"Problem: {problem}")
        result = solver.solve(problem)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Vote distribution: {result['vote_distribution']}")
        print(f"\nSample reasoning chain:")
        print(result['all_chains'][0][:200] + "...")
    
    # 对比：单次推理 vs Self-Consistency
    print(f"\n{'='*50}")
    print("Comparison: Single vs Self-Consistency")
    
    single_correct = 0
    sc_correct = 0
    num_trials = 50
    
    for _ in range(num_trials):
        problem = f"What is {random.randint(10,99)} + {random.randint(10,99)}?"
        a, b = int(problem.split()[2]), int(problem.split()[4])
        true_answer = str(a + b)
        
        # Single shot
        single_chain = llm.generate_cot(problem, temperature=0.0)
        single_answer = llm.extract_answer(single_chain)
        if single_answer == true_answer:
            single_correct += 1
        
        # Self-Consistency
        sc_result = solver.solve(problem)
        if sc_result['answer'] == true_answer:
            sc_correct += 1
    
    print(f"Single-shot accuracy: {single_correct/num_trials:.2%}")
    print(f"Self-Consistency accuracy: {sc_correct/num_trials:.2%}")
    print(f"Improvement: {sc_correct - single_correct} more correct out of {num_trials}")
```

### 代码解析

这个求解器展示了几个关键概念：

1. **CoT 生成**：模拟 LLM 生成逐步推理链。在真实系统中，`generate_cot` 应调用 LLM API（如 GPT-4）并解析返回的文本。
2. **答案提取**：使用正则表达式从 CoT 中提取最终答案。这是 Self-Consistency 的关键步骤——需要统一解析不同格式（"Final answer: 52"、"Answer: 52"、"= 52"）。
3. **多数投票**：对多个采样链的答案进行投票，选择最频繁的答案。如果分布均匀（无明确多数），说明模型对该问题高度不确定。
4. **准确率对比**：通过蒙特卡洛模拟，展示 Self-Consistency 相比单次推理的提升。注意：提升只在模型"勉强正确"（准确率 > 50%）时有效。

---

## 5.8 工程实践框：CoT 在 Agent 系统中的应用

### 常见陷阱

1. **推理链过长导致上下文溢出**：当 CoT 生成数百个步骤时，可能超出模型的上下文窗口限制。解决方案：使用分层 CoT（先生成高层计划，再逐层细化），或在关键步骤后使用总结模型压缩历史。
2. **温度选择的矛盾**：低温度（T=0.1）产生确定性但可能错误的推理链；高温度（T=0.8）产生多样性但可能包含幻觉。Self-Consistency 需要在"多样性"和"质量"之间平衡。经验法则：T=0.5-0.7 是 sweet spot。
3. **工具调用与 CoT 的耦合**：当 Agent 在 CoT 中调用工具（如计算器）时，工具的返回值可能破坏 CoT 的连贯性。例如，模型在推理链中预测"37+15=53"，但工具返回"52"，模型可能无法正确处理这一矛盾。解决方案：在工具返回值与模型预测不一致时，强制模型重新生成该步骤。

### 硬件需求估算

- **单次 CoT 推理**（GPT-4 级别，生成 500 tokens）：约 0.5-1 秒，成本约 $0.03-0.06（取决于 API 定价）；
- **Self-Consistency**（k=10 次采样）：成本和时间均乘以 10，即约 $0.30-0.60 和 5-10 秒。对于实时应用（如客服机器人），这可能太慢。解决方案：使用小模型（如 LLaMA-7B）进行 CoT 推理，成本降低 100 倍，但准确率可能下降；
- **Long CoT 训练**（DeepSeek-R1 风格）：需要数百张 GPU 进行数周的 RL 训练，成本数百万美元。这不适合个人研究者，但可以通过开源模型（如 DeepSeek-R1 开源权重）进行蒸馏学习。

### 超参选择建议

- **Few-shot 示例数量**：通常在 3-8 个示例之间。太少则模型无法学习格式，太多则占用上下文空间。对于复杂任务（如数学竞赛），使用 6-8 个示例；对于简单任务，使用 2-3 个示例。
- **CoT 格式**：实验表明，格式 "Step 1: ... Step 2: ..." 优于自由文本格式。结构化格式帮助模型保持逻辑连贯性。
- **答案格式**：在 CoT 末尾明确标注 "Final answer: [答案]"，便于自动提取和验证。避免让模型在推理链中间直接输出答案。
- **Self-Consistency 的 k 值**：在准确率和成本之间权衡。k=5 是性价比最高的选择（相比 k=1 提升显著，相比 k=10 成本减半）。如果准确率仍然不足，可以增加到 k=10-20，但收益递减。

---

## 5.9 本章小结

本章的核心结论是：Chain-of-Thought 不是让模型变聪明的魔法，而是**结构化解码策略**对预训练模型能力的激活。CoT 的有效性取决于三个因素：
1. **模型规模**：模型需要足够大（通常 > 10B）才能隐含地习得推理模式；
2. **任务类型**：CoT 对可分解的符号推理任务有效，对整体性的常识任务无效；
3. **解码策略**：Self-Consistency 和 Tree of Thoughts 可以进一步提升性能，但增加了计算成本。

但 CoT 的深层问题——**忠实性**——尚未解决。模型生成的推理链可能不是真实推理的反映，而是答案的事后合理化。对于 Agent 系统的设计者，这意味着：不能盲目信任 CoT 的输出，必须引入外部验证机制（如工具执行、形式化验证）来确保推理的可靠性。第6章将讨论如何通过 RLHF 和对齐技术来诱导模型生成更忠实、更安全的推理链，第8章将展示如何在 ReAct Agent 中集成 CoT 和工具调用。

---

## 延伸阅读

1. [Sprague et al., 2024] **MuSR: Testing the limits of chain-of-thought** —— 系统测试了 CoT 在复杂推理任务上的极限，发现 CoT 在需要多步逻辑和数学证明的任务上表现有限。本章对 CoT 边界的分析参考了该文的实验设计。
2. [Chen et al., 2026] **Towards reasoning era: A survey of long CoT** —— 2026 年的 Long CoT 综述，系统梳理了从 few-shot CoT 到 RL 诱导的长程推理的演进。本章 5.5 节的 Long CoT 讨论直接基于该文的分类框架。
3. [Zhang et al., 2023] **Multimodal chain-of-thought reasoning** —— 将 CoT 扩展到视觉-语言推理，展示了如何在多模态场景中使用 CoT 解决视觉问答和视觉推理任务。本章对 CoT 任务依赖性的分析参考了该文的多模态扩展。
4. [Fang et al., 2026] **Thinkless: LLM learns when to think** —— 提出了"选择性推理"框架，让模型自动决定何时使用 CoT（复杂问题）和何时直接回答（简单问题）。本章对 CoT 计算效率的讨论直接受到该文的启发。
5. [Sprague et al., 2025] **To CoT or not to CoT** —— 系统分析了 CoT 在 20 多种任务类型上的效果，建立了任务特征与 CoT 收益之间的定量关系。本章对 CoT 任务依赖性的讨论基于该文的实验结论。

---

> **交叉引用**：本章的 CoT 实现代码（5.7 节）将在第8章的 ReAct Agent 中被扩展为工具增强的推理链。第4章的 Scaling Laws 分析解释了为什么 CoT 需要大模型（>10B）才能有效。第6章的 RLHF 技术可以用来训练模型生成更忠实、更安全的推理链。第7章的多模态模型将 CoT 扩展到视觉推理场景。第12章的 World Model 训练使用类似的逐步预测机制，但目标是在潜在空间而非文本空间。
