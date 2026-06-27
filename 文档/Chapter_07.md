# 第7章：多模态与工具化（视觉语言模型、Function Calling、Code Generation）

## 核心思想

大语言模型的能力边界不仅取决于参数规模，更取决于它能接入的感知模态与外部工具。多模态扩展（将视觉、音频映射到语言空间）为 LLM 提供了"眼睛"，工具化（Function Calling、API 调用）为 LLM 提供了"手"——这两者共同构成了 LLM 从"对话者"变为"行动者"的关键一跃。理解多模态融合架构的 trade-off（投影层 vs 交叉注意力 vs 统一编码器）和工具调用的工程化细节（参数填充、错误恢复、多步调用链），是构建生产级 Agent 系统的必要技能。

---

## 历史脉络：从独立模态到统一行动接口

2019-2020 年，多模态 AI 的主流范式是"双塔架构"：独立的视觉编码器和文本编码器通过对比学习对齐（如 CLIP）。这种架构在图像检索和零样本分类上表现优异，但无法处理复杂的视觉-语言推理任务——例如，回答"图片中有几个人，他们在做什么？"需要理解图像的语义内容和空间关系，而不仅仅是将图像和文本映射到共享空间。

2021-2022 年，BLIP 和 Flamingo 展示了另一种路径：将视觉编码器与语言模型通过**交叉注意力**连接，让语言模型在生成文本时直接"查看"图像特征。这种架构允许模型进行复杂的视觉推理，但训练成本高昂——需要在大规模图像-文本对上进行端到端训练。

2023 年，LLaVA 和 MiniGPT-4 提出了更轻量的方案：使用预训练的视觉编码器（如 CLIP 的 ViT）提取图像特征，然后通过一个简单的**线性投影层**将视觉特征映射到语言模型的输入空间。这种"冻结视觉编码器 + 轻量投影 + 微调语言模型"的策略大幅降低了训练成本，使得学术研究者也能训练视觉语言模型（VLM）。

与此同时，工具化也在平行发展。2022 年的 Toolformer [Schick et al., 2023] 首次证明 LLM 可以在预训练阶段学习调用 API（如搜索引擎、计算器、翻译器）。2023 年，GPT-4 的 Function Calling 能力将工具使用标准化为 JSON 格式的结构化输出。ToolLLM [Qin et al., 2024] 进一步将工具使用扩展到 16000+ 真实 API，并提出了系统化的工具学习框架。Code Generation 则是一个特殊的"工具化"形式：LLM 生成可执行代码，从而间接操控计算环境（如数据分析、可视化、自动化脚本）。

这三条线的交汇点在于：一个能"看见"世界（多模态）、能"使用工具"（工具化）、能"编写代码"（Code Generation）的 LLM，已经具备了 Agent 的基本能力。第8-11章将讨论如何将这些能力组织成系统化的 Agent 架构。

---

## 7.1 多模态融合的三大架构

### 7.1.1 投影层架构（Projection Layer）

投影层架构是 LLaVA 和 MiniGPT-4 采用的方法。其核心思想是：预训练的视觉编码器（如 CLIP ViT）已经提取了高质量的视觉特征，我们只需要将这些特征映射到语言模型的输入空间。具体实现：

1. 用 CLIP ViT 提取图像特征：f_img = ViT(image) ∈ R^{n×d_v}
2. 用线性投影层映射：f_proj = W · f_img ∈ R^{n×d_l}
3. 将投影后的特征作为语言模型的"视觉 token"：input = [text_tokens; f_proj]

投影层架构的优点：
- **训练成本低**：视觉编码器冻结，只需训练投影层和语言模型的 adapter；
- **架构简单**：不需要修改语言模型的内部结构。

缺点：
- **能力受限**：视觉信息只通过线性投影进入语言模型，复杂的视觉-语言交互（如"图像左上角的文字是什么？"）可能处理不好；
- **分辨率限制**：CLIP ViT 通常使用固定分辨率（如 224×224），对于高分辨率图像（如 4K 照片），细节信息丢失严重。

### 7.1.2 交叉注意力架构（Cross-Attention）

交叉注意力架构（如 Flamingo、BLIP-2）在语言模型的 Transformer 层中插入交叉注意力模块，让语言模型的每一层都可以直接 attending 到视觉特征。具体实现：

在 Transformer 的每一层，除了标准的自注意力（Self-Attention），还增加一个交叉注意力（Cross-Attention）：

Q_self = XW_q,  K_self = XW_k,  V_self = XW_v  → Self-Attention
Q_cross = XW_q',  K_cross = F_imgW_k',  V_cross = F_imgW_v'  → Cross-Attention

Output = X + Self-Attention(X) + Cross-Attention(X, F_img)

交叉注意力架构的优点：
- **深层交互**：视觉信息在语言模型的每一层都参与计算，可以捕捉复杂的视觉-语言关系；
- **灵活性强**：可以处理任意数量的视觉 token（如多图像输入）。

缺点：
- **训练成本高**：需要端到端训练视觉编码器和语言模型，或者至少训练交叉注意力模块；
- **推理延迟**：每层额外的交叉注意力增加了计算量。

### 7.1.3 统一编码器架构（Unified Encoder）

统一编码器架构（如 GPT-4V、Gemini）将文本和图像统一处理为 token 序列，用同一个 Transformer 编码。具体实现：

1. 图像被分割为 patch（如 16×16 像素），每个 patch 线性投影为向量；
2. 文本被 tokenize 为嵌入向量；
3. 视觉 token 和文本 token 拼接为一个序列，输入到统一的 Transformer 中。

统一编码器的优点：
- **架构简洁**：不需要额外的交叉注意力或投影层，文本和视觉完全对称处理；
- **天然支持多图像**：多个图像的 patch 可以像文本 token 一样拼接在序列中。

缺点：
- **数据需求大**：统一编码器需要从头学习视觉-语言的交互关系，需要比投影层架构更多的训练数据；
- **计算开销**：视觉 patch 的数量通常很大（如 1024×1024 图像有 4096 个 16×16 patch），导致输入序列很长，注意力计算昂贵。

### 7.1.4 架构选择的工程指南

| 架构 | 适用场景 | 训练成本 | 推理成本 | 数据需求 |
|------|----------|----------|----------|----------|
| 投影层 | 快速原型、资源有限 | 低 | 低 | 中等 |
| 交叉注意力 | 需要复杂视觉推理 | 高 | 高 | 高 |
| 统一编码器 | 追求极致性能、大规模部署 | 极高 | 高 | 极高 |

工程上的共识是：对于大多数应用，投影层架构（如 LLaVA）是性价比最高的选择。只有在需要处理复杂视觉推理（如文档理解、医学图像分析）时，才需要交叉注意力或统一编码器架构。

---

## 7.2 视觉语言模型：从 CLIP 到 LLaVA

### 7.2.1 LLaVA 的训练流程

LLaVA（Large Language and Vision Assistant）的训练分为两个阶段：

**阶段1：预训练投影层**
使用 CC3M 等图像-文本对齐数据集，训练视觉特征到语言模型输入空间的投影层。语言模型和视觉编码器都冻结，只训练投影层。这一步的目标是让视觉 token 和文本 token 在语言模型的输入空间中对齐。

**阶段2：视觉指令微调**
使用视觉指令数据集（如 LLaVA-Instruct-150K），在对话格式中微调语言模型（包括投影层）。这些指令数据包含图像-问题-回答三元组，例如：

[图像：一只猫在垫子上]
Human: 图中有什么动物？
Assistant: 图中有一只猫，它正坐在垫子上。

### 7.2.2 视觉 token 的位置编码问题

当视觉 patch 和文本 token 拼接时，一个关键问题是：如何给视觉 patch 分配位置编码？

方案1：**共享位置编码**：视觉 patch 和文本 token 使用同一套位置编码。但这忽略了视觉的空间结构——图像中相邻的 patch 在空间上相关，但位置编码只编码了序列顺序。

方案2：**2D 位置编码**：为视觉 patch 分配二维位置编码（x, y 坐标），保留空间结构。例如，对于 2×2 的图像 patch，位置编码为 (0,0), (0,1), (1,0), (1,1)。

方案3：**相对位置编码**：使用 RoPE（Rotary Position Embedding）的二维扩展，让模型学习视觉 patch 的相对空间关系。

当前主流方案是方案2（2D 位置编码）或方案3（RoPE 扩展），因为视觉的空间结构对理解至关重要。

---

## 7.3 工具使用：Function Calling 的工程化

### 7.3.1 Function Calling 的协议设计

Function Calling（工具调用）的标准协议包含三个组件：

1. **工具描述**：每个可用工具用 JSON Schema 描述，包含工具名称、参数类型和描述：
```json
{
  "name": "calculator",
  "description": "Perform arithmetic calculations",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {"type": "string", "description": "Math expression to evaluate"}
    },
    "required": ["expression"]
  }
}
```

2. **调用生成**：LLM 在生成文本时，如果判断需要使用工具，则输出一个 JSON 对象：
```json
{"name": "calculator", "arguments": {"expression": "37 + 15"}}
```

3. **结果注入**：执行工具后，将结果注入对话上下文：
```
User: What is 37 + 15?
Assistant: {"name": "calculator", "arguments": {"expression": "37 + 15"}}
System: Tool result: 52
Assistant: The answer is 52.
```

### 7.3.2 工具调用的工程挑战

**参数填充**：LLM 需要准确地将用户请求映射到工具参数。例如，用户说"查一下北京今天的天气"，模型需要提取出 location="北京" 和 date="今天"。这涉及实体识别、指代消解和日期解析。

**错误恢复**：工具调用可能失败（如 API 超时、参数错误、服务不可用）。Agent 需要能够：
- 检测错误（解析 HTTP 状态码或异常信息）；
- 重试（指数退避策略）；
- 降级（如果主工具失败，尝试备用工具）；
- 向用户解释（"天气服务暂时不可用，请稍后再试"）。

**多步调用链**：复杂任务需要多个工具的串联调用。例如："帮我规划一个从北京到上海的周末旅行，预算 2000 元"需要调用：
1. 火车票查询工具（获取车次和价格）；
2. 酒店查询工具（获取住宿选项）；
3. 景点查询工具（获取推荐景点）；
4. 计算器工具（汇总费用并检查预算）。

多步调用链的管理需要**状态机**或**有向图**来跟踪调用进度和依赖关系。

---

## 7.4 Code Generation：特殊的工具化形式

### 7.4.1 代码作为通用工具

代码生成是工具化的一个特殊形式：与其他工具（如搜索引擎、计算器）不同，代码执行器是**通用的**——任何可以用 Python 表达的任务，都可以通过代码生成来解决。这使得代码生成成为 Agent 的"瑞士军刀"。

例如，面对"计算这组数据的均值、中位数和标准差，并绘制直方图"，Agent 可以生成 Python 代码：

```python
import numpy as np
import matplotlib.pyplot as plt

data = [23, 45, 56, 78, 32, 67, 89, 12, 34, 55]
mean = np.mean(data)
median = np.median(data)
std = np.std(data)

plt.hist(data, bins=10)
plt.title(f'Mean={mean:.2f}, Median={median:.2f}, Std={std:.2f}')
plt.savefig('histogram.png')
```

执行这段代码后，Agent 不仅获得了统计结果，还生成了可视化图表。

### 7.4.2 代码执行的安全隔离

代码执行的危险性在于：LLM 生成的代码可能包含恶意操作（如删除文件、访问网络、窃取数据）。因此，代码执行必须在**沙箱环境**中进行：
- **容器隔离**：在 Docker 容器中执行代码，限制文件系统访问；
- **资源限制**：设置 CPU 时间限制（如 5 秒）、内存限制（如 256MB）；
- **网络隔离**：禁止代码访问外部网络；
- **输入消毒**：对用户输入进行消毒，防止代码注入攻击。

---

## 7.5 多模态与工具化的融合挑战

### 7.5.1 视觉-工具联合推理

当 Agent 同时拥有视觉输入和工具使用能力时，新的挑战出现：
- **视觉 grounding**：用户说"点击这个按钮"，Agent 需要理解"这个"指的是图像中的哪个按钮，然后调用点击工具；
- **视觉验证**：Agent 执行操作后，需要通过视觉反馈确认操作是否成功（如"点击保存按钮后，是否出现了保存成功的提示？"）；
- **跨模态一致性**：文本描述和视觉内容可能不一致（如用户说"红色杯子"，但图像中的杯子是蓝色的），Agent 需要处理这种冲突。

### 7.5.2 学术界与工业界的断层

论文中的多模态模型通常使用精心准备的评估数据集（如 VQAv2、OK-VQA），但工业界的真实场景更加混乱：
- 用户上传的图像质量参差不齐（模糊、低分辨率、遮挡）；
- 用户问题含糊不清（如"这个怎么样？"缺乏上下文）；
- 工具 API 的响应格式不一致，需要动态解析。

这些工程问题在论文中很少讨论，但它们是生产系统的核心挑战。第9章将讨论如何用 RAG 和记忆系统来增强 Agent 的工具使用能力，第11章将讨论多模态 Agent 的评估方法。

---

## 7.6 代码：基于 LLM 的 ReAct 风格工具调用循环

下面的代码实现了一个完整的 ReAct 风格工具调用 Agent，支持多种工具（计算器、搜索模拟器、日期查询）。这个实现可以直接扩展到真实 LLM API 和真实工具。

```python
import json
import re
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

class SimpleAgent:
    """
    ReAct 风格工具调用 Agent。
    模拟 LLM 的推理-行动循环，使用规则系统替代真实 LLM API。
    """
    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.memory = []  # 对话历史
    
    def simulate_llm_reasoning(self, query: str, available_tools: List[str]) -> Dict[str, Any]:
        """
        模拟 LLM 的推理过程。在真实系统中，这里调用 LLM API。
        """
        query_lower = query.lower()
        
        # 规则：检测需要计算的问题
        if any(op in query for op in ['+', '-', '*', '/', 'calculate', 'sum', 'compute']):
            # 提取数学表达式
            expr_match = re.search(r'[\d\s+\-*/().]+', query)
            if expr_match:
                expr = expr_match.group().strip()
                return {
                    'thought': f'I need to calculate "{expr}". I will use the calculator tool.',
                    'action': 'calculator',
                    'arguments': {'expression': expr}
                }
        
        # 规则：检测日期查询
        if any(word in query_lower for word in ['date', 'today', 'time', 'day']):
            return {
                'thought': 'The user is asking about the current date. I will use the date tool.',
                'action': 'date_query',
                'arguments': {}
            }
        
        # 规则：检测搜索需求
        if any(word in query_lower for word in ['search', 'find', 'look up', 'information about']):
            # 提取搜索关键词
            keywords = query_lower.replace('search', '').replace('find', '').replace('for', '').strip()
            return {
                'thought': f'I need to search for "{keywords}". I will use the search tool.',
                'action': 'search',
                'arguments': {'query': keywords}
            }
        
        # 默认：直接回答
        return {
            'thought': 'This is a general question. I can answer directly without tools.',
            'action': 'direct_answer',
            'arguments': {'query': query}
        }
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
        
        tool = self.tools[tool_name]
        try:
            result = tool.function(**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询，执行完整的 ReAct 循环。
        """
        # 1. 推理
        reasoning = self.simulate_llm_reasoning(query, list(self.tools.keys()))
        thought = reasoning['thought']
        action = reasoning['action']
        arguments = reasoning['arguments']
        
        # 2. 行动
        if action == 'direct_answer':
            observation = "Answered directly."
        else:
            observation = self.execute_tool(action, arguments)
        
        # 3. 生成最终回答（模拟）
        if action == 'calculator':
            final_answer = f"The result is {observation}."
        elif action == 'date_query':
            final_answer = f"Today is {observation}."
        elif action == 'search':
            final_answer = f"I found the following information: {observation}"
        else:
            final_answer = f"Based on my knowledge: {query}"
        
        # 记录到记忆
        self.memory.append({
            'query': query,
            'thought': thought,
            'action': action,
            'arguments': arguments,
            'observation': observation,
            'answer': final_answer
        })
        
        return {
            'thought': thought,
            'action': action,
            'arguments': arguments,
            'observation': observation,
            'answer': final_answer,
            'memory_size': len(self.memory)
        }

# ============ 工具实现 ============
def calculator(expression: str) -> float:
    """安全计算器：只支持基本算术"""
    # 只允许数字和基本运算符
    allowed = set('0123456789+-*/(). ')
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid characters in expression")
    return eval(expression)

def search_simulator(query: str) -> str:
    """模拟搜索引擎"""
    mock_results = {
        'python': 'Python is a high-level programming language created by Guido van Rossum.',
        'transformer': 'Transformer is a neural network architecture introduced by Vaswani et al. in 2017.',
        'llm': 'LLM stands for Large Language Model, a type of AI model trained on vast text data.'
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': No relevant information found."

def date_query() -> str:
    """返回当前日期"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %A")

# ============ 演示 ============
if __name__ == "__main__":
    # 定义工具
    tools = [
        Tool('calculator', 'Perform arithmetic calculations', 
             {'expression': 'string'}, calculator),
        Tool('search', 'Search for information', 
             {'query': 'string'}, search_simulator),
        Tool('date_query', 'Get current date', 
             {}, date_query)
    ]
    
    # 创建 Agent
    agent = SimpleAgent(tools)
    
    # 测试查询
    queries = [
        "What is 37 + 15 * 2?",
        "What is the date today?",
        "Search for information about Python",
        "Tell me a joke"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"User: {query}")
        result = agent.process(query)
        print(f"Thought: {result['thought']}")
        print(f"Action: {result['action']}")
        print(f"Observation: {result['observation']}")
        print(f"Answer: {result['answer']}")
    
    print(f"\n{'='*50}")
    print(f"Memory recorded {len(agent.memory)} interactions.")
```

### 代码解析

这个 ReAct 工具调用实现展示了几个关键概念：

1. **工具注册**：通过 `Tool` 数据类定义工具的名称、描述、参数模式和执行函数。工具描述在真实系统中会被传递给 LLM，作为 Function Calling 的 schema。
2. **推理模拟**：`simulate_llm_reasoning` 用规则系统模拟 LLM 的决策过程。在真实系统中，这里应调用 LLM API，传入工具描述和当前对话，让 LLM 生成 Thought + Action + Arguments。
3. **安全执行**：`calculator` 工具通过字符白名单和 `eval` 实现安全计算。在真实系统中，应使用更安全的沙箱（如 `ast.literal_eval` 或 Docker 容器）。
4. **错误恢复**：`execute_tool` 捕获异常并返回错误信息，Agent 可以在下一轮推理中处理错误（如重试或向用户解释）。
5. **记忆管理**：每次交互记录到 `memory` 中，支持多轮对话的上下文保持。

---

## 7.7 工程实践框：多模态与工具化的部署要点

### 常见陷阱

1. **视觉输入的分辨率陷阱**：许多 VLM 使用 224×224 或 336×336 的固定输入分辨率。如果用户上传 4K 图像，直接 resize 会导致文字模糊、小物体消失。解决方案：使用**图像分块**（将大图切分为多个子图分别编码）或**高分辨率适配器**（如 LLaVA-1.5 的 AnyRes）。
2. **工具描述的上下文占用**：当可用工具很多时（如 100+ 个 API），工具描述会占用大量上下文空间，导致用户输入空间被压缩。解决方案：使用**工具检索**（根据用户查询动态选择最相关的工具，只传入这些工具的描述）或**工具分层**（将工具分类，先选择类别再选择具体工具）。
3. **Function Calling 的 JSON 解析失败**：LLM 生成的 JSON 可能包含语法错误（如缺少引号、多余的逗号）。解决方案：使用**JSON 修复库**（如 `json_repair`）或**结构化输出**（如 OpenAI 的 JSON mode、Outlines 库）。
4. **多模态的 token 成本**：视觉 patch 的数量通常很大（如 336×336 图像用 14×14 patch，有 576 个视觉 token）。如果按 token 计费，视觉输入的成本可能是文本的 10-100 倍。解决方案：**压缩视觉 token**（如使用视觉摘要模型提取关键 patch）或**降低图像分辨率**（对于简单任务，224×224 通常足够）。

### 硬件需求估算

- **VLM 推理（LLaVA-7B，图像 336×336）**：单张 RTX 4090 可运行，显存占用约 14-18 GB（7B 参数 + 视觉编码器 + KV cache）。批量推理时，显存随 batch size 线性增长；
- **VLM 微调（LLaVA-7B）**：单张 A100 40GB 可训练投影层，全参数微调需要 2-4 张 A100；
- **工具调用 Agent（GPT-4 API）**：每次工具调用增加约 500-2000 个 token 的上下文（工具描述 + 参数 + 结果）。对于多步任务，成本随步骤数线性增长。经验法则：简单工具调用（1-2 步）成本约 $0.01-0.05，复杂任务（5-10 步）成本约 $0.10-0.50；
- **Code Generation 执行**：沙箱执行 Python 代码需要约 100-500 MB 内存和 1-5 秒 CPU 时间。对于需要数据分析（如 Pandas 处理 1GB CSV）的任务，需要 2-4 GB 内存和 10-30 秒。

### 超参选择建议

- **VLM 的图像分辨率**：简单分类/描述任务用 224×224；需要读取文字的文档理解用 336×336 或更高；需要精细视觉推理（如医学图像）用 448×448 或分块处理。
- **工具调用的温度参数**：工具选择阶段用低温度（T=0.1-0.3）确保参数填充的准确性；工具结果总结阶段用高温度（T=0.7-0.9）增加回答的自然性。
- **上下文窗口管理**：当对话历史 + 工具描述 + 工具结果超过上下文限制时，使用**摘要策略**（对历史对话进行摘要，保留关键信息）或**滑动窗口**（只保留最近 N 轮对话）。
- **重试策略**：工具调用失败时，使用指数退避重试（间隔 1s, 2s, 4s, 8s），最多重试 3-5 次。如果仍然失败，向用户返回错误信息并提供替代方案。

---

## 7.8 本章小结

本章的核心结论是：多模态扩展和工具化是 LLM 能力边界的**关键扩展维度**，而不是次要的附加功能。一个只能处理文本的 LLM，其能力上限是文本知识的边界；而一个能处理图像、能调用工具、能编写代码的 LLM，其能力上限是外部世界和计算环境的边界。

多模态融合的三大架构（投影层、交叉注意力、统一编码器）代表了不同的 trade-off：投影层低成本、交叉注意力高灵活性、统一编码器极致性能。在工程实践中，投影层架构（如 LLaVA）是大多数场景的最佳选择。工具化则涉及更广泛的工程问题：Function Calling 的协议设计、参数填充、错误恢复、多步调用链和成本管理。

Code Generation 是多模态与工具化的交汇点：代码既是文本（可以用 LLM 生成），又是工具（可以执行并改变环境）。这种双重身份使得 Code Generation 成为 Agent 最强大的通用工具。第8章将讨论如何将这些能力组织成系统化的 Agent 架构（从 ReAct 到 AutoGen），第9章将深入工具使用、记忆与规划的细节。

---

## 延伸阅读

1. [Yin et al., 2024] **A survey on multimodal large language models** —— 最全面的多模态 LLM 综述，覆盖了视觉、音频、视频等模态的融合架构。本章的多模态分类框架直接基于该文的分类体系。
2. [Qin et al., 2024] **ToolLLM: Facilitating large language models to master 16000+ real-world APIs** —— 展示了大规模工具学习的可行性，提出了系统化的工具调用训练框架。本章对工具化的工程讨论参考了该文的技术细节。
3. [Wu et al., 2024] **VisionLLM v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks** —— 提出了通用视觉语言模型架构，支持数百种视觉-语言任务。本章对多模态架构的分析参考了该文的设计思路。
4. [Ye et al., 2024] **mPLUG-Owl2: Revolutionizing multimodal large language model with modality collaboration** —— 提出了模态协作的多模态模型架构，通过共享注意力机制实现更深层的视觉-语言交互。本章对交叉注意力架构的讨论参考了该文的技术方案。
5. [Cui et al., 2024] **A survey on multimodal LLMs for autonomous driving** —— 将多模态 LLM 应用于自动驾驶场景，展示了视觉-语言-动作融合的具体工程方案。本章对多模态工具化应用的讨论参考了该文的场景分析。

---

> **交叉引用**：本章的 ReAct 工具调用代码（7.6 节）将在第8章被扩展为完整的 Agent 架构。第3章的 CLIP 实现是本章视觉语言模型的底层组件。第6章的对齐技术（DPO/RLHF）可以应用于多模态模型的安全对齐（如防止生成有害图像描述）。第9章将深入讨论工具使用中的记忆和规划问题。第14章的三者融合将综合运用本章的多模态和工具化技术。
