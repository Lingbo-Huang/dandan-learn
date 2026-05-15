---
layout: default
title: "W7D6 · 工具链组合与编排"
---

# 工具链编排：让工具协同工作

> **Week 7 · Day 6** | 难度：⭐⭐⭐⭐

---

## 工具链的设计模式

### 模式1：顺序工具链

```
搜索 → 提取内容 → 分析 → 生成报告
```

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import List

class SequentialToolChain:
    """顺序工具链：上一步输出作为下一步输入"""
    
    def __init__(self, tools: List, llm: ChatOpenAI):
        self.tools = tools
        self.llm = llm
        self.results = []
    
    def run(self, initial_input: str) -> str:
        current_input = initial_input
        
        for tool in self.tools:
            print(f"执行工具：{tool.name}")
            result = tool.invoke({"input": current_input})
            self.results.append({"tool": tool.name, "input": current_input, "output": result})
            current_input = result  # 输出作为下一步输入
        
        return current_input

# 研究报告生成链：搜索 → 摘要 → 格式化
search_tool = search_web  # 来自前面的定义
```

### 模式2：条件分支工具链

```python
from typing import Callable

class ConditionalToolChain:
    """条件分支工具链"""
    
    def __init__(self, tools: dict, router: Callable):
        self.tools = tools  # {"name": tool}
        self.router = router  # 决定用哪个工具的函数
    
    def run(self, query: str) -> str:
        tool_name = self.router(query)
        tool = self.tools.get(tool_name)
        
        if not tool:
            return f"没有找到合适的工具处理：{query}"
        
        print(f"路由到工具：{tool_name}")
        return tool.invoke({"query": query})

# 路由器：根据查询类型选择工具
def query_router(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""分析以下查询，选择最合适的工具：

查询：{query}

工具选项：
- web_search：需要搜索最新信息、新闻、实时数据
- calculate：纯数学计算
- database：查询内部业务数据
- code_execute：需要运行代码

只回答工具名称："""
    
    return llm.invoke(prompt).content.strip().lower()
```

### 模式3：并行工具链 + 聚合

```python
import asyncio

class ParallelToolChain:
    """并行执行多个工具，聚合结果"""
    
    def __init__(self, tools: List, aggregator_llm: ChatOpenAI):
        self.tools = tools
        self.aggregator = aggregator_llm
    
    async def run(self, query: str) -> str:
        # 并行执行所有工具
        async def run_tool(tool):
            try:
                result = tool.invoke({"query": query})
                return {"tool": tool.name, "result": result, "success": True}
            except Exception as e:
                return {"tool": tool.name, "result": str(e), "success": False}
        
        tasks = [run_tool(tool) for tool in self.tools]
        results = await asyncio.gather(*tasks)
        
        # 聚合成功的结果
        successful = [r for r in results if r["success"]]
        
        if not successful:
            return "所有工具都失败了"
        
        results_text = "\n\n".join([
            f"来源（{r['tool']}）：\n{r['result']}"
            for r in successful
        ])
        
        aggregate_prompt = f"""综合以下不同来源的信息，回答问题：

问题：{query}

{results_text}

综合答案："""
        
        return self.aggregator.invoke(aggregate_prompt).content
```

## 完整工具编排示例

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

class SmartToolAgent:
    """智能工具编排 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 初始化工具集
        self.tools = self._build_toolset()
        
        # 创建 Agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个功能强大的 Agent，能够使用多种工具完成任务。

工具使用原则：
1. 优先使用最简单有效的工具
2. 避免重复调用相同工具
3. 工具失败时尝试替代方案
4. 不要暴露用户的个人信息给外部工具"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _build_toolset(self) -> List:
        """构建工具集"""
        tools = []
        
        # 搜索工具
        @tool
        def web_search(query: str) -> str:
            """搜索网络获取最新信息"""
            return _search_duckduckgo(query)
        
        # 计算工具
        @tool
        def math_calc(expression: str) -> str:
            """执行数学计算"""
            import math
            try:
                result = eval(expression, {"__builtins__": {}}, {"math": math})
                return str(result)
            except Exception as e:
                return f"计算错误：{e}"
        
        # 时间工具
        @tool
        def get_current_time() -> str:
            """获取当前日期和时间"""
            return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        # 文本处理工具
        @tool
        def summarize_text(text: str, max_words: int = 100) -> str:
            """将长文本总结为指定长度的摘要"""
            if len(text) <= max_words * 5:
                return text
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = f"将以下文本总结为不超过{max_words}字的摘要：\n\n{text}"
            return llm.invoke(prompt).content
        
        tools.extend([web_search, math_calc, get_current_time, summarize_text])
        return tools
    
    def run(self, task: str) -> dict:
        result = self.executor.invoke({"input": task})
        return {
            "output": result["output"],
            "steps": len(result.get("intermediate_steps", [])),
            "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])]
        }

# 测试
agent = SmartToolAgent()
result = agent.run("现在是几点？计算今天从0点到现在经过了多少分钟？")
print(f"答案：{result['output']}")
print(f"使用了 {result['steps']} 步，工具：{result['tools_used']}")
```

## 踩坑经验

### 坑1：工具链中某步骤失败，整链崩溃

**解法**：每个工具都返回字符串（包括错误信息），让 Agent 决定如何继续。

### 坑2：并行工具结果矛盾

**问题**：两个工具返回了相互矛盾的信息，Agent 不知道该信任谁。  
**解法**：在聚合阶段明确说明"如果信息有矛盾，优先信任来源更可靠的（如官方数据库）"。

---

*W7D6 · 工具链组合与编排 | Agent + Claw 系列*
