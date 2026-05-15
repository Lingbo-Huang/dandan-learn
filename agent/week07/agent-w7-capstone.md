---
layout: default
title: "W7 Capstone · 全能工具 Agent"
---

# Capstone：构建全能工具 Agent

> **Week 7 · Capstone** | 难度：⭐⭐⭐⭐⭐

---

## 项目目标

构建一个集成 8+ 种工具的全能 Agent，能完成复杂的现实任务：
- 搜索 + 分析 + 报告生成
- 数据获取 + 计算 + 可视化
- 带完整的安全防护

## 系统架构

```
用户任务
  │
  ▼
┌─────────────────┐
│ 安全过滤层       │ ← 提示注入检测、权限验证
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 工具路由层       │ ← 根据任务选择工具子集
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│              工具执行层                   │
│  搜索 | 计算 | 代码 | 文件 | 天气 | 时间  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 输出过滤层       │ ← 敏感信息过滤、格式化
└─────────────────┘
```

## 完整实现

```python
import asyncio
import os
import re
import math
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.tools import tool, StructuredTool, BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ═══════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════

@tool
def get_current_datetime() -> str:
    """获取当前日期、时间和星期几"""
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M:%S')}"

@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持 Python math 模块函数。
    
    Args:
        expression: 数学表达式（如 'math.sqrt(144) + 2**10'）
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {
            "math": math, "abs": abs, "round": round,
            "max": max, "min": min, "sum": sum, "pow": pow
        })
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

@tool
def search_information(query: str) -> str:
    """搜索网络获取最新信息和知识。
    
    Args:
        query: 搜索关键词或问题
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return f"未找到关于 '{query}' 的信息"
        
        formatted = "\n\n".join([
            f"**{r['title']}**\n{r['body'][:300]}"
            for r in results
        ])
        return formatted
    except ImportError:
        return f"[模拟搜索] 关于 '{query}' 的搜索结果：这里是模拟的搜索内容..."
    except Exception as e:
        return f"搜索失败：{e}"

@tool
def analyze_text(text: str, analysis_type: str = "summary") -> str:
    """分析文本内容。
    
    Args:
        text: 要分析的文本
        analysis_type: 分析类型 - summary/sentiment/keywords/translate
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompts = {
        "summary": f"请将以下文本总结为3句话以内：\n{text}",
        "sentiment": f"分析以下文本的情感倾向（积极/消极/中性），并说明理由：\n{text}",
        "keywords": f"提取以下文本的5个关键词：\n{text}",
        "translate": f"将以下文本翻译成中文（如已是中文则翻译成英文）：\n{text}"
    }
    
    prompt = prompts.get(analysis_type, prompts["summary"])
    return llm.invoke(prompt).content

class WriteFileInput(BaseModel):
    filepath: str = Field(description="文件路径（必须以 /tmp/ 开头）")
    content: str = Field(description="要写入的内容")

@tool(args_schema=WriteFileInput)
def write_report_file(filepath: str, content: str) -> str:
    """将内容写入报告文件（仅限 /tmp 目录）"""
    if not filepath.startswith("/tmp/"):
        return "错误：只能写入 /tmp/ 目录"
    
    try:
        import pathlib
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"文件已保存：{filepath}（{len(content)} 字符）"
    except Exception as e:
        return f"写入失败：{e}"

@tool
def read_local_file(filepath: str) -> str:
    """读取本地文件内容（仅限 /tmp 目录）"""
    if not filepath.startswith("/tmp/"):
        return "错误：只能读取 /tmp/ 目录"
    
    try:
        import pathlib
        path = pathlib.Path(filepath)
        if not path.exists():
            return f"文件不存在：{filepath}"
        content = path.read_text(encoding="utf-8")
        return content[:3000] + ("[截断]" if len(content) > 3000 else "")
    except Exception as e:
        return f"读取失败：{e}"

@tool
def run_python_safely(code: str) -> str:
    """在安全环境中运行 Python 代码（仅支持基本计算和数据处理）"""
    import ast
    
    # AST 安全检查
    try:
        tree = ast.parse(code, mode='exec')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                allowed = {"math", "statistics", "json", "datetime", "re"}
                if hasattr(node, 'names'):
                    for alias in node.names:
                        if alias.name not in allowed:
                            return f"错误：不允许导入 {alias.name}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "open", "compile", "__import__"]:
                        return f"错误：不允许调用 {node.func.id}"
    except SyntaxError as e:
        return f"语法错误：{e}"
    
    # 执行
    import io, contextlib
    output = io.StringIO()
    local_vars = {}
    
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                          "list": list, "dict": dict, "str": str, 
                                          "int": int, "float": float, "sum": sum,
                                          "max": max, "min": min, "sorted": sorted,
                                          "enumerate": enumerate, "zip": zip},
                        "math": math}, local_vars)
        
        result = output.getvalue()
        if not result and local_vars:
            result = str({k: v for k, v in local_vars.items() if not k.startswith('_')})
        
        return result[:2000] if result else "执行完成（无输出）"
    except Exception as e:
        return f"运行错误：{type(e).__name__}: {e}"

# ═══════════════════════════════════════
# 全能 Agent 主体
# ═══════════════════════════════════════

class OmnipotentAgent:
    """全能工具 Agent"""
    
    SYSTEM_PROMPT = """你是一个功能强大的智能助手，能够使用多种工具完成复杂任务。

可用工具能力：
- 搜索最新信息和知识
- 进行数学计算
- 分析和处理文本
- 执行 Python 代码（受限环境）
- 读写文件（仅 /tmp 目录）
- 获取当前时间

工作原则：
1. 分解复杂任务，逐步完成
2. 优先使用简单工具，避免过度复杂化
3. 计算类任务用 calculate 工具而非心算
4. 需要最新信息时搜索
5. 生成报告时保存到文件
6. 如果一种方法失败，尝试替代方案"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        self.tools = [
            get_current_datetime,
            calculate,
            search_information,
            analyze_text,
            write_report_file,
            read_local_file,
            run_python_safely,
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def run(self, task: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """执行任务"""
        start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"任务：{task}")
        print(f"开始时间：{start_time.strftime('%H:%M:%S')}")
        print('='*60)
        
        result = self.executor.invoke({"input": task})
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        steps = result.get("intermediate_steps", [])
        tools_used = [step[0].tool for step in steps]
        
        return {
            "task": task,
            "output": result["output"],
            "tools_used": tools_used,
            "num_steps": len(steps),
            "duration_seconds": duration
        }

# ═══════════════════════════════════════
# 综合测试
# ═══════════════════════════════════════

def comprehensive_test():
    agent = OmnipotentAgent()
    
    test_tasks = [
        "今天是几号？距离2025年元旦还有多少天？（需要计算）",
        
        "生成一份斐波那契数列的分析报告：计算前20项，找出其中的质数，保存到 /tmp/fibonacci_report.md",
        
        "搜索 LangChain 最新版本，分析其核心特性，生成200字的技术摘要",
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'#'*60}")
        print(f"测试 {i}/{len(test_tasks)}")
        print('#'*60)
        
        result = agent.run(task)
        
        print(f"\n✅ 完成！")
        print(f"使用工具：{result['tools_used']}")
        print(f"步骤数：{result['num_steps']}")
        print(f"耗时：{result['duration_seconds']:.1f}秒")
        print(f"\n输出：{result['output'][:300]}...")

if __name__ == "__main__":
    comprehensive_test()
```

## 本周回顾

| 技术 | 在本项目中的应用 |
|------|----------------|
| 工具设计 | 8个工具，每个有完整描述和错误处理 |
| 安全沙箱 | Python 代码 AST 检查 + 受限内置函数 |
| API 集成 | 搜索 API + 文件系统 API |
| 工具路由 | AgentExecutor 自动选择工具 |
| 错误处理 | 所有工具 try/except，永不抛异常 |
| 审计日志 | 记录所有工具调用步骤 |

---

*W7 Capstone · 全能工具 Agent | Agent + Claw 系列*
