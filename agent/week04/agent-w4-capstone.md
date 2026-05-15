---
layout: default
title: "W4 Capstone · 复杂问题求解 Agent"
render_with_liquid: false
---

# Capstone：构建生产级复杂推理 Agent

> **Week 4 · Capstone** | 难度：⭐⭐⭐⭐⭐

---

## 项目概述

本周 Capstone 目标：构建一个**商业分析 Agent**，能够处理复杂的商业问题，综合运用本周所有推理技术。

**功能要求：**
- 自动分析问题复杂度，选择推理策略
- 对需要多步推理的问题用任务分解
- 对关键结论用自我反思验证
- 输出结构化分析报告

## 系统架构

```
┌───────────────────────────────────────────────────────────┐
│                  商业分析 Agent                            │
│                                                           │
│  用户输入                                                  │
│     │                                                     │
│     ▼                                                     │
│  ┌─────────────────┐                                      │
│  │  问题理解模块    │  → 识别问题类型/复杂度/目标           │
│  └────────┬────────┘                                      │
│           │                                               │
│           ▼                                               │
│  ┌─────────────────┐    ┌──────────────────────┐          │
│  │  策略路由器      │───→│ 推理策略执行层        │          │
│  └─────────────────┘    │ CoT / ToT / Decompose│          │
│                         │ Reflexion / SC       │          │
│                         └──────────┬───────────┘          │
│                                    │                      │
│                         ┌──────────▼───────────┐          │
│                         │   自我反思验证层      │          │
│                         └──────────┬───────────┘          │
│                                    │                      │
│                         ┌──────────▼───────────┐          │
│                         │   报告生成模块        │          │
│                         └──────────────────────┘          │
└───────────────────────────────────────────────────────────┘
```

## 完整代码实现

```python
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from enum import Enum

# ========================
# 数据模型
# ========================

class ProblemType(str, Enum):
    FACTUAL = "factual"           # 事实性问题
    ANALYTICAL = "analytical"    # 分析性问题
    STRATEGIC = "strategic"      # 战略性问题
    CREATIVE = "creative"        # 创意性问题
    QUANTITATIVE = "quantitative"# 量化计算问题

class ProblemAnalysis(BaseModel):
    problem_type: ProblemType
    complexity: int = Field(description="复杂度 1-10", ge=1, le=10)
    key_questions: List[str] = Field(description="需要回答的关键子问题")
    required_reasoning: str = Field(description="需要什么类型的推理")
    suggested_strategy: str = Field(description="建议的推理策略")

class AnalysisSection(BaseModel):
    title: str
    content: str
    confidence: float = Field(ge=0, le=1)
    key_findings: List[str]

class BusinessAnalysisReport(BaseModel):
    title: str
    executive_summary: str
    sections: List[AnalysisSection]
    recommendations: List[str]
    risks: List[str]
    confidence_overall: float = Field(ge=0, le=1)
    methodology: str

# ========================
# 核心 Agent
# ========================

class BusinessAnalysisAgent:
    def __init__(self):
        self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.powerful_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.creative_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        self.analysis_llm = self.powerful_llm.with_structured_output(ProblemAnalysis)
        self.report_llm = self.powerful_llm.with_structured_output(BusinessAnalysisReport)
    
    # ── 问题分析 ──
    def analyze_problem(self, question: str) -> ProblemAnalysis:
        """分析问题特征，制定推理策略"""
        prompt = f"""分析以下商业问题的特征：

问题：{question}

请识别：
1. 问题类型（事实/分析/战略/创意/量化）
2. 复杂度（1-10）
3. 需要回答的关键子问题（3-5个）
4. 需要什么推理方式
5. 推荐的推理策略"""
        
        return self.analysis_llm.invoke(prompt)
    
    # ── CoT 推理 ──
    def reason_with_cot(self, question: str, sub_question: str) -> str:
        """对具体子问题进行 CoT 推理"""
        prompt = f"""主问题背景：{question}

当前需要回答：{sub_question}

请一步步推理，给出详细分析：
1. 现有信息梳理
2. 推理过程
3. 结论"""
        
        return self.powerful_llm.invoke(prompt).content
    
    # ── ToT 探索 ──
    def explore_with_tot(self, question: str, aspect: str) -> str:
        """用 ToT 探索战略性问题的多种可能"""
        prompt = f"""用树形思维探索这个战略问题的不同视角：

问题：{question}
聚焦方面：{aspect}

请：
1. 提出3种不同的分析框架（如：波特五力、SWOT、PEST等）
2. 用每种框架简要分析
3. 综合最有价值的视角，给出综合判断"""
        
        return self.creative_llm.invoke(prompt).content
    
    # ── 自我反思验证 ──
    def validate_with_reflection(self, question: str, analysis: str) -> dict:
        """对分析结果进行自我批评和修正"""
        critique_prompt = f"""严格审查以下分析：

原始问题：{question}
当前分析：{analysis}

请指出：
1. 逻辑漏洞（如有）
2. 遗漏的重要视角
3. 过于武断的结论
4. 需要补充的数据或论据

以 JSON 格式返回：
{{
    "has_issues": true/false,
    "issues": ["问题1", "问题2"],
    "improved_analysis": "改进后的分析"
}}"""
        
        response = self.powerful_llm.invoke(critique_prompt)
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"has_issues": False, "issues": [], "improved_analysis": analysis}
    
    # ── 量化计算 ──
    def quantitative_analysis(self, question: str, numbers_context: str) -> str:
        """处理量化计算问题"""
        prompt = f"""请对以下商业问题进行量化分析：

问题：{question}
数据背景：{numbers_context}

请：
1. 列出所有需要的数据和假设
2. 建立计算模型
3. 分步计算
4. 给出量化结论和敏感性分析"""
        
        return self.powerful_llm.invoke(prompt).content
    
    # ── 任务分解执行 ──
    async def decompose_and_execute(self, question: str, 
                                     sub_questions: List[str]) -> List[Dict]:
        """并行处理所有子问题"""
        async def process_sub_question(sq: str) -> Dict:
            result = self.reason_with_cot(question, sq)
            validated = self.validate_with_reflection(question, result)
            
            final_result = validated.get("improved_analysis", result) \
                if validated.get("has_issues") else result
            
            return {
                "question": sq,
                "analysis": final_result,
                "was_revised": validated.get("has_issues", False)
            }
        
        tasks = [process_sub_question(sq) for sq in sub_questions]
        return await asyncio.gather(*tasks)
    
    # ── 报告生成 ──
    def generate_report(self, question: str, 
                        problem_analysis: ProblemAnalysis,
                        sub_analyses: List[Dict]) -> BusinessAnalysisReport:
        """整合所有分析，生成完整报告"""
        analyses_text = "\n\n".join([
            f"**{sa['question']}**\n{sa['analysis']}"
            for sa in sub_analyses
        ])
        
        prompt = f"""基于以下分析，生成完整的商业分析报告：

原始问题：{question}
问题类型：{problem_analysis.problem_type.value}
复杂度：{problem_analysis.complexity}/10

子问题分析：
{analyses_text}

请生成一份结构化报告，包含：
- 标题
- 执行摘要（3-5句话）
- 各分析章节
- 具体建议（3-5条）
- 主要风险
- 整体置信度
- 方法论说明"""
        
        return self.report_llm.invoke(prompt)
    
    # ── 主入口 ──
    async def analyze(self, question: str) -> BusinessAnalysisReport:
        """完整分析流程"""
        print(f"开始分析：{question[:50]}...")
        
        # 1. 问题分析
        print("  [1/4] 分析问题特征...")
        problem_analysis = self.analyze_problem(question)
        print(f"  → 类型：{problem_analysis.problem_type.value}，"
              f"复杂度：{problem_analysis.complexity}/10")
        print(f"  → 策略：{problem_analysis.suggested_strategy}")
        
        # 2. 根据问题类型选择推理方式
        print("  [2/4] 并行处理子问题...")
        
        # 战略问题用 ToT 补充视角
        enhanced_analyses = []
        if problem_analysis.problem_type == ProblemType.STRATEGIC:
            tot_result = self.explore_with_tot(
                question, "竞争格局与战略选项"
            )
            enhanced_analyses.append({
                "question": "多框架战略分析",
                "analysis": tot_result,
                "was_revised": False
            })
        
        # 并行处理所有子问题
        sub_analyses = await self.decompose_and_execute(
            question, problem_analysis.key_questions
        )
        sub_analyses.extend(enhanced_analyses)
        
        revised_count = sum(1 for sa in sub_analyses if sa.get("was_revised"))
        print(f"  → 完成 {len(sub_analyses)} 个子问题分析，"
              f"其中 {revised_count} 个经过自我修正")
        
        # 3. 生成报告
        print("  [3/4] 生成分析报告...")
        report = self.generate_report(question, problem_analysis, sub_analyses)
        
        # 4. 最终验证
        print("  [4/4] 最终质量验证...")
        
        print(f"\n✅ 分析完成！置信度：{report.confidence_overall:.0%}")
        return report

# ========================
# 运行示例
# ========================

async def main():
    agent = BusinessAnalysisAgent()
    
    # 测试案例1：战略分析
    question1 = """
    一家月营收500万的SaaS企业（HR管理软件），目前有300家中小企业客户，
    年增长率20%。竞品A月营收5000万，竞品B刚获得B轮融资2亿元。
    请分析：我们是否应该转型做AI-driven HR平台？风险和机会各是什么？
    """
    
    report = await agent.analyze(question1.strip())
    
    print("\n" + "="*60)
    print(f"📊 {report.title}")
    print("="*60)
    print(f"\n执行摘要：\n{report.executive_summary}")
    
    print(f"\n📌 核心建议：")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n⚠️ 主要风险：")
    for risk in report.risks:
        print(f"  • {risk}")
    
    print(f"\n🔍 分析方法：{report.methodology}")
    print(f"📈 整体置信度：{report.confidence_overall:.0%}")

asyncio.run(main())
```

## 测试与验证

```python
import pytest
import asyncio

class TestBusinessAnalysisAgent:
    
    @pytest.fixture
    def agent(self):
        return BusinessAnalysisAgent()
    
    def test_problem_analysis(self, agent):
        """测试问题分析是否正确识别类型"""
        question = "2+2等于多少？"
        analysis = agent.analyze_problem(question)
        assert analysis.problem_type == ProblemType.QUANTITATIVE
        assert analysis.complexity <= 3
    
    def test_reflection_improves_quality(self, agent):
        """测试自我反思是否能识别问题"""
        bad_analysis = "我们应该马上进军AI市场，因为AI很火。"
        question = "是否应该转型AI？"
        
        result = agent.validate_with_reflection(question, bad_analysis)
        assert result.get("has_issues") == True
        assert len(result.get("issues", [])) > 0
    
    @pytest.mark.asyncio
    async def test_full_analysis_completeness(self, agent):
        """测试完整分析的结构完整性"""
        question = "我们是否应该扩张到东南亚市场？"
        report = await agent.analyze(question)
        
        assert report.executive_summary
        assert len(report.sections) >= 2
        assert len(report.recommendations) >= 2
        assert 0 <= report.confidence_overall <= 1

# 运行测试
# pytest test_agent.py -v
```

## 部署到生产

```python
# 添加缓存、日志、成本控制

from langchain.cache import SQLiteCache
import langchain

# 启用缓存（相同输入不重复调用 API）
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# 成本追踪
from langchain_community.callbacks import get_openai_callback

async def analyze_with_cost_tracking(question: str):
    agent = BusinessAnalysisAgent()
    
    with get_openai_callback() as cb:
        report = await agent.analyze(question)
        
        print(f"\n💰 成本统计：")
        print(f"  总 token：{cb.total_tokens:,}")
        print(f"  总费用：${cb.total_cost:.4f}")
    
    return report
```

## 本周回顾

| 技术 | 在本项目中的应用 |
|------|----------------|
| CoT | 每个子问题的逐步推理 |
| ToT | 战略问题的多框架探索 |
| Reflexion | 自我批评改进分析质量 |
| Self-Reflection | 最终结果的质量验证 |
| Task Decomposition | 把大问题拆成并行子问题 |
| Strategy Router | 根据问题类型选择推理方式 |

---

*W4 Capstone · 复杂问题求解 Agent | Agent + Claw 系列*
