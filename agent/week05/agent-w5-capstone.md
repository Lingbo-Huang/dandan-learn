---
layout: default
title: "W5 Capstone · 多 Agent 内容生产系统"
---

# Capstone：构建多 Agent 内容生产系统

> **Week 5 · Capstone** | 难度：⭐⭐⭐⭐⭐

---

## 项目目标

构建一个完整的多 Agent 内容生产系统，能够：
- 接收主题，自动生产高质量技术文章
- 多 Agent 协作：研究 → 提纲 → 撰写 → 审核 → 优化
- 全链路追踪和质量监控

## 系统架构

```
用户输入主题
     │
     ▼
┌─────────────┐
│ Coordinator │  ← 协调者：分配任务，监控进度
└──────┬──────┘
       │
   ┌───┴────────────────────┐
   │        工作流           │
   ▼                        │
┌──────────┐                │
│Researcher│ → 市场调研      │
└──────────┘                │
       │                    │
       ▼                    │
┌──────────┐                │
│ Outliner │ → 生成提纲      │
└──────────┘                │
       │                    │
       ▼                    │
┌──────────┐                │
│  Writer  │ → 撰写初稿      │
└──────────┘                │
       │                    │
       ▼                    │
┌──────────┐                │
│  Critic  │ → 批评审核   ──→│（不通过返回Writer）
└──────────┘                │
       │（通过）              │
       ▼                    │
┌──────────┐                │
│ SEO Agent│ → SEO优化       │
└──────────┘                │
       │                    │
       ▼                    │
   最终输出                  │
```

## 完整实现

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import time

# ── 数据模型 ──

class OutlineSection(BaseModel):
    title: str
    key_points: List[str]
    estimated_words: int

class ArticleOutline(BaseModel):
    article_title: str
    target_audience: str
    estimated_total_words: int
    sections: List[OutlineSection]
    seo_keywords: List[str]

class QualityCheck(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=10)
    issues: List[str]
    suggestions: List[str]

# ── Agent 类 ──

class ResearcherAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def research(self, topic: str) -> str:
        prompt = f"""作为研究员，为以下主题提供全面的背景信息：

主题：{topic}

请提供：
1. 核心概念解释
2. 最新发展趋势（2023-2024）
3. 主要技术挑战
4. 实际应用案例
5. 相关工具和框架

输出格式：结构化的研究报告"""
        
        return self.llm.invoke(prompt).content

class OutlinerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.outline_llm = self.llm.with_structured_output(ArticleOutline)
    
    def create_outline(self, topic: str, research: str) -> ArticleOutline:
        prompt = f"""基于研究报告，为技术文章创建详细提纲：

主题：{topic}

研究报告：
{research[:2000]}

请创建一个适合技术博客的文章提纲，目标读者是中级开发者，
文章长度约2000字，包含SEO关键词。"""
        
        return self.outline_llm.invoke(prompt)

class WriterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    
    def write_section(self, section: OutlineSection, topic: str, 
                     context: str) -> str:
        prompt = f"""撰写文章的以下章节：

文章主题：{topic}
章节标题：{section.title}
关键点：{', '.join(section.key_points)}
目标字数：约{section.estimated_words}字

背景上下文：
{context[:1000]}

要求：
- 技术准确，有深度
- 包含代码示例（如适用）
- 语言清晰流畅
- 结合实际案例"""
        
        return self.llm.invoke(prompt).content
    
    def write_article(self, outline: ArticleOutline, research: str) -> str:
        sections_content = []
        context = research
        
        for section in outline.sections:
            content = self.write_section(section, outline.article_title, context)
            sections_content.append(f"## {section.title}\n\n{content}")
            context = content  # 用上一节作为下一节的上下文
        
        article = f"# {outline.article_title}\n\n"
        article += "\n\n".join(sections_content)
        return article

class CriticAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.check_llm = self.llm.with_structured_output(QualityCheck)
    
    def review(self, article: str, outline: ArticleOutline) -> QualityCheck:
        prompt = f"""审查以下技术文章：

原定提纲：
{outline.model_dump_json(indent=2)[:500]}

文章内容：
{article[:3000]}

评估维度：
1. 内容完整性（是否覆盖了提纲的所有要点）
2. 技术准确性（概念是否正确）
3. 代码示例（是否有且可运行）
4. 可读性（结构清晰，表达流畅）
5. 实用价值（读者能学到什么）

质量标准：8分以上通过"""
        
        return self.check_llm.invoke(prompt)
    
    def provide_revision_guidance(self, article: str, check: QualityCheck) -> str:
        prompt = f"""基于以下问题，提供具体的修改建议：

问题列表：
{chr(10).join(f'- {issue}' for issue in check.issues)}

改进建议：
{chr(10).join(f'- {s}' for s in check.suggestions)}

原文（节选）：
{article[:2000]}

请给出具体的修改指导（不要重写全文，只指出需要改的地方和怎么改）："""
        
        return self.llm.invoke(prompt).content

class SEOAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    def optimize(self, article: str, keywords: List[str]) -> str:
        prompt = f"""对以下文章进行SEO优化：

目标关键词：{', '.join(keywords)}

文章：
{article[:3000]}

优化要求：
1. 自然融入关键词（不要堆砌）
2. 优化标题和小标题
3. 添加/优化 meta 描述（文章开头的摘要）
4. 确保内部链接建议（[相关文章: xxx]格式）

请返回优化后的完整文章："""
        
        return self.llm.invoke(prompt).content

# ── 协调者 ──

class ContentProductionCoordinator:
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.outliner = OutlinerAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self.seo_agent = SEOAgent()
        self.metrics: Dict = {}
    
    async def produce(self, topic: str, max_revisions: int = 2) -> dict:
        """完整内容生产流程"""
        start_time = time.time()
        self.metrics = {"topic": topic, "stages": []}
        
        print(f"📝 开始生产：{topic}")
        
        # Stage 1: 研究
        print("\n[1/5] 🔍 研究阶段...")
        t0 = time.time()
        research = self.researcher.research(topic)
        self.metrics["stages"].append({"name": "research", "duration": time.time() - t0})
        print(f"  完成（{len(research)}字，{time.time()-t0:.1f}s）")
        
        # Stage 2: 提纲
        print("[2/5] 📋 提纲阶段...")
        t0 = time.time()
        outline = self.outliner.create_outline(topic, research)
        self.metrics["stages"].append({"name": "outline", "duration": time.time() - t0})
        print(f"  完成（{len(outline.sections)}节，{time.time()-t0:.1f}s）")
        
        # Stage 3: 撰写
        print("[3/5] ✍️ 撰写阶段...")
        t0 = time.time()
        article = self.writer.write_article(outline, research)
        self.metrics["stages"].append({"name": "writing", "duration": time.time() - t0})
        print(f"  完成（{len(article)}字，{time.time()-t0:.1f}s）")
        
        # Stage 4: 审核（循环直到通过或达到最大修改次数）
        print("[4/5] 🔍 审核阶段...")
        revision_count = 0
        check = None
        
        while revision_count <= max_revisions:
            t0 = time.time()
            check = self.critic.review(article, outline)
            duration = time.time() - t0
            
            print(f"  审核结果：{check.score:.1f}/10 {'✅通过' if check.passed else '❌不通过'}")
            
            if check.passed:
                break
            
            if revision_count < max_revisions:
                print(f"  进行第{revision_count+1}次修改...")
                guidance = self.critic.provide_revision_guidance(article, check)
                
                # 根据指导修改文章
                revision_prompt = f"""根据以下修改建议，改进文章：

修改建议：{guidance}

原文：{article[:3000]}

请给出改进后的文章："""
                article = ChatOpenAI(model="gpt-4o", temperature=0.3).invoke(revision_prompt).content
            
            revision_count += 1
            self.metrics["stages"].append({
                "name": f"review_{revision_count}", 
                "duration": duration,
                "score": check.score
            })
        
        # Stage 5: SEO 优化
        print("[5/5] 🚀 SEO优化...")
        t0 = time.time()
        final_article = self.seo_agent.optimize(article, outline.seo_keywords)
        self.metrics["stages"].append({"name": "seo", "duration": time.time() - t0})
        print(f"  完成（{time.time()-t0:.1f}s）")
        
        total_time = time.time() - start_time
        
        return {
            "topic": topic,
            "article": final_article,
            "outline": outline.model_dump(),
            "quality_score": check.score if check else 0,
            "revisions": revision_count,
            "total_time_seconds": total_time,
            "word_count": len(final_article),
            "metrics": self.metrics
        }

# 运行
async def main():
    coordinator = ContentProductionCoordinator()
    result = await coordinator.produce(
        "LangGraph 多 Agent 工作流：从入门到生产",
        max_revisions=2
    )
    
    print(f"\n{'='*60}")
    print(f"生产完成！")
    print(f"字数：{result['word_count']:,}字")
    print(f"质量评分：{result['quality_score']:.1f}/10")
    print(f"修改次数：{result['revisions']}")
    print(f"总耗时：{result['total_time_seconds']:.1f}秒")
    
    # 保存文章
    with open(f"/tmp/article_{int(time.time())}.md", "w", encoding="utf-8") as f:
        f.write(result["article"])
    print(f"\n文章已保存！")
    
    print(f"\n文章开头：")
    print(result["article"][:500])

asyncio.run(main())
```

## 本周回顾

| 技术 | 在本项目中的应用 |
|------|----------------|
| 层级架构 | Coordinator 协调所有 Agent |
| 流水线 | Research→Outline→Write→Review→SEO |
| 角色分工 | 每个 Agent 专注一个职责 |
| 质量门控 | Critic 审核，不通过则修改 |
| 自我反思 | Writer 根据 Critic 建议修改 |

---

*W5 Capstone · 多 Agent 内容生产系统 | Agent + Claw 系列*
