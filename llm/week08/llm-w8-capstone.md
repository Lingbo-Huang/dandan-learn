---
layout: default
title: "D6 · Capstone：完整 LLM 应用"
render_with_liquid: false
---

# D6 · Capstone：构建完整 LLM 应用——面试助手

> **目标**：构建一个"LLM 算法/工程岗面试助手"，集成 RAG、Function Calling、结构化输出和安全护栏。

---

## 一、系统功能

```
面试助手功能：
├── 题目生成（按岗位、难度、主题生成面试题）
├── 答案评估（评分 + 反馈 + 参考答案）
├── 知识检索（RAG 检索相关知识点）
├── 学习建议（个性化学习路径推荐）
└── 模拟面试（多轮对话，模拟面试官）
```

---

## 二、核心实现

```python
# interview_assistant.py
import asyncio
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal
import instructor
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:8000/v1", api_key="token")
)

# 数据模型
class InterviewQuestion(BaseModel):
    """面试题"""
    question: str
    topic: str
    difficulty: Literal["初级", "中级", "高级"]
    expected_time_mins: int = Field(ge=1, le=30)
    key_points: list[str] = Field(description="答题要点，3-5条")
    follow_up_questions: list[str] = Field(description="追问问题，1-3个")

class AnswerEvaluation(BaseModel):
    """答案评估"""
    score: int = Field(ge=0, le=100)
    level: Literal["优秀", "良好", "一般", "需要改进"]
    
    strengths: list[str] = Field(description="答案的优点")
    weaknesses: list[str] = Field(description="答案的不足")
    missing_points: list[str] = Field(description="遗漏的关键点")
    
    reference_answer: str = Field(description="参考答案（简洁版）")
    improvement_suggestion: str = Field(description="改进建议")

class StudyPlan(BaseModel):
    """学习计划"""
    
    class WeekPlan(BaseModel):
        week: int
        topics: list[str]
        resources: list[str]
        practice_goals: list[str]
    
    overall_assessment: str
    weak_areas: list[str]
    strong_areas: list[str]
    weekly_plans: list[WeekPlan]


class InterviewAssistant:
    """面试助手主类"""
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base  # RAG 系统
        self.session_history = []
        self.weak_areas = []  # 跟踪弱项
    
    def generate_question(
        self,
        topic: str,
        difficulty: Literal["初级", "中级", "高级"] = "中级",
        avoid_repeats: list[str] = None,
    ) -> InterviewQuestion:
        """生成面试题"""
        
        context = ""
        if self.knowledge_base:
            docs = self.knowledge_base.retriever.search(topic, k=3)
            context = "\n".join(r['doc'] for r in docs)
        
        avoid_str = f"\n避免重复以下问题：{avoid_repeats}" if avoid_repeats else ""
        
        question = client.chat.completions.create(
            model="qwen2.5-7b",
            response_model=InterviewQuestion,
            messages=[{
                "role": "user",
                "content": f"""生成一道关于"{topic}"的{difficulty}面试题。
                
参考知识：
{context}

要求：
- 题目要考察深度理解，不要表面知识
- 对于{difficulty}级别，合理设置难度
- key_points 要覆盖核心知识点{avoid_str}"""
            }]
        )
        return question
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        key_points: list[str] = None,
    ) -> AnswerEvaluation:
        """评估答案"""
        
        key_points_str = "\n".join(f"- {kp}" for kp in (key_points or []))
        
        evaluation = client.chat.completions.create(
            model="qwen2.5-7b",
            response_model=AnswerEvaluation,
            messages=[{
                "role": "user",
                "content": f"""请评估以下面试答案：

面试题：{question}

关键要点：
{key_points_str}

候选人答案：
{answer}

请严格、客观地评估，给出 0-100 的分数。"""
            }]
        )
        
        # 记录弱项
        if evaluation.score < 60:
            self.weak_areas.extend(evaluation.missing_points[:2])
        
        return evaluation
    
    def generate_study_plan(self, assessment_history: list[AnswerEvaluation]) -> StudyPlan:
        """根据答题历史生成学习计划"""
        
        avg_score = sum(e.score for e in assessment_history) / len(assessment_history)
        all_weak_points = []
        for e in assessment_history:
            all_weak_points.extend(e.missing_points)
        
        plan = client.chat.completions.create(
            model="qwen2.5-7b",
            response_model=StudyPlan,
            messages=[{
                "role": "user",
                "content": f"""根据候选人的面试表现，制定个性化学习计划：

平均分：{avg_score:.0f}/100

弱点领域：
{chr(10).join(f'- {p}' for p in all_weak_points[:10])}

请制定 4 周的学习计划，重点补强弱项。"""
            }]
        )
        return plan
    
    async def mock_interview(self, position: str = "LLM 算法工程师") -> None:
        """模拟面试（多轮对话）"""
        
        topics = ["Transformer 架构", "预训练", "SFT", "RLHF", "推理优化", "RAG"]
        evaluations = []
        question_history = []
        
        print(f"\n{'='*60}")
        print(f"📋 模拟面试开始 - 应聘岗位：{position}")
        print(f"{'='*60}\n")
        
        for i, topic in enumerate(topics[:3], 1):  # 模拟3道题
            print(f"\n[题目 {i}]（主题：{topic}）")
            
            # 生成题目
            question = self.generate_question(
                topic=topic,
                difficulty="中级",
                avoid_repeats=question_history
            )
            question_history.append(question.question[:50])
            
            print(f"面试官：{question.question}")
            print(f"（预计作答时间：{question.expected_time_mins}分钟）\n")
            
            # 获取用户答案
            answer = input("您的回答：").strip()
            if not answer:
                answer = "[未作答]"
            
            # 评估
            print("\n评估中...")
            eval_result = self.evaluate_answer(
                question.question,
                answer,
                question.key_points
            )
            
            evaluations.append(eval_result)
            
            print(f"\n📊 评分：{eval_result.score}/100 ({eval_result.level})")
            print(f"✅ 优点：{', '.join(eval_result.strengths[:2])}")
            if eval_result.weaknesses:
                print(f"❌ 不足：{', '.join(eval_result.weaknesses[:2])}")
            print(f"💡 参考答案要点：{eval_result.reference_answer[:100]}...")
        
        # 生成总结
        print("\n" + "="*60)
        print("📈 面试总结")
        avg = sum(e.score for e in evaluations) / len(evaluations)
        print(f"总体得分：{avg:.0f}/100")
        
        if avg >= 80:
            print("表现：优秀，可以进入下一轮面试 🎉")
        elif avg >= 60:
            print("表现：良好，有一定提升空间 💪")
        else:
            print("表现：需要加强学习，建议 2-3 周后再试 📚")
        
        # 学习计划
        print("\n生成个性化学习计划...")
        plan = self.generate_study_plan(evaluations)
        print(f"\n📅 4周学习计划：")
        print(f"需要重点补强：{', '.join(plan.weak_areas[:3])}")
        for week_plan in plan.weekly_plans[:2]:
            print(f"\n第{week_plan.week}周：{', '.join(week_plan.topics)}")
            for goal in week_plan.practice_goals[:2]:
                print(f"  - {goal}")


# 快速评估模式（不需要完整面试）
def quick_evaluate(question: str, answer: str) -> dict:
    """快速答案评估（命令行工具）"""
    assistant = InterviewAssistant()
    result = assistant.evaluate_answer(question, answer)
    
    return {
        "score": result.score,
        "level": result.level,
        "top_strength": result.strengths[0] if result.strengths else "无",
        "top_weakness": result.weaknesses[0] if result.weaknesses else "无",
        "missing": result.missing_points,
        "reference": result.reference_answer[:200],
    }


# 启动
if __name__ == "__main__":
    assistant = InterviewAssistant()
    asyncio.run(assistant.mock_interview())
```

---

## 三、Week 8 总结 & Phase 1 回顾

```
Week 8 完成！

本周核心技能：
  ✅ Prompt 工程（CoT/Few-shot/Self-consistency）
  ✅ Function Calling（工具调用、Agent 构建）
  ✅ 结构化输出（Pydantic/Outlines）
  ✅ LLM 安全（Prompt Injection、Guardrails）

Week 4-8 全部完成！Phase 1 技术栈：

Week 4: 预训练（数据/Tokenizer/GPT/Scaling Law）
Week 5: SFT 与对齐（LoRA/QLoRA/RLHF/DPO）
Week 6: 推理与部署（KV Cache/量化/vLLM/投机采样）
Week 7: RAG 全链路（向量库/Embedding/Rerank/评估）
Week 8: 应用工程（Prompt/Function Calling/安全）
```

### 面试必备知识点汇总

| 主题 | 必须能答的问题 |
|------|--------------|
| 预训练 | BPE 原理、Scaling Law、FLOPs 估算 |
| SFT | Chat Template、损失掩码、LoRA 推导 |
| 对齐 | DPO 公式推导、RLHF 三阶段 |
| 推理 | KV Cache 内存计算、vLLM PagedAttention |
| RAG | 混合检索、Reranker 原理、RAGAS 评估 |
| 应用 | CoT 原理、Function Calling 机制、Prompt Injection |

---

## 四、下一步

Phase 2 展望（Week 9-16）：
- BERT/GPT/LLaMA/Qwen 主流模型深度解析
- MoE、长上下文、稀疏注意力
- 多模态（VLM/语音/视频）
- 前沿研究（o1 推理、Mamba、扩散 LLM）
