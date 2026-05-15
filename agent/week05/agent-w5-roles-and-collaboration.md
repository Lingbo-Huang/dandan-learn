---
layout: default
title: "W5D3 · 角色分工与协作模式"
---

# 角色分工：让每个 Agent 成为专家

> **Week 5 · Day 3** | 难度：⭐⭐⭐⭐

---

## 角色设计的核心原则

好的多 Agent 角色分工应该像一个优秀团队：
- **单一职责**：每个 Agent 专注一件事
- **明确边界**：角色间没有责任重叠
- **互补能力**：角色组合覆盖所有需求
- **清晰接口**：角色间的输入输出格式明确

## 常见角色模式

### 模式1：研究-分析-决策三角

```
┌───────────────────────────────────────┐
│           团队协作流程                  │
│                                        │
│  Researcher ──→ Analyst ──→ Decider   │
│  (信息收集)     (数据分析)   (最终决策)  │
│       ↑                        │      │
│       └────── Critic ──────────┘      │
│              (批评质疑)                │
└───────────────────────────────────────┘
```

```python
from langchain_openai import ChatOpenAI
from typing import Dict, List

class RoleBasedAgent:
    """基于角色的 Agent"""
    
    ROLE_PROMPTS = {
        "researcher": """你是一个严谨的研究员。
你的职责：收集、整理、验证信息。
行为准则：
- 区分事实与观点
- 标注信息来源的可靠性
- 对不确定信息明确标注
- 绝不捏造数据""",
        
        "analyst": """你是一个数据分析师。
你的职责：从数据和信息中提取洞察。
行为准则：
- 基于数据得出结论
- 识别模式和趋势
- 量化评估不确定性
- 避免过度解读""",
        
        "critic": """你是一个严格的批评者。
你的职责：找出论点中的漏洞和缺陷。
行为准则：
- 质疑每个假设
- 寻找反例和反驳
- 评估逻辑一致性
- 建设性而非破坏性""",
        
        "decider": """你是一个决策者。
你的职责：综合信息做出最终决策。
行为准则：
- 权衡利弊
- 考虑风险
- 给出可行建议
- 为决策负责""",
        
        "executor": """你是一个执行专家。
你的职责：将决策转化为具体行动计划。
行为准则：
- 步骤具体可操作
- 考虑资源约束
- 设置里程碑
- 识别依赖关系"""
    }
    
    def __init__(self, role: str, model: str = "gpt-4o-mini"):
        if role not in self.ROLE_PROMPTS:
            raise ValueError(f"未知角色：{role}。可用角色：{list(self.ROLE_PROMPTS.keys())}")
        
        self.role = role
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.system_prompt = self.ROLE_PROMPTS[role]
        self.memory: List[Dict] = []  # 短期记忆
    
    def process(self, input_text: str, context: str = "") -> str:
        """处理输入，输出角色视角的结果"""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"背景信息：\n{context}\n\n你的任务：\n{input_text}"
            })
        else:
            messages.append({"role": "user", "content": input_text})
        
        response = self.llm.invoke(messages)
        result = response.content
        
        # 记录到短期记忆
        self.memory.append({"input": input_text, "output": result})
        
        return result

class CollaborationOrchestrator:
    """协作编排器：协调多个角色的工作"""
    
    def __init__(self):
        self.agents = {
            "researcher": RoleBasedAgent("researcher", "gpt-4o-mini"),
            "analyst": RoleBasedAgent("analyst", "gpt-4o-mini"),
            "critic": RoleBasedAgent("critic", "gpt-4o"),
            "decider": RoleBasedAgent("decider", "gpt-4o"),
        }
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def run_investigation(self, question: str) -> dict:
        """完整的调查流程"""
        print(f"开始调查：{question}")
        results = {}
        
        # Phase 1: 研究
        print("\n[1/4] 研究阶段...")
        research = self.agents["researcher"].process(
            f"请收集关于以下问题的相关信息和数据：{question}"
        )
        results["research"] = research
        print(f"  研究完成（{len(research)}字）")
        
        # Phase 2: 分析
        print("[2/4] 分析阶段...")
        analysis = self.agents["analyst"].process(
            f"基于研究结果，分析关键趋势和洞察：{question}",
            context=research
        )
        results["analysis"] = analysis
        print(f"  分析完成（{len(analysis)}字）")
        
        # Phase 3: 批评
        print("[3/4] 批评阶段...")
        combined = f"研究：{research}\n\n分析：{analysis}"
        critique = self.agents["critic"].process(
            "请严格审查以上研究和分析，找出漏洞、不足和可能的错误：",
            context=combined
        )
        results["critique"] = critique
        print(f"  批评完成（{len(critique)}字）")
        
        # Phase 4: 决策
        print("[4/4] 决策阶段...")
        all_context = f"""
研究发现：{research}

分析结论：{analysis}

批评意见：{critique}
"""
        decision = self.agents["decider"].process(
            f"综合以上信息，给出关于以下问题的最终建议：{question}",
            context=all_context
        )
        results["decision"] = decision
        print(f"  决策完成（{len(decision)}字）")
        
        return results

# 测试
orchestrator = CollaborationOrchestrator()
results = orchestrator.run_investigation(
    "一家初创公司是否应该在2024年将主要技术栈迁移到 Rust？"
)
print("\n最终建议：")
print(results["decision"])
```

### 模式2：辩论协商模式

```python
class DebateAgent:
    """辩论模式：两个 Agent 持不同立场，通过辩论得出更好的结论"""
    
    def __init__(self):
        self.pro_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.con_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.judge = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def debate(self, proposition: str, rounds: int = 3) -> dict:
        """进行多轮辩论"""
        debate_history = []
        
        for round_num in range(rounds):
            print(f"\n第 {round_num + 1} 轮辩论")
            
            # 正方发言
            history_text = "\n".join([
                f"{'正方' if h['side'] == 'pro' else '反方'}：{h['argument']}"
                for h in debate_history
            ])
            
            pro_prompt = f"""你支持以下命题："{proposition}"

{'之前的辩论内容：' + history_text if history_text else ''}

请提出你最有力的支持论点（200字以内）："""
            
            pro_arg = self.pro_agent.invoke(pro_prompt).content
            debate_history.append({"side": "pro", "argument": pro_arg, "round": round_num})
            print(f"正方：{pro_arg[:100]}...")
            
            # 反方发言
            con_prompt = f"""你反对以下命题："{proposition}"

之前的辩论内容：
{chr(10).join([f"{'正方' if h['side'] == 'pro' else '反方'}：{h['argument']}" for h in debate_history])}

请反驳正方观点并提出你的论点（200字以内）："""
            
            con_arg = self.con_agent.invoke(con_prompt).content
            debate_history.append({"side": "con", "argument": con_arg, "round": round_num})
            print(f"反方：{con_arg[:100]}...")
        
        # 裁判总结
        all_arguments = "\n\n".join([
            f"{'正方' if h['side'] == 'pro' else '反方'}（第{h['round']+1}轮）：{h['argument']}"
            for h in debate_history
        ])
        
        judge_prompt = f"""作为公正的裁判，评估以下辩论：

命题：{proposition}

辩论内容：
{all_arguments}

请给出：
1. 哪方论点更有说服力，为什么
2. 综合两方观点的平衡结论
3. 需要特别注意的关键因素"""
        
        verdict = self.judge.invoke(judge_prompt).content
        
        return {
            "proposition": proposition,
            "debate_history": debate_history,
            "verdict": verdict
        }

# 测试辩论
debater = DebateAgent()
result = debater.debate("AI 将在5年内取代大多数软件工程师的工作", rounds=2)
print("\n裁判结论：")
print(result["verdict"])
```

### 模式3：专家委员会模式

多个领域专家，每个领域给出专业建议，最终委员会投票：

```python
class ExpertCommittee:
    """专家委员会：多领域专家共同决策"""
    
    EXPERT_PROFILES = {
        "技术专家": "你是资深技术专家，关注技术可行性、架构质量、性能和安全。",
        "业务专家": "你是业务专家，关注商业价值、用户需求、市场竞争和ROI。",
        "风险专家": "你是风险管理专家，关注潜在风险、合规要求、应急预案。",
        "财务专家": "你是财务专家，关注成本、收益、资金流和财务健康。",
    }
    
    def __init__(self):
        self.experts = {
            name: ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            for name in self.EXPERT_PROFILES
        }
        self.chair = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def consult(self, proposal: str) -> dict:
        """召开专家委员会"""
        expert_opinions = {}
        
        for expert_name, profile in self.EXPERT_PROFILES.items():
            prompt = f"""{profile}

请从你的专业角度评估以下提案：

{proposal}

给出：
1. 你的专业评分（1-10）
2. 主要支持理由（2-3点）
3. 主要顾虑（2-3点）
4. 建议（具体可行）"""
            
            response = self.experts[expert_name].invoke(prompt)
            expert_opinions[expert_name] = response.content
        
        # 主席总结
        opinions_text = "\n\n".join([
            f"**{name}的意见**：\n{opinion}"
            for name, opinion in expert_opinions.items()
        ])
        
        chair_prompt = f"""作为委员会主席，综合各专家意见，给出最终建议：

提案：{proposal}

各专家意见：
{opinions_text}

请给出：
1. 综合评分
2. 核心建议（通过/修改/拒绝）
3. 如果通过，需要满足哪些条件
4. 最需要关注的3个要点"""
        
        final_recommendation = self.chair.invoke(chair_prompt).content
        
        return {
            "proposal": proposal,
            "expert_opinions": expert_opinions,
            "final_recommendation": final_recommendation
        }

committee = ExpertCommittee()
result = committee.consult("在现有产品中加入AI客服功能，预算50万，3个月内上线")
print(result["final_recommendation"])
```

## 踩坑经验

### 坑1：角色提示词不够清晰——Agent 角色扮演失败

**问题**：写了 system prompt 但 Agent 仍然以通用助手方式回答，没有体现角色特色。  
**解法**：
1. 在 prompt 中明确角色的"禁忌"（你不会做什么）
2. 提供角色的典型思维方式示例
3. 要求 Agent 先声明自己的角色视角再作答

### 坑2：角色间循环等待

**问题**：Researcher 等 Analyst 的需求，Analyst 又等 Researcher 的数据。  
**解法**：在设计阶段明确依赖关系，绝不允许循环依赖。

---

*W5D3 · 角色分工与协作模式 | Agent + Claw 系列*
