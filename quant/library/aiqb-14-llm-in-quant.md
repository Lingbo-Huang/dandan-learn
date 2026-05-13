---
layout: default
title: "AIQuantBook L14 · LLM 在量化中的应用"
source: "https://github.com/waylandzhang/ai-quant-book"
---

# L14 · LLM 在量化中的应用

> **来源**：[AI Quant Book](https://github.com/waylandzhang/ai-quant-book) · waylandzhang

---

## LLM 能做什么？

量化 + LLM 的交叉点：

```
1. 另类数据处理
   ├── 财报/公告的情绪分析
   ├── 新闻事件的影响评估
   └── 社交媒体舆情量化

2. 策略研究辅助
   ├── 自然语言描述策略逻辑
   ├── 代码生成与调试
   └── 文献综述与知识提取

3. 智能分析助手
   ├── 自然语言查询数据库
   ├── 研报解读与摘要
   └── 多模态信息融合

4. Agent 化策略（前沿）
   ├── 自主研究因子
   ├── 自动回测和迭代
   └── 多智能体协作
```

---

## 一、情绪因子：用 LLM 分析财报

```python
from openai import OpenAI
import pandas as pd

client = OpenAI()

def analyze_earnings_sentiment(text: str) -> dict:
    """
    分析财报/公告的情绪
    返回：{sentiment, confidence, key_points}
    """
    prompt = f"""你是一位专业的金融分析师。请分析以下上市公司公告的情绪倾向。

公告内容：
{text[:3000]}  # 限制长度

请以JSON格式返回：
{{
  "sentiment": "positive/neutral/negative",
  "confidence": 0-1之间的小数,
  "revenue_outlook": "positive/neutral/negative",
  "profit_outlook": "positive/neutral/negative",
  "key_positives": ["积极因素1", "积极因素2"],
  "key_risks": ["风险1", "风险2"],
  "summary": "一句话总结"
}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    import json
    return json.loads(response.choices[0].message.content)

# 批量处理财报
def batch_analyze_reports(reports: list[dict], 
                            delay: float = 0.5) -> pd.DataFrame:
    """
    批量分析财报情绪，构建情绪因子
    reports: [{'ts_code': '000001.SZ', 'date': '2024-01-01', 'content': '...'}]
    """
    import time
    results = []
    
    for report in reports:
        try:
            sentiment = analyze_earnings_sentiment(report['content'])
            results.append({
                'ts_code': report['ts_code'],
                'date': report['date'],
                **sentiment
            })
            time.sleep(delay)  # 避免触发限速
        except Exception as e:
            print(f"Error processing {report['ts_code']}: {e}")
    
    df = pd.DataFrame(results)
    
    # 情绪数值化
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    df['sentiment_factor'] = df['sentiment_score'] * df['confidence']
    
    return df
```

---

## 二、自然语言查询数据

```python
from openai import OpenAI
import pandas as pd
import sqlite3

client = OpenAI()

SCHEMA = """
数据库表结构：
- daily_prices: ts_code, trade_date, open, high, low, close, volume, pct_chg
- company_info: ts_code, name, industry, market_cap
- financial_data: ts_code, report_date, roe, eps, revenue_growth, profit_growth
"""

def nl_to_sql(question: str) -> str:
    """将自然语言问题转换为 SQL 查询"""
    prompt = f"""根据以下数据库结构，将用户的自然语言问题转换为 SQL 查询。
只返回 SQL，不要任何解释。

{SCHEMA}

用户问题：{question}

SQL："""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

def query_data(question: str, db_path: str = 'quant.db') -> pd.DataFrame:
    """自然语言查询数据库"""
    sql = nl_to_sql(question)
    print(f"生成的 SQL：\n{sql}\n")
    
    conn = sqlite3.connect(db_path)
    result = pd.read_sql(sql, conn)
    conn.close()
    
    return result

# 使用示例
# result = query_data("找出最近一个季度ROE超过20%且市值在50-200亿之间的股票")
# result = query_data("过去一年涨幅最大的10只电子行业股票")
```

---

## 三、LLM Agent 策略研究

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_factor_ic(factor_name: str, lookback_months: int = 12) -> str:
    """计算因子的历史IC（信息系数）。输入因子名称和回测月数。"""
    # 实际中调用因子计算引擎
    # 这里返回模拟结果
    import random
    ic = random.uniform(-0.05, 0.15)
    icir = ic / random.uniform(0.02, 0.08)
    return f"因子 {factor_name} 过去{lookback_months}个月：IC均值={ic:.4f}, ICIR={icir:.2f}"

@tool
def run_backtest(strategy_description: str, start_date: str, end_date: str) -> str:
    """根据策略描述运行回测。"""
    # 实际中调用回测引擎
    return f"回测完成（{start_date}→{end_date}）：年化收益=12.3%, 夏普=1.45, 最大回撤=-15.2%"

@tool
def search_academic_papers(topic: str) -> str:
    """搜索相关学术论文。"""
    # 实际中调用学术搜索 API
    return f"找到 {topic} 相关论文：\n1. Fama & French (1992) - 价值因子\n2. Jegadeesh (1993) - 动量因子"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_factor_ic, run_backtest, search_academic_papers]

system_prompt = """你是一位专业的量化研究助手。
你可以使用工具来研究因子、运行回测和查找文献。
请系统性地分析用户的研究问题，给出有数据支撑的建议。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({
    "input": "请研究一下低波动因子的有效性，包括学术文献支持、历史IC表现，以及一个简单的策略回测"
})
print(result['output'])
```

---

## 关键认识

1. **LLM 最擅长的是处理非结构化文本**：财报情绪、新闻影响是真正的 Alpha 来源
2. **自然语言→SQL**：让非技术的投研人员直接查数据，提高研究效率
3. **Agent 策略研究**：自动化因子挖掘流程，但需要人类审核
4. **局限性**：LLM 不能替代量化的数学基础，只是加速工具

---

## 延伸阅读

- [AI Quant Book](https://github.com/waylandzhang/ai-quant-book)
- [LangChain 文档](https://python.langchain.com/)
- Lopez de Prado - "Quantamental Investing"
