---
layout: default
title: "D3 · 结构化输出"
render_with_liquid: false
---

# D3 · 结构化输出：JSON Mode 与约束解码

> **生产环境中，LLM 的输出必须是可靠的结构化数据。** JSON mode、Pydantic 验证和约束解码是保证输出格式的三种手段。

---

## 一、为什么需要结构化输出？

```python
# ❌ 不可靠：LLM 可能输出任意格式
response = "情感分析结果：这段文字表达了积极的情感，大约 85% 的正面倾向。"

# ✅ 可靠：强制 JSON 格式
response = {"sentiment": "positive", "confidence": 0.85, "label": "正面"}

# 下游代码可以直接使用
data = json.loads(response)
if data["sentiment"] == "positive" and data["confidence"] > 0.7:
    notify_user("好评！")
```

---

## 二、方法 1：JSON Mode（OpenAI API）

```python
import openai
import json

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

def analyze_sentiment(text: str) -> dict:
    """情感分析，返回结构化结果"""
    
    response = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[
            {
                "role": "system",
                "content": """你是情感分析专家。请以 JSON 格式输出分析结果，包含以下字段：
{
  "sentiment": "positive|negative|neutral",
  "confidence": 0-1之间的浮点数,
  "key_phrases": ["关键短语列表"],
  "explanation": "简要解释"
}
只输出 JSON，不要其他内容。"""
            },
            {"role": "user", "content": f"分析以下文本的情感：{text}"}
        ],
        response_format={"type": "json_object"},  # JSON Mode
        temperature=0.1,
    )
    
    result = json.loads(response.choices[0].message.content)
    return result

# 使用
result = analyze_sentiment("这款产品真的太棒了，超出预期！")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 三、方法 2：Instructor + Pydantic

```python
"""
Instructor 库：让 LLM 输出 Pydantic 模型实例
优点：
1. 类型安全（Python 类型提示）
2. 自动验证（Pydantic v2）
3. 自动重试（输出不合法时自动重试）
4. 支持嵌套结构
"""

import instructor
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from openai import OpenAI

# 用 instructor 包装 client
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:8000/v1", api_key="token")
)

# 定义数据模型
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0-1")
    key_phrases: list[str] = Field(description="关键短语列表，最多5个")
    explanation: str = Field(description="简要解释，不超过100字")
    
    @validator('key_phrases')
    def limit_phrases(cls, v):
        return v[:5]  # 最多5个短语

class EntityExtraction(BaseModel):
    """命名实体识别结果"""
    
    class Entity(BaseModel):
        text: str
        type: Literal["PERSON", "ORG", "LOC", "DATE", "PRODUCT", "OTHER"]
        start: Optional[int] = None
        confidence: float = Field(default=1.0, ge=0, le=1)
    
    entities: list[Entity]
    sentence_count: int


def extract_entities(text: str) -> EntityExtraction:
    """命名实体识别，返回 Pydantic 模型"""
    return client.chat.completions.create(
        model="qwen2.5-7b",
        response_model=EntityExtraction,  # 指定返回类型
        messages=[{
            "role": "user",
            "content": f"请提取以下文本中的命名实体：{text}"
        }],
        max_retries=3,  # 格式不对时自动重试
    )

# 使用
result = extract_entities("阿里巴巴集团的马云于2019年在杭州宣布退休。")
print(f"发现 {len(result.entities)} 个实体：")
for entity in result.entities:
    print(f"  {entity.text} ({entity.type})")
```

---

## 四、方法 3：Outlines（约束解码）

```python
"""
Outlines：在 token 级别约束 LLM 输出
原理：在每个 token 的 logits 上加掩码，只允许符合当前语法的 token

优点：
- 100% 保证输出格式正确（不依赖 LLM 的"理解"）
- 零成本（不需要额外推理）
- 支持正则表达式、JSON Schema、枚举等约束

缺点：
- 需要直接控制推理过程（不能用 OpenAI API）
- 需要加载 Outlines 兼容的模型
"""

import outlines
from pydantic import BaseModel
from typing import Literal

# 加载模型（支持 transformers 格式）
model = outlines.models.transformers(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
)

# 方式 1：JSON Schema 约束
class Product(BaseModel):
    name: str
    price: float
    category: Literal["electronics", "clothing", "food", "other"]
    in_stock: bool

generator = outlines.generate.json(model, Product)

prompt = "请以JSON格式描述一个苹果手机的基本信息"
result = generator(prompt)
print(type(result))  # <class 'Product'>
print(f"产品名: {result.name}, 价格: {result.price}")

# 方式 2：正则表达式约束
phone_generator = outlines.generate.regex(
    model,
    r"\+86[0-9]{11}|0[0-9]{10,11}"  # 中国电话号码格式
)
phone = phone_generator("请提供一个中国电话号码")

# 方式 3：枚举选择
choice_generator = outlines.generate.choice(
    model,
    ["是", "否", "不确定"]
)
answer = choice_generator("这道数学题答案是 42 吗？")
print(answer)  # 必然是 "是", "否", 或 "不确定" 之一
```

---

## 五、结构化输出最佳实践

```python
from functools import wraps
import json

def require_json_output(schema: dict):
    """装饰器：强制 LLM 输出符合 schema 的 JSON"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
            # 将 schema 注入 prompt
            kwargs['json_schema'] = schema_str
            
            for attempt in range(3):
                try:
                    result = func(*args, **kwargs)
                    # 验证 JSON
                    parsed = json.loads(result)
                    # 可以加更严格的 schema 验证
                    return parsed
                except (json.JSONDecodeError, KeyError) as e:
                    if attempt == 2:
                        raise ValueError(f"LLM 输出格式错误: {e}")
                    # 重试，加入错误提示
                    kwargs['error_hint'] = f"上次输出格式错误：{e}，请严格按 JSON schema 输出"
            
        return wrapper
    return decorator


# 实用工具函数
def safe_json_parse(text: str) -> dict | None:
    """安全解析 LLM 输出的 JSON"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 提取 JSON 块
    import re
    patterns = [
        r'```json\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'\{.*\}',
        r'\[.*\]',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


# 选择策略
def choose_structured_output_method(
    has_api_access: bool,
    need_100_percent_reliable: bool,
    complex_schema: bool,
) -> str:
    if need_100_percent_reliable:
        return "Outlines（约束解码，100%可靠）"
    elif has_api_access and not complex_schema:
        return "JSON Mode（最简单，依赖 LLM 能力）"
    else:
        return "Instructor + Pydantic（最灵活，有自动重试）"
```

---

## 六、面试题精讲

**Q: 如何保证 LLM 输出 100% 是合法 JSON？**

A: 三种方式，可靠性递增：
1. **JSON Mode + Prompt**：在 prompt 中要求 JSON，API 支持 json_object 格式。可靠性 ~90%（LLM 仍可能输出不合法的 JSON）
2. **Instructor + 重试**：自动验证 Pydantic 模型，失败时自动重试。可靠性 ~99%
3. **Outlines 约束解码**：在 token 级别强制语法，理论上 100% 可靠，但需要控制推理过程

**Q: 什么情况下用约束解码（Outlines），什么情况下用 Prompt 方法？**

A:
- **约束解码**：本地部署、输出格式非常严格（如医疗/金融数据）、不能接受任何格式错误
- **Prompt/Instructor**：使用第三方 API（无法控制解码）、格式稍微灵活、允许少量重试

---

## 小结

| 方法 | 可靠性 | 复杂度 | 适用场景 |
|------|--------|--------|---------|
| Prompt + 解析 | 低 | 低 | 原型、不重要场景 |
| JSON Mode | 中 | 低 | 开发测试 |
| Instructor + Pydantic | 高 | 中 | 生产环境常用 |
| Outlines 约束解码 | 极高 | 高 | 关键业务 |
