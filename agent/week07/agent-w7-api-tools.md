---
layout: default
title: "W7D4 · API 调用与外部服务集成"
---

# API 工具：让 Agent 连接外部世界

> **Week 7 · Day 4** | 难度：⭐⭐⭐⭐

---

## API 工具设计原则

1. **失败不抛异常**：永远返回字符串，包含错误信息
2. **超时控制**：每个 API 调用必须有超时
3. **重试机制**：网络错误自动重试
4. **速率限制**：避免被 API 封禁
5. **凭据安全**：API Key 不能出现在日志里

## 通用 API 客户端基类

```python
import requests
import time
from typing import Optional, Dict, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class BaseAPIClient:
    """API 客户端基类：提供重试、限流、错误处理"""
    
    def __init__(self, 
                 base_url: str,
                 api_key: str = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 rate_limit_per_minute: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit_per_minute
        
        self._call_times = []  # 用于速率限制
        self.session = requests.Session()
        
        # 设置默认 headers
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            })
    
    def _rate_limit_check(self):
        """简单的滑动窗口速率限制"""
        now = time.time()
        # 清理1分钟前的记录
        self._call_times = [t for t in self._call_times if now - t < 60]
        
        if len(self._call_times) >= self.rate_limit:
            sleep_time = 60 - (now - self._call_times[0]) + 0.1
            if sleep_time > 0:
                logger.warning(f"速率限制，等待 {sleep_time:.1f} 秒")
                time.sleep(sleep_time)
        
        self._call_times.append(now)
    
    def _request(self, method: str, endpoint: str, 
                 params: dict = None, json: dict = None,
                 **kwargs) -> dict:
        """带重试的 HTTP 请求"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self._rate_limit_check()
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method, url,
                    params=params, json=json,
                    timeout=self.timeout,
                    **kwargs
                )
                
                if response.status_code == 429:
                    # Rate limit hit，等待 Retry-After
                    retry_after = int(response.headers.get("Retry-After", 10))
                    logger.warning(f"API 速率限制，等待 {retry_after} 秒")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
            
            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt  # 指数退避
                    time.sleep(wait)
                    continue
                raise APIError(f"请求超时（{self.timeout}秒）")
            
            except requests.HTTPError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    # 服务器错误，重试
                    time.sleep(2 ** attempt)
                    continue
                raise APIError(
                    f"HTTP 错误 {e.response.status_code}: {e.response.text[:200]}",
                    e.response.status_code
                )
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise APIError(str(e))
        
        raise APIError("超过最大重试次数")
    
    def get(self, endpoint: str, params: dict = None) -> dict:
        return self._request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, json: dict = None) -> dict:
        return self._request("POST", endpoint, json=json)
```

## 实战：常用 API 工具集

### 天气 API

```python
from langchain.tools import tool
import os

weather_client = BaseAPIClient(
    base_url="https://api.openweathermap.org/data/2.5",
    api_key=os.getenv("OPENWEATHER_API_KEY"),
    timeout=10
)

@tool
def get_current_weather(city: str, country_code: str = "CN") -> str:
    """获取城市当前天气信息。
    
    Args:
        city: 城市名（支持中英文）
        country_code: 国家代码，默认CN（中国）
    """
    try:
        data = weather_client.get("weather", params={
            "q": f"{city},{country_code}",
            "appid": weather_client.api_key,
            "units": "metric",
            "lang": "zh_cn"
        })
        
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        
        return (f"{city}天气：{weather}，"
                f"温度{temp:.1f}°C（体感{feels_like:.1f}°C），"
                f"湿度{humidity}%，风速{wind_speed}m/s")
    
    except APIError as e:
        if e.status_code == 404:
            return f"找不到城市：{city}"
        return f"获取天气失败：{e.message}"
    except KeyError:
        return "天气数据格式异常"
```

### 搜索 API（SerpAPI）

```python
@tool
def search_google(query: str, num_results: int = 5) -> str:
    """使用 Google 搜索获取最新信息。
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量（1-10）
    """
    try:
        import serpapi
        
        search = serpapi.search({
            "q": query,
            "num": min(num_results, 10),
            "hl": "zh-cn",
            "api_key": os.getenv("SERPAPI_KEY")
        })
        
        results = search.get("organic_results", [])
        
        if not results:
            return f"没有找到关于 '{query}' 的搜索结果"
        
        formatted = []
        for r in results[:num_results]:
            formatted.append(
                f"**{r.get('title', '')}**\n"
                f"{r.get('snippet', '')}\n"
                f"来源：{r.get('link', '')}"
            )
        
        return "\n\n".join(formatted)
    
    except ImportError:
        # 降级到 DuckDuckGo（无需 API key）
        return _search_duckduckgo(query, num_results)
    except Exception as e:
        return f"搜索失败：{e}"

def _search_duckduckgo(query: str, num: int = 5) -> str:
    """DuckDuckGo 搜索备选"""
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num))
        
        if not results:
            return f"没有找到关于 '{query}' 的结果"
        
        formatted = [
            f"**{r['title']}**\n{r['body']}\n来源：{r['href']}"
            for r in results
        ]
        return "\n\n".join(formatted)
    except Exception as e:
        return f"搜索失败：{e}"
```

### 文件存储 API

```python
import base64
from pathlib import Path

@tool
def read_local_file(filepath: str) -> str:
    """读取本地文件内容。仅支持文本文件，最大1MB。
    
    Args:
        filepath: 文件的绝对路径
    """
    try:
        path = Path(filepath)
        
        # 安全检查：只允许访问特定目录
        allowed_dirs = ["/tmp", "/workspace", os.path.expanduser("~/Documents")]
        if not any(str(path).startswith(d) for d in allowed_dirs):
            return f"错误：不允许访问 {filepath}，仅允许访问指定目录"
        
        if not path.exists():
            return f"文件不存在：{filepath}"
        
        if path.stat().st_size > 1_000_000:  # 1MB
            return f"文件太大（{path.stat().st_size/1024:.0f}KB），最大支持1MB"
        
        content = path.read_text(encoding="utf-8", errors="replace")
        
        if len(content) > 5000:
            return content[:5000] + f"\n...[文件已截断，共{len(content)}字符]"
        
        return content
    
    except PermissionError:
        return f"权限不足，无法读取：{filepath}"
    except Exception as e:
        return f"读取文件失败：{e}"

@tool
def write_local_file(filepath: str, content: str) -> str:
    """写入内容到本地文件。仅支持写入 /tmp 目录。
    
    Args:
        filepath: 文件路径（必须在 /tmp 目录下）
        content: 要写入的内容
    """
    try:
        path = Path(filepath)
        
        # 严格限制写入目录
        if not str(path).startswith("/tmp/"):
            return "错误：只能写入 /tmp/ 目录"
        
        # 创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding="utf-8")
        return f"成功写入 {filepath}（{len(content)} 字符）"
    
    except Exception as e:
        return f"写入文件失败：{e}"
```

### Slack 通知工具

```python
@tool
def send_slack_message(channel: str, message: str) -> str:
    """发送 Slack 消息到指定频道。
    
    Args:
        channel: 频道名或 ID（如 #general 或 C1234567890）
        message: 要发送的消息内容（支持 Markdown）
    """
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        return "错误：未配置 Slack Bot Token"
    
    try:
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {slack_token}"},
            json={
                "channel": channel,
                "text": message,
                "mrkdwn": True
            },
            timeout=10
        )
        
        data = response.json()
        if data.get("ok"):
            return f"消息已发送到 {channel}"
        else:
            return f"发送失败：{data.get('error', '未知错误')}"
    
    except Exception as e:
        return f"Slack 通知失败：{e}"
```

## 踩坑经验

### 坑1：API Key 泄漏到日志

**问题**：LangChain 的 verbose 模式会打印所有 tool 输入，包括含 API key 的参数。  
**解法**：API Key 从环境变量读取，不作为工具参数传入。

### 坑2：并发调用 API 触发速率限制

**问题**：多个 Agent 并发调用同一 API，导致频繁 429 错误。  
**解法**：全局速率限制器，使用令牌桶或信号量控制并发数。

```python
import asyncio

semaphore = asyncio.Semaphore(5)  # 最多5个并发 API 调用

async def rate_limited_api_call(func, *args, **kwargs):
    async with semaphore:
        return await func(*args, **kwargs)
```

### 坑3：外部 API 超时导致 Agent 卡住

**解法**：每个 API 调用设置明确的 timeout，并在工具层面捕获 TimeoutError。

---

*W7D4 · API 调用与外部服务集成 | Agent + Claw 系列*
