---
layout: default
title: "W7D2 · 浏览器自动化工具"
---

# 浏览器自动化：让 Agent 浏览网络

> **Week 7 · Day 2** | 难度：⭐⭐⭐⭐

---

## 浏览器工具的选择

| 工具 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| Playwright | 快速、稳定、支持现代JS | 环境复杂 | 全功能爬取 |
| Selenium | 生态成熟 | 慢 | 遗留系统 |
| Requests+BeautifulSoup | 最简单 | 不支持JS | 静态页面 |
| Firecrawl API | 开箱即用 | 有成本 | 快速原型 |

## 方案1：Requests + BeautifulSoup（静态页面）

```python
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from urllib.parse import urljoin, urlparse

@tool
def fetch_webpage(url: str, extract_type: str = "text") -> str:
    """获取网页内容。
    
    Args:
        url: 要访问的网页 URL
        extract_type: 提取类型 - "text"(纯文本)/"links"(链接)/"structured"(结构化)
    
    Returns:
        网页内容或链接列表
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除脚本和样式
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        if extract_type == "text":
            text = soup.get_text(separator="\n", strip=True)
            # 清理空行
            lines = [l for l in text.split("\n") if l.strip()]
            return "\n".join(lines)[:3000]  # 限制输出长度
        
        elif extract_type == "links":
            links = []
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                if href.startswith("http"):
                    links.append(f"{a.get_text(strip=True)}: {href}")
            return "\n".join(links[:50])
        
        elif extract_type == "structured":
            # 提取标题、段落、列表
            result = {}
            
            h1 = soup.find("h1")
            result["title"] = h1.get_text() if h1 else ""
            
            headings = [h.get_text() for h in soup.find_all(["h2", "h3"])]
            result["headings"] = headings[:10]
            
            paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50]
            result["content"] = paragraphs[:5]
            
            import json
            return json.dumps(result, ensure_ascii=False)
    
    except requests.Timeout:
        return "错误：请求超时（10秒）"
    except requests.HTTPError as e:
        return f"错误：HTTP {e.response.status_code}"
    except Exception as e:
        return f"错误：{e}"

# 测试
result = fetch_webpage.invoke({"url": "https://python.org", "extract_type": "structured"})
print(result[:500])
```

## 方案2：Playwright（支持 JavaScript）

```bash
pip install playwright
playwright install chromium
```

```python
import asyncio
from playwright.async_api import async_playwright
from langchain.tools import tool

class PlaywrightBrowserTool:
    """Playwright 浏览器工具"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser = None
        self.context = None
    
    async def setup(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        )
    
    async def teardown(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def get_page_content(self, url: str, wait_for: str = None) -> str:
        """获取页面内容（支持 JavaScript 渲染）"""
        page = await self.context.new_page()
        
        try:
            await page.goto(url, timeout=30000)
            
            # 等待特定元素（如动态加载内容）
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=10000)
            else:
                await page.wait_for_load_state("networkidle")
            
            # 获取页面文本
            text = await page.inner_text("body")
            return text[:5000]  # 截断
        
        finally:
            await page.close()
    
    async def screenshot(self, url: str, filepath: str = "/tmp/screenshot.png") -> str:
        """截取页面截图"""
        page = await self.context.new_page()
        
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle")
            await page.screenshot(path=filepath, full_page=True)
            return f"截图已保存到 {filepath}"
        finally:
            await page.close()
    
    async def fill_and_submit_form(self, url: str, 
                                   form_data: dict, 
                                   submit_selector: str) -> str:
        """填写并提交表单"""
        page = await self.context.new_page()
        
        try:
            await page.goto(url, timeout=30000)
            
            for selector, value in form_data.items():
                await page.fill(selector, value)
            
            await page.click(submit_selector)
            await page.wait_for_load_state("networkidle")
            
            return await page.inner_text("body")
        finally:
            await page.close()
    
    async def extract_structured_data(self, url: str, css_selectors: dict) -> dict:
        """用 CSS 选择器提取结构化数据"""
        page = await self.context.new_page()
        
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle")
            
            result = {}
            for key, selector in css_selectors.items():
                try:
                    elements = await page.query_selector_all(selector)
                    texts = [await el.inner_text() for el in elements[:10]]
                    result[key] = texts
                except:
                    result[key] = []
            
            return result
        finally:
            await page.close()

# 使用示例
async def demo_playwright():
    browser_tool = PlaywrightBrowserTool()
    await browser_tool.setup()
    
    # 获取动态页面内容
    content = await browser_tool.get_page_content("https://github.com/langchain-ai/langchain")
    print(f"页面内容（前500字）：\n{content[:500]}")
    
    # 截图
    await browser_tool.screenshot("https://openai.com", "/tmp/openai.png")
    
    await browser_tool.teardown()

asyncio.run(demo_playwright())
```

## 集成到 LangChain Agent

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel

class BrowseInput(BaseModel):
    url: str = Field(description="要访问的网页 URL")
    action: str = Field(
        default="get_text",
        description="操作类型：get_text/screenshot/get_links"
    )

# 同步包装器（LangChain 工具通常需要同步接口）
def browse_sync(url: str, action: str = "get_text") -> str:
    """同步浏览网页"""
    import asyncio
    
    async def _browse():
        tool = PlaywrightBrowserTool()
        await tool.setup()
        try:
            if action == "get_text":
                return await tool.get_page_content(url)
            elif action == "screenshot":
                return await tool.screenshot(url)
            elif action == "get_links":
                return await tool.get_page_content(url)  # 简化
        finally:
            await tool.teardown()
    
    return asyncio.run(_browse())

browser_tool = StructuredTool.from_function(
    func=browse_sync,
    name="browse_webpage",
    description="""使用浏览器访问网页，支持 JavaScript 渲染的现代网页。
    适合需要访问动态内容、SPA 应用的场景。
    注意：只能访问公开网页，不能登录私有系统。""",
    args_schema=BrowseInput
)
```

## 踩坑经验

### 坑1：Playwright 在服务器上没有显示器

**问题**：`headless=False` 在 Linux 服务器上报错。  
**解法**：始终用 `headless=True`，或在 Docker 中安装 Xvfb。

### 坑2：动态内容没加载就开始提取

**问题**：`page.goto()` 完成后，JavaScript 数据还没加载，获取到空内容。  
**解法**：
```python
# 等待网络请求完成
await page.wait_for_load_state("networkidle")
# 或等待特定元素出现
await page.wait_for_selector(".content-loaded", timeout=10000)
```

### 坑3：被网站反爬封禁

**解法**：
1. 设置真实的 User-Agent
2. 添加随机延迟（`await asyncio.sleep(random.uniform(1, 3))`）
3. 使用轮换代理
4. 优先考虑网站的官方 API

---

*W7D2 · 浏览器自动化工具 | Agent + Claw 系列*
