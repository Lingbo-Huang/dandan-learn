---
layout: default
title: "W5D2 · Agent 通信协议"
---

# Agent 通信协议：让 Agent 高效"说话"

> **Week 5 · Day 2** | 难度：⭐⭐⭐⭐

---

## Agent 通信的核心挑战

多 Agent 系统中，通信质量直接决定系统质量。核心挑战：
1. **消息格式**：Agent 间发什么、怎么发？
2. **路由机制**：消息发给谁？
3. **状态同步**：多 Agent 如何保持一致的共享状态？
4. **错误处理**：消息丢失或处理失败怎么办？

## 统一消息格式设计

```python
from pydantic import BaseModel, Field
from typing import Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

class MessageType(str, Enum):
    TASK = "task"           # 任务分配
    RESULT = "result"       # 任务结果
    QUERY = "query"         # 查询请求
    RESPONSE = "response"   # 查询响应
    EVENT = "event"         # 事件通知
    ERROR = "error"         # 错误报告
    HEARTBEAT = "heartbeat" # 心跳检测

class AgentMessage(BaseModel):
    """标准 Agent 消息格式"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    sender_id: str
    receiver_id: str           # "broadcast" 表示广播
    correlation_id: Optional[str] = None  # 关联消息 ID（用于请求-响应配对）
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # 内容
    content: Any
    metadata: dict = Field(default_factory=dict)
    
    # 可靠性
    priority: int = Field(default=5, ge=1, le=10)
    ttl_seconds: Optional[int] = None   # 消息有效期
    retry_count: int = 0
    max_retries: int = 3

class AgentMessageBus:
    """中央消息总线"""
    
    def __init__(self):
        self.subscribers: dict = {}  # agent_id -> callback
        self.message_queue: list = []
        self.message_history: list = []
        self.dead_letter_queue: list = []  # 处理失败的消息
    
    def subscribe(self, agent_id: str, callback):
        """Agent 订阅消息"""
        self.subscribers[agent_id] = callback
        print(f"Agent {agent_id} 已订阅消息总线")
    
    def unsubscribe(self, agent_id: str):
        """Agent 取消订阅"""
        self.subscribers.pop(agent_id, None)
    
    async def publish(self, message: AgentMessage):
        """发布消息"""
        # 检查消息有效期
        if message.ttl_seconds:
            age = (datetime.now() - message.timestamp).seconds
            if age > message.ttl_seconds:
                print(f"消息 {message.message_id} 已过期，丢弃")
                return False
        
        self.message_history.append(message)
        
        # 路由消息
        if message.receiver_id == "broadcast":
            # 广播给所有 Agent
            receivers = list(self.subscribers.keys())
        else:
            receivers = [message.receiver_id]
        
        success = True
        for receiver_id in receivers:
            if receiver_id not in self.subscribers:
                print(f"警告：Agent {receiver_id} 未找到")
                continue
            
            try:
                await self.subscribers[receiver_id](message)
            except Exception as e:
                print(f"消息投递失败：{receiver_id} - {e}")
                # 重试逻辑
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    self.message_queue.append(message)
                else:
                    self.dead_letter_queue.append(message)
                success = False
        
        return success
    
    def get_history(self, agent_id: str = None, 
                   message_type: MessageType = None) -> List[AgentMessage]:
        """查询消息历史"""
        history = self.message_history
        if agent_id:
            history = [m for m in history 
                      if m.sender_id == agent_id or m.receiver_id == agent_id]
        if message_type:
            history = [m for m in history if m.message_type == message_type]
        return history
```

## 事件驱动通信模式

```python
import asyncio
from typing import Callable, Dict

class EventDrivenAgentSystem:
    """事件驱动的多 Agent 系统"""
    
    def __init__(self):
        self.bus = AgentMessageBus()
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: dict, sender_id: str):
        """发出事件"""
        handlers = self.event_handlers.get(event_type, [])
        message = AgentMessage(
            message_type=MessageType.EVENT,
            sender_id=sender_id,
            receiver_id="broadcast",
            content={"event_type": event_type, "data": data}
        )
        
        await self.bus.publish(message)
        
        # 触发本地处理器
        for handler in handlers:
            await handler(data)

class BaseAgent:
    """Agent 基类，内置通信能力"""
    
    def __init__(self, agent_id: str, bus: AgentMessageBus):
        self.agent_id = agent_id
        self.bus = bus
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.inbox: List[AgentMessage] = []
        
        # 订阅消息总线
        bus.subscribe(agent_id, self.handle_message)
    
    async def handle_message(self, message: AgentMessage):
        """处理收到的消息"""
        self.inbox.append(message)
        
        if message.message_type == MessageType.TASK:
            result = await self.process_task(message.content)
            # 发送结果
            await self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT,
                content=result,
                correlation_id=message.message_id
            )
        elif message.message_type == MessageType.QUERY:
            response = await self.answer_query(message.content)
            await self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=response,
                correlation_id=message.message_id
            )
    
    async def send_message(self, receiver_id: str, message_type: MessageType,
                          content: Any, correlation_id: str = None):
        """发送消息"""
        message = AgentMessage(
            message_type=message_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            correlation_id=correlation_id
        )
        return await self.bus.publish(message)
    
    async def process_task(self, task_content: Any) -> str:
        """处理任务（子类重写）"""
        response = await self.llm.ainvoke(str(task_content))
        return response.content
    
    async def answer_query(self, query: Any) -> str:
        """回答查询（子类重写）"""
        response = await self.llm.ainvoke(str(query))
        return response.content

# 使用示例：构建研究助手系统
class ResearcherAgent(BaseAgent):
    async def process_task(self, task_content) -> str:
        prompt = f"作为研究员，收集关于以下主题的信息：{task_content}"
        response = await self.llm.ainvoke(prompt)
        return response.content

class AnalystAgent(BaseAgent):
    async def process_task(self, task_content) -> str:
        prompt = f"作为分析师，分析以下数据的趋势和含义：{task_content}"
        response = await self.llm.ainvoke(prompt)
        return response.content

class WriterAgent(BaseAgent):
    async def process_task(self, task_content) -> str:
        prompt = f"作为写作者，将以下内容整理成清晰的报告：{task_content}"
        response = await self.llm.ainvoke(prompt)
        return response.content

async def demo_event_driven():
    bus = AgentMessageBus()
    
    researcher = ResearcherAgent("researcher", bus)
    analyst = AnalystAgent("analyst", bus)
    writer = WriterAgent("writer", bus)
    
    # Orchestrator 分配任务
    topic = "2024年大模型技术进展"
    
    # 1. 给 researcher 发任务
    task_msg = AgentMessage(
        message_type=MessageType.TASK,
        sender_id="orchestrator",
        receiver_id="researcher",
        content=topic
    )
    await bus.publish(task_msg)
    
    # 等待处理
    await asyncio.sleep(2)
    
    # 获取 researcher 的结果
    research_results = [m.content for m in researcher.inbox 
                       if m.message_type == MessageType.TASK]
    
    print("通信历史：")
    for msg in bus.get_history():
        print(f"  {msg.sender_id} → {msg.receiver_id}: {msg.message_type.value}")

asyncio.run(demo_event_driven())
```

## 共享状态管理

多 Agent 系统需要共享全局状态：

```python
from threading import Lock
import json

class SharedStateManager:
    """线程安全的共享状态管理器"""
    
    def __init__(self):
        self._state: dict = {}
        self._lock = Lock()
        self._history: List[dict] = []
    
    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any, agent_id: str = "unknown"):
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            self._history.append({
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "key": key,
                "old_value": old_value,
                "new_value": value
            })
    
    def update(self, updates: dict, agent_id: str = "unknown"):
        """批量更新"""
        with self._lock:
            for key, value in updates.items():
                self._state[key] = value
    
    def get_snapshot(self) -> dict:
        """获取当前状态快照"""
        with self._lock:
            return dict(self._state)
    
    def get_audit_trail(self) -> List[dict]:
        """获取变更历史（审计）"""
        return list(self._history)

# 全局共享状态示例
shared_state = SharedStateManager()
shared_state.set("project_status", "in_progress", "orchestrator")
shared_state.set("research_results", [], "orchestrator")
```

## 踩坑经验

### 坑1：消息风暴——广播引发雪崩
**问题**：一个事件触发广播，所有 Agent 都响应，产生大量次级消息。  
**解法**：设置消息频率限制，用令牌桶算法控制每个 Agent 的发消息速率。

### 坑2：消息乱序——并发导致顺序错误
**问题**：并发的消息不按发送顺序到达，导致处理出错。  
**解法**：关键消息附加序列号，接收方按序列号排序后再处理。

### 坑3：死锁——Agent A 等 B，B 等 A
**解法**：超时机制 + 心跳检测，检测到超时自动释放等待。

---

*W5D2 · Agent 通信协议 | Agent + Claw 系列*
