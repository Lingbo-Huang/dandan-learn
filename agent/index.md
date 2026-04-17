---
layout: default
title: 🦾 Agent + Claw
---

<style>
.track-hero { background: linear-gradient(135deg, #78350f, #f59e0b); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.9; line-height: 1.6; }
.track-hero .stats { display: flex; gap: 32px; margin-top: 20px; }
.track-hero .stat { text-align: center; }
.track-hero .stat .num { font-size: 2em; font-weight: 700; }
.track-hero .stat .label { font-size: 0.82em; opacity: 0.75; }
.back { margin-bottom: 16px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }
.phase { margin-bottom: 12px; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.phase-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 22px; background: white; cursor: pointer; border-left: 5px solid #f59e0b; user-select: none; }
.phase-header:hover { background: #fffbeb; }
.phase-header .phase-title { font-weight: 600; font-size: 1em; color: #78350f; }
.phase-header .phase-meta { font-size: 0.82em; color: #6b7280; }
.phase-header .arrow { transition: transform 0.2s; color: #f59e0b; }
.phase-body { display: none; background: #fffbeb; padding: 12px 22px 16px; }
.phase.open .phase-body { display: block; }
.phase.open .arrow { transform: rotate(90deg); }
.week-section { margin-bottom: 16px; }
.week-title { font-size: 0.9em; font-weight: 600; color: #374151; margin-bottom: 8px; padding: 6px 0; border-bottom: 1px solid #fde68a; }
.day-links { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px; }
.day-links a { display: block; padding: 7px 12px; background: white; border-radius: 7px; color: #b45309; text-decoration: none; font-size: 0.84em; border: 1px solid #fef3c7; transition: all 0.15s; }
.day-links a:hover { background: #fef3c7; text-decoration: none; }
.day-links a.plan { color: #059669; border-color: #a7f3d0; font-weight: 600; }
.coming-soon { color: #9ca3af; font-size: 0.85em; font-style: italic; padding: 8px 0; }
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.phase-header').forEach(function(h) {
    h.addEventListener('click', function() { h.parentElement.classList.toggle('open'); });
  });
  var first = document.querySelector('.phase');
  if (first) first.classList.add('open');
});
</script>

<div class="track-hero">
  <h1>🦾 Agent + Claw</h1>
  <p>从Agent原理到Claw深度实战，构建真实可用的智能体系统</p>
  <div class="stats">
    <div class="stat"><div class="num">6</div><div class="label">学习阶段</div></div>
    <div class="stat"><div class="num">52</div><div class="label">周学习</div></div>
    <div class="stat"><div class="num">364</div><div class="label">天内容</div></div>
  </div>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🤖 Phase 1 · Agent 基础</span>
    <span><span class="phase-meta">W1–W8 · Agent定义/ReAct/规划/记忆/工具调用</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <div class="week-section">
      <div class="week-title">Week 1 · 什么是 Agent</div>
      <div class="day-links">
        <a href="{{ '/agent/week01/agent-w1-what-is-agent' | relative_url }}">D1 · 什么是Agent</a>
        <a href="{{ '/agent/week01/agent-w1-react-framework' | relative_url }}">D2 · ReAct框架</a>
        <a href="{{ '/agent/week01/agent-w1-planning' | relative_url }}">D3 · 规划能力</a>
        <a href="{{ '/agent/week01/agent-w1-memory-system' | relative_url }}">D4 · 记忆系统</a>
        <a href="{{ '/agent/week01/agent-w1-tool-calling' | relative_url }}">D5 · 工具调用</a>
        <a href="{{ '/agent/week01/agent-w1-multi-agent' | relative_url }}">D6 · 多Agent协作</a>
        <a href="{{ '/agent/week01/agent-w1-capstone-project' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 2 · 主流框架：LangChain / AutoGen</div>
      <div class="day-links">
        <a href="{{ '/agent/week02/agent-w2-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/agent/week02/agent-w2-langchain-basics' | relative_url }}">D1 · LangChain基础</a>
        <a href="{{ '/agent/week02/agent-w2-langchain-chains' | relative_url }}">D2 · LangChain链</a>
        <a href="{{ '/agent/week02/agent-w2-llamaindex-intro' | relative_url }}">D3 · LlamaIndex</a>
        <a href="{{ '/agent/week02/agent-w2-autogen-multiagent' | relative_url }}">D4 · AutoGen多Agent</a>
        <a href="{{ '/agent/week02/agent-w2-crewai-roles' | relative_url }}">D5 · CrewAI角色</a>
        <a href="{{ '/agent/week02/agent-w2-framework-comparison' | relative_url }}">D6 · 框架横向对比</a>
        <a href="{{ '/agent/week02/agent-w2-capstone' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 3–8 · 深入各框架</div>
      <p class="coming-soon">🔜 即将更新…</p>
    </div>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🦞 Phase 2 · Claw 深度实战</span>
    <span><span class="phase-meta">W9–W16 · Claw架构/Skill编写/多Agent协作/任务分解</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🧠 Phase 3 · 高级模式</span>
    <span><span class="phase-meta">W17–W24 · 自我反思/长程规划/Human-in-loop</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🏭 Phase 4 · 垂直场景</span>
    <span><span class="phase-meta">W25–W34 · 代码Agent/数据分析/研究/运维</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🏗️ Phase 5 · 系统设计</span>
    <span><span class="phase-meta">W35–W44 · 多Agent系统架构/评测/生产部署</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🏆 Phase 6 · 综合项目</span>
    <span><span class="phase-meta">W45–W52 · 完整Agent系统落地</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>
