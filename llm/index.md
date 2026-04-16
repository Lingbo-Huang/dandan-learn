---
layout: default
title: 🤖 大模型 LLM
---

<style>
.track-hero { background: linear-gradient(135deg, #4c1d95, #8b5cf6); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.85; font-size: 1em; line-height: 1.6; }
.week-card { background: white; border-radius: 14px; padding: 24px 28px; margin-bottom: 24px; box-shadow: 0 2px 16px rgba(0,0,0,0.07); border-left: 5px solid #8b5cf6; }
.week-card h2 { font-size: 1.15em; margin-bottom: 14px; color: #4c1d95; }
.week-card ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
.week-card ul li a { display: block; padding: 8px 12px; background: #f5f3ff; border-radius: 8px; color: #6d28d9; text-decoration: none; font-size: 0.88em; transition: background 0.15s; }
.week-card ul li a:hover { background: #ede9fe; text-decoration: none; }
.back { margin-bottom: 8px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }
</style>

<div class="track-hero">
  <h1>🤖 大模型 LLM</h1>
  <p>Transformer → 预训练 → 微调对齐 → 推理部署 → 应用工程 → 前沿研究<br>52周深入大模型的每一个细节</p>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="week-card">
  <h2>📅 Week 1 · Transformer 总览</h2>
  <ul>
    <li><a href="{{ '/llm/week01/llm-w1-week-plan' | relative_url }}">📋 周规划总览</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-transformer-overview' | relative_url }}">D1 · Transformer全景</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-tokenization-embedding' | relative_url }}">D2 · 输入处理</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-self-attention' | relative_url }}">D3 · 自注意力机制</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-ffn-layernorm' | relative_url }}">D4 · FFN与LayerNorm</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-encoder-decoder' | relative_url }}">D5 · 编码器解码器</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-training-practice' | relative_url }}">D6 · 训练流程实战</a></li>
    <li><a href="{{ '/llm/week01/llm-w1-review' | relative_url }}">D7 · 综合复习</a></li>
  </ul>
</div>
