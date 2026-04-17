---
layout: default
title: 🤖 大模型 LLM
---

<style>
.track-hero { background: linear-gradient(135deg, #4c1d95, #8b5cf6); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.85; line-height: 1.6; }
.track-hero .stats { display: flex; gap: 32px; margin-top: 20px; }
.track-hero .stat { text-align: center; }
.track-hero .stat .num { font-size: 2em; font-weight: 700; }
.track-hero .stat .label { font-size: 0.82em; opacity: 0.75; }
.back { margin-bottom: 16px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }
.phase { margin-bottom: 12px; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.phase-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 22px; background: white; cursor: pointer; border-left: 5px solid #8b5cf6; user-select: none; }
.phase-header:hover { background: #f5f3ff; }
.phase-header .phase-title { font-weight: 600; font-size: 1em; color: #4c1d95; }
.phase-header .phase-meta { font-size: 0.82em; color: #6b7280; }
.phase-header .arrow { transition: transform 0.2s; color: #8b5cf6; }
.phase-body { display: none; background: #faf8ff; padding: 12px 22px 16px; }
.phase.open .phase-body { display: block; }
.phase.open .arrow { transform: rotate(90deg); }
.week-section { margin-bottom: 16px; }
.week-title { font-size: 0.9em; font-weight: 600; color: #374151; margin-bottom: 8px; padding: 6px 0; border-bottom: 1px solid #e9d5ff; }
.day-links { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px; }
.day-links a { display: block; padding: 7px 12px; background: white; border-radius: 7px; color: #6d28d9; text-decoration: none; font-size: 0.84em; border: 1px solid #ede9fe; transition: all 0.15s; }
.day-links a:hover { background: #ede9fe; text-decoration: none; }
.day-links a.plan { color: #059669; border-color: #a7f3d0; }
.day-links a.plan:hover { background: #d1fae5; }
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
  <h1>🤖 大模型 LLM</h1>
  <p>从Transformer原理到前沿研究，深入大模型的每一个细节</p>
  <div class="stats">
    <div class="stat"><div class="num">6</div><div class="label">学习阶段</div></div>
    <div class="stat"><div class="num">52</div><div class="label">周学习</div></div>
    <div class="stat"><div class="num">364</div><div class="label">天内容</div></div>
  </div>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🏗️ Phase 1 · 架构基础</span>
    <span><span class="phase-meta">W1–W8 · Transformer/Attention/位置编码/Tokenizer</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <div class="week-section">
      <div class="week-title">Week 1 · Transformer 总览</div>
      <div class="day-links">
        <a href="{{ '/llm/week01/llm-w1-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/llm/week01/llm-w1-transformer-overview' | relative_url }}">D1 · Transformer全景</a>
        <a href="{{ '/llm/week01/llm-w1-tokenization-embedding' | relative_url }}">D2 · 输入处理</a>
        <a href="{{ '/llm/week01/llm-w1-self-attention' | relative_url }}">D3 · 自注意力机制</a>
        <a href="{{ '/llm/week01/llm-w1-ffn-layernorm' | relative_url }}">D4 · FFN与LayerNorm</a>
        <a href="{{ '/llm/week01/llm-w1-encoder-decoder' | relative_url }}">D5 · 编码器解码器</a>
        <a href="{{ '/llm/week01/llm-w1-training-practice' | relative_url }}">D6 · 训练流程实战</a>
        <a href="{{ '/llm/week01/llm-w1-review' | relative_url }}">D7 · 综合复习</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 2 · Attention 机制深度推导</div>
      <div class="day-links">
        <a href="{{ '/llm/week02/llm-w2-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/llm/week02/llm-w2-attention-intuition' | relative_url }}">D1 · Attention直觉</a>
        <a href="{{ '/llm/week02/llm-w2-scaled-dot-product' | relative_url }}">D2 · Scaled Dot-Product</a>
        <a href="{{ '/llm/week02/llm-w2-multi-head-attention' | relative_url }}">D3 · 多头注意力</a>
        <a href="{{ '/llm/week02/llm-w2-attention-complexity' | relative_url }}">D4 · 复杂度分析</a>
        <a href="{{ '/llm/week02/llm-w2-attention-variants' | relative_url }}">D5 · Attention变体</a>
        <a href="{{ '/llm/week02/llm-w2-attention-code' | relative_url }}">D6 · 代码实现</a>
        <a href="{{ '/llm/week02/llm-w2-capstone' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 3–8 · 位置编码 / Tokenizer / Scaling Law</div>
      <p class="coming-soon">🔜 即将更新…</p>
    </div>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">📚 Phase 2 · 主流模型</span>
    <span><span class="phase-meta">W9–W16 · BERT/GPT/LLaMA/Qwen/MoE</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🔧 Phase 3 · 微调与对齐</span>
    <span><span class="phase-meta">W17–W26 · SFT/LoRA/RLHF/DPO</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">⚡ Phase 4 · 推理与部署</span>
    <span><span class="phase-meta">W27–W34 · KV Cache/量化/vLLM/模型并行</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🛠️ Phase 5 · 应用工程</span>
    <span><span class="phase-meta">W35–W44 · Prompt/RAG/Function Call/LLMOps</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🔬 Phase 6 · 前沿研究</span>
    <span><span class="phase-meta">W45–W52 · o1推理/World Model/多模态</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>
