---
layout: default
title: ⚙️ AI Infra
---

<style>
.track-hero { background: linear-gradient(135deg, #7f1d1d, #ef4444); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.9; line-height: 1.6; }
.track-hero .stats { display: flex; gap: 32px; margin-top: 20px; }
.track-hero .stat { text-align: center; }
.track-hero .stat .num { font-size: 2em; font-weight: 700; }
.track-hero .stat .label { font-size: 0.82em; opacity: 0.75; }
.back { margin-bottom: 16px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }
.phase { margin-bottom: 12px; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.phase-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 22px; background: white; cursor: pointer; border-left: 5px solid #ef4444; user-select: none; }
.phase-header:hover { background: #fff1f2; }
.phase-header .phase-title { font-weight: 600; font-size: 1em; color: #7f1d1d; }
.phase-header .phase-meta { font-size: 0.82em; color: #6b7280; }
.phase-header .arrow { transition: transform 0.2s; color: #ef4444; }
.phase-body { display: none; background: #fff1f2; padding: 12px 22px 16px; }
.phase.open .phase-body { display: block; }
.phase.open .arrow { transform: rotate(90deg); }
.week-section { margin-bottom: 16px; }
.week-title { font-size: 0.9em; font-weight: 600; color: #374151; margin-bottom: 8px; padding: 6px 0; border-bottom: 1px solid #fecaca; }
.day-links { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px; }
.day-links a { display: block; padding: 7px 12px; background: white; border-radius: 7px; color: #b91c1c; text-decoration: none; font-size: 0.84em; border: 1px solid #fee2e2; transition: all 0.15s; }
.day-links a:hover { background: #fee2e2; text-decoration: none; }
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
  <h1>⚙️ AI Infra</h1>
  <p>从GPU架构到千卡训练，深入AI基础设施的每一层</p>
  <div class="stats">
    <div class="stat"><div class="num">6</div><div class="label">学习阶段</div></div>
    <div class="stat"><div class="num">52</div><div class="label">周学习</div></div>
    <div class="stat"><div class="num">364</div><div class="label">天内容</div></div>
  </div>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🖥️ Phase 1 · 基础设施</span>
    <span><span class="phase-meta">W1–W8 · GPU架构/CUDA/分布式训练原理</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <div class="week-section">
      <div class="week-title">Week 1 · GPU架构与CUDA基础</div>
      <div class="day-links">
        <a href="{{ '/infra/week01/infra-w1-gpu-vs-cpu' | relative_url }}">D1 · GPU vs CPU</a>
        <a href="{{ '/infra/week01/infra-w1-sm-architecture' | relative_url }}">D2 · SM架构</a>
        <a href="{{ '/infra/week01/infra-w1-cuda-programming-model' | relative_url }}">D3 · CUDA编程模型</a>
        <a href="{{ '/infra/week01/infra-w1-first-cuda-kernel' | relative_url }}">D4 · 第一个CUDA Kernel</a>
        <a href="{{ '/infra/week01/infra-w1-memory-hierarchy-reduction' | relative_url }}">D5 · 内存层次与规约</a>
        <a href="{{ '/infra/week01/infra-w1-nsight-profiling' | relative_url }}">D6 · Nsight性能分析</a>
        <a href="{{ '/infra/week01/infra-w1-sgemm-week-review' | relative_url }}">D7 · SGEMM综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 2 · 分布式训练基础</div>
      <div class="day-links">
        <a href="{{ '/infra/week02/infra-w2-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/infra/week02/infra-w2-distributed-overview' | relative_url }}">D1 · 分布式训练概览</a>
        <a href="{{ '/infra/week02/infra-w2-data-parallelism' | relative_url }}">D2 · 数据并行</a>
        <a href="{{ '/infra/week02/infra-w2-model-parallelism' | relative_url }}">D3 · 模型并行</a>
        <a href="{{ '/infra/week02/infra-w2-zero-stages' | relative_url }}">D4 · ZeRO三个阶段</a>
        <a href="{{ '/infra/week02/infra-w2-deepspeed-intro' | relative_url }}">D5 · DeepSpeed入门</a>
        <a href="{{ '/infra/week02/infra-w2-mixed-precision' | relative_url }}">D6 · 混合精度训练</a>
        <a href="{{ '/infra/week02/infra-w2-capstone' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 3–8 · 更多基础内容</div>
      <p class="coming-soon">🔜 即将更新…</p>
    </div>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🚂 Phase 2 · 训练系统</span>
    <span><span class="phase-meta">W9–W16 · DeepSpeed/Megatron/混合精度/ZeRO</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">⚡ Phase 3 · 推理系统</span>
    <span><span class="phase-meta">W17–W24 · TensorRT/vLLM/Triton/推理服务</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🔧 Phase 4 · MLOps</span>
    <span><span class="phase-meta">W25–W32 · 实验管理/数据流水线/训练监控</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🌐 Phase 5 · 大规模系统</span>
    <span><span class="phase-meta">W33–W42 · 千卡训练/通信优化/故障恢复</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🔬 Phase 6 · 前沿架构</span>
    <span><span class="phase-meta">W43–W52 · FlashAttention/PagedAttention/编译器优化</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>
