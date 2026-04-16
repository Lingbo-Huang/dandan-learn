---
layout: default
title: ⚙️ AI Infra
---

<style>
.track-hero { background: linear-gradient(135deg, #7f1d1d, #ef4444); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.9; font-size: 1em; line-height: 1.6; }
.week-card { background: white; border-radius: 14px; padding: 24px 28px; margin-bottom: 24px; box-shadow: 0 2px 16px rgba(0,0,0,0.07); border-left: 5px solid #ef4444; }
.week-card h2 { font-size: 1.15em; margin-bottom: 14px; color: #7f1d1d; }
.week-card ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
.week-card ul li a { display: block; padding: 8px 12px; background: #fff1f2; border-radius: 8px; color: #b91c1c; text-decoration: none; font-size: 0.88em; transition: background 0.15s; }
.week-card ul li a:hover { background: #fee2e2; text-decoration: none; }
.back { margin-bottom: 8px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }
</style>

<div class="track-hero">
  <h1>⚙️ AI Infra</h1>
  <p>GPU架构 → 训练系统 → 推理系统 → MLOps → 大规模系统 → 前沿架构<br>深入AI基础设施的每一层，从CUDA到千卡训练</p>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="week-card">
  <h2>📅 Week 1 · GPU架构与CUDA基础</h2>
  <ul>
    <li><a href="{{ '/infra/week01/infra-w1-gpu-vs-cpu' | relative_url }}">D1 · GPU vs CPU</a></li>
    <li><a href="{{ '/infra/week01/infra-w1-sm-architecture' | relative_url }}">D2 · SM架构</a></li>
    <li><a href="{{ '/infra/week01/infra-w1-cuda-programming-model' | relative_url }}">D3 · CUDA编程模型</a></li>
    <li><a href="{{ '/infra/week01/infra-w1-first-cuda-kernel' | relative_url }}">D4 · 第一个CUDA Kernel</a></li>
    <li><a href="{{ '/infra/week01/infra-w1-memory-hierarchy-reduction' | relative_url }}">D5 · 内存层次与规约</a></li>
    <li><a href="{{ '/infra/week01/infra-w1-nsight-profiling' | relative_url }}">D6 · Nsight性能分析</a></li>
    <li><a href="{{ '/infra/week01/infra-w1-sgemm-week-review' | relative_url }}">D7 · SGEMM综合实战</a></li>
  </ul>
</div>
