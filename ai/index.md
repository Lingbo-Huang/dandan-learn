---
layout: default
title: 🧠 AI 基础
---

<style>
.track-hero { background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.85; line-height: 1.6; }
.track-hero .stats { display: flex; gap: 32px; margin-top: 20px; }
.track-hero .stat { text-align: center; }
.track-hero .stat .num { font-size: 2em; font-weight: 700; }
.track-hero .stat .label { font-size: 0.82em; opacity: 0.75; }
.back { margin-bottom: 16px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }

.phase { margin-bottom: 12px; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.phase-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 22px; background: white; cursor: pointer;
  border-left: 5px solid #3b82f6; user-select: none;
}
.phase-header:hover { background: #f0f7ff; }
.phase-header .phase-title { font-weight: 600; font-size: 1em; color: #1e3a8a; }
.phase-header .phase-meta { font-size: 0.82em; color: #6b7280; }
.phase-header .arrow { transition: transform 0.2s; color: #3b82f6; font-size: 1.1em; }
.phase-body { display: none; background: #f8faff; padding: 12px 22px 16px; }
.phase.open .phase-body { display: block; }
.phase.open .arrow { transform: rotate(90deg); }

.week-section { margin-bottom: 16px; }
.week-title { font-size: 0.9em; font-weight: 600; color: #374151; margin-bottom: 8px; padding: 6px 0; border-bottom: 1px solid #e0eaff; }
.day-links { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px; }
.day-links a {
  display: block; padding: 7px 12px; background: white;
  border-radius: 7px; color: #1d4ed8; text-decoration: none;
  font-size: 0.84em; border: 1px solid #dbeafe;
  transition: all 0.15s;
}
.day-links a:hover { background: #dbeafe; border-color: #93c5fd; text-decoration: none; }
.day-links a.plan { color: #059669; border-color: #a7f3d0; }
.day-links a.plan:hover { background: #d1fae5; }

.coming-soon { color: #9ca3af; font-size: 0.85em; font-style: italic; padding: 8px 0; }
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.phase-header').forEach(function(h) {
    h.addEventListener('click', function() {
      h.parentElement.classList.toggle('open');
    });
  });
  // 默认展开第一个
  var first = document.querySelector('.phase');
  if (first) first.classList.add('open');
});
</script>

<div class="track-hero">
  <h1>🧠 AI 基础</h1>
  <p>从数学基础到前沿方向，打牢所有AI知识的底层根基</p>
  <div class="stats">
    <div class="stat"><div class="num">6</div><div class="label">学习阶段</div></div>
    <div class="stat"><div class="num">52</div><div class="label">周学习</div></div>
    <div class="stat"><div class="num">364</div><div class="label">天内容</div></div>
  </div>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">📐 Phase 1 · 数学基础</span>
    <span><span class="phase-meta">W1–W8 · 线性代数、微积分、概率统计、信息论</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <div class="week-section">
      <div class="week-title">Week 1 · 线性代数①：向量与矩阵</div>
      <div class="day-links">
        <a href="{{ '/ai/week01/ai-w1-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/ai/week01/ai-w1-vector-basics' | relative_url }}">D1 · 向量基础</a>
        <a href="{{ '/ai/week01/ai-w1-vector-advanced' | relative_url }}">D2 · 向量进阶</a>
        <a href="{{ '/ai/week01/ai-w1-matrix-basics' | relative_url }}">D3 · 矩阵基础</a>
        <a href="{{ '/ai/week01/ai-w1-matrix-multiplication' | relative_url }}">D4 · 矩阵乘法</a>
        <a href="{{ '/ai/week01/ai-w1-determinant' | relative_url }}">D5 · 行列式</a>
        <a href="{{ '/ai/week01/ai-w1-inverse-rank' | relative_url }}">D6 · 逆矩阵与秩</a>
        <a href="{{ '/ai/week01/ai-w1-capstone' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 2 · 线性代数②：特征值与SVD</div>
      <div class="day-links">
        <a href="{{ '/ai/week02/ai-w2-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/ai/week02/ai-w2-eigenvalues' | relative_url }}">D1 · 特征值与特征向量</a>
        <a href="{{ '/ai/week02/ai-w2-eigenvectors' | relative_url }}">D2 · 特征向量深入</a>
        <a href="{{ '/ai/week02/ai-w2-diagonalization' | relative_url }}">D3 · 矩阵对角化</a>
        <a href="{{ '/ai/week02/ai-w2-svd-theory' | relative_url }}">D4 · SVD理论</a>
        <a href="{{ '/ai/week02/ai-w2-svd-computation' | relative_url }}">D5 · SVD计算</a>
        <a href="{{ '/ai/week02/ai-w2-pca-from-svd' | relative_url }}">D6 · PCA与SVD</a>
        <a href="{{ '/ai/week02/ai-w2-capstone' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 3–8 · 微积分 / 概率统计 / 信息论</div>
      <p class="coming-soon">🔜 即将更新，敬请期待…</p>
    </div>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🤖 Phase 2 · 机器学习</span>
    <span><span class="phase-meta">W9–W18 · 回归/分类/树模型/集成/SVM/聚类/降维</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <p class="coming-soon">🔜 Week 1-8 学完后开启，内容持续生成中…</p>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🧠 Phase 3 · 深度学习</span>
    <span><span class="phase-meta">W19–W28 · 神经网络/CNN/RNN/注意力/PyTorch</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <p class="coming-soon">🔜 即将开启…</p>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🖼️ Phase 4 · CV / NLP 应用</span>
    <span><span class="phase-meta">W29–W38 · 目标检测/语义分割/文本分类/机器翻译</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <p class="coming-soon">🔜 即将开启…</p>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🚀 Phase 5 · 前沿方向</span>
    <span><span class="phase-meta">W39–W46 · Diffusion/多模态/RLHF/自监督</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <p class="coming-soon">🔜 即将开启…</p>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🏆 Phase 6 · 综合项目</span>
    <span><span class="phase-meta">W47–W52 · 完整AI应用落地</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <p class="coming-soon">🔜 即将开启…</p>
  </div>
</div>
