---
layout: default
title: 📈 量化 Quant
---

<style>
.track-hero { background: linear-gradient(135deg, #064e3b, #10b981); color: white; padding: 48px 32px 40px; margin: -32px -24px 40px; border-radius: 0 0 20px 20px; }
.track-hero h1 { font-size: 2em; margin-bottom: 10px; }
.track-hero p { opacity: 0.85; line-height: 1.6; }
.track-hero .stats { display: flex; gap: 32px; margin-top: 20px; }
.track-hero .stat { text-align: center; }
.track-hero .stat .num { font-size: 2em; font-weight: 700; }
.track-hero .stat .label { font-size: 0.82em; opacity: 0.75; }
.back { margin-bottom: 16px; font-size: 0.88em; }
.back a { color: #6b7280; text-decoration: none; }
.phase { margin-bottom: 12px; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.phase-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 22px; background: white; cursor: pointer; border-left: 5px solid #10b981; user-select: none; }
.phase-header:hover { background: #f0fdf4; }
.phase-header .phase-title { font-weight: 600; font-size: 1em; color: #064e3b; }
.phase-header .phase-meta { font-size: 0.82em; color: #6b7280; }
.phase-header .arrow { transition: transform 0.2s; color: #10b981; }
.phase-body { display: none; background: #f0fdf4; padding: 12px 22px 16px; }
.phase.open .phase-body { display: block; }
.phase.open .arrow { transform: rotate(90deg); }
.week-section { margin-bottom: 16px; }
.week-title { font-size: 0.9em; font-weight: 600; color: #374151; margin-bottom: 8px; padding: 6px 0; border-bottom: 1px solid #a7f3d0; }
.day-links { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px; }
.day-links a { display: block; padding: 7px 12px; background: white; border-radius: 7px; color: #065f46; text-decoration: none; font-size: 0.84em; border: 1px solid #d1fae5; transition: all 0.15s; }
.day-links a:hover { background: #d1fae5; text-decoration: none; }
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
  <h1>📈 量化 Quant</h1>
  <p>从金融基础到AI量化融合，构建完整量化投资体系</p>
  <div class="stats">
    <div class="stat"><div class="num">6</div><div class="label">学习阶段</div></div>
    <div class="stat"><div class="num">52</div><div class="label">周学习</div></div>
    <div class="stat"><div class="num">364</div><div class="label">天内容</div></div>
  </div>
</div>

<p class="back"><a href="{{ '/' | relative_url }}">← 返回首页</a></p>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">💹 Phase 1 · 金融与市场基础</span>
    <span><span class="phase-meta">W1–W8 · 市场结构/收益率/技术分析/基本面</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body">
    <div class="week-section">
      <div class="week-title">Week 1 · 金融市场全景</div>
      <div class="day-links">
        <a href="{{ '/quant/week01/quant-w1-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/quant/week01/quant-w1-market-overview' | relative_url }}">D1 · 金融市场全景</a>
        <a href="{{ '/quant/week01/quant-w1-returns' | relative_url }}">D2 · 价格与收益率</a>
        <a href="{{ '/quant/week01/quant-w1-technical-analysis' | relative_url }}">D3 · 技术分析基础</a>
        <a href="{{ '/quant/week01/quant-w1-fundamentals' | relative_url }}">D4 · 基本面基础</a>
        <a href="{{ '/quant/week01/quant-w1-market-microstructure' | relative_url }}">D5 · 市场微观结构</a>
        <a href="{{ '/quant/week01/quant-w1-time-series' | relative_url }}">D6 · 时间序列统计</a>
        <a href="{{ '/quant/week01/quant-w1-data-tools' | relative_url }}">D7 · 数据工具实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 2 · 价格与收益率深度统计</div>
      <div class="day-links">
        <a href="{{ '/quant/week02/quant-w2-week-plan' | relative_url }}" class="plan">📋 周规划总览</a>
        <a href="{{ '/quant/week02/quant-w2-log-returns' | relative_url }}">D1 · 对数收益率</a>
        <a href="{{ '/quant/week02/quant-w2-normal-hypothesis' | relative_url }}">D2 · 正态假设</a>
        <a href="{{ '/quant/week02/quant-w2-fat-tails' | relative_url }}">D3 · 胖尾分布</a>
        <a href="{{ '/quant/week02/quant-w2-risk-measures' | relative_url }}">D4 · 风险度量</a>
        <a href="{{ '/quant/week02/quant-w2-correlation' | relative_url }}">D5 · 相关性分析</a>
        <a href="{{ '/quant/week02/quant-w2-data-practice' | relative_url }}">D6 · 数据实战</a>
        <a href="{{ '/quant/week02/quant-w2-capstone' | relative_url }}">D7 · 综合实战</a>
      </div>
    </div>
    <div class="week-section">
      <div class="week-title">Week 3–8 · 市场微观结构/数据工具</div>
      <p class="coming-soon">🔜 即将更新…</p>
    </div>
  </div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">📊 Phase 2 · 因子与策略</span>
    <span><span class="phase-meta">W9–W18 · 动量/价值/质量/CTA/统计套利</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🔁 Phase 3 · 回测体系</span>
    <span><span class="phase-meta">W19–W26 · Backtrader/Qlib/过拟合/Walk-Forward</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🛡️ Phase 4 · 风控与实盘</span>
    <span><span class="phase-meta">W27–W34 · 仓位/VaR/VWAP/实盘接口</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🤖 Phase 5 · AI量化融合</span>
    <span><span class="phase-meta">W35–W44 · ML选股/时序预测/GNN/NLP因子</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>

<div class="phase">
  <div class="phase-header">
    <span class="phase-title">🏆 Phase 6 · 综合策略</span>
    <span><span class="phase-meta">W45–W52 · 多策略组合/归因/宏观量化</span> <span class="arrow">▶</span></span>
  </div>
  <div class="phase-body"><p class="coming-soon">🔜 即将开启…</p></div>
</div>
