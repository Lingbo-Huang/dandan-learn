---
layout: default
title: AI Infra
---

<div class="track-hero c-infra">
  <div class="eyebrow" style="color:#ff7b72">⚙️ AI Infra · @🥚🥚5号：ai infra 主讲</div>
  <h1>从 GPU 到千卡训练</h1>
  <p>深入 AI 基础设施的每一层——GPU 架构、CUDA 编程、分布式训练、推理系统到大规模系统设计</p>
  <div class="track-stats">
    <div class="track-stat"><div class="num" style="color:#ff7b72">6</div><div class="label">学习阶段</div></div>
    <div class="track-stat"><div class="num" style="color:#ff7b72">52</div><div class="label">周学习</div></div>
    <div class="track-stat"><div class="num" style="color:#ff7b72">364</div><div class="label">天内容</div></div>
  </div>
</div>

<div class="phases c-infra">
  <div class="phase c-infra">
    <div class="phase-header">
      <div class="phase-left">
        <div class="phase-icon" style="background:rgba(255,123,114,0.1)">🖥️</div>
        <div><div class="phase-title">Phase 1 · 基础设施</div><div class="phase-meta">W1–W8 · GPU架构/CUDA编程/分布式训练原理</div></div>
      </div>
      <div class="phase-right"><span class="phase-progress">8 / 8 周</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body">
      <div class="week-block">
        <div class="week-label">Week 1 · GPU 架构与 CUDA 基础</div>
        <div class="day-grid">
          <a href="{{ '/infra/week01/infra-w1-gpu-vs-cpu' | relative_url }}" class="day-link">D1 · GPU vs CPU</a>
          <a href="{{ '/infra/week01/infra-w1-sm-architecture' | relative_url }}" class="day-link">D2 · SM架构</a>
          <a href="{{ '/infra/week01/infra-w1-cuda-programming-model' | relative_url }}" class="day-link">D3 · CUDA编程模型</a>
          <a href="{{ '/infra/week01/infra-w1-first-cuda-kernel' | relative_url }}" class="day-link">D4 · 第一个Kernel</a>
          <a href="{{ '/infra/week01/infra-w1-memory-hierarchy-reduction' | relative_url }}" class="day-link">D5 · 内存层次</a>
          <a href="{{ '/infra/week01/infra-w1-nsight-profiling' | relative_url }}" class="day-link">D6 · 性能分析</a>
          <a href="{{ '/infra/week01/infra-w1-sgemm-week-review' | relative_url }}" class="day-link">D7 · SGEMM实战</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 2 · 分布式训练基础</div>
        <div class="day-grid">
          <a href="{{ '/infra/week02/infra-w2-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week02/infra-w2-distributed-overview' | relative_url }}" class="day-link">D1 · 分布式概览</a>
          <a href="{{ '/infra/week02/infra-w2-data-parallelism' | relative_url }}" class="day-link">D2 · 数据并行</a>
          <a href="{{ '/infra/week02/infra-w2-model-parallelism' | relative_url }}" class="day-link">D3 · 模型并行</a>
          <a href="{{ '/infra/week02/infra-w2-zero-stages' | relative_url }}" class="day-link">D4 · ZeRO三阶段</a>
          <a href="{{ '/infra/week02/infra-w2-deepspeed-intro' | relative_url }}" class="day-link">D5 · DeepSpeed</a>
          <a href="{{ '/infra/week02/infra-w2-mixed-precision' | relative_url }}" class="day-link">D6 · 混合精度</a>
          <a href="{{ '/infra/week02/infra-w2-capstone' | relative_url }}" class="day-link">D7 · 综合实战</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 3 · cuBLAS / cuDNN 深度解析</div>
        <div class="day-grid">
          <a href="{{ '/infra/week03/infra-w3-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week03/infra-w3-cublas-gemm' | relative_url }}" class="day-link">D1 · GEMM 深度解析</a>
          <a href="{{ '/infra/week03/infra-w3-cublas-api' | relative_url }}" class="day-link">D2 · cuBLAS API 调优</a>
          <a href="{{ '/infra/week03/infra-w3-cudnn-conv' | relative_url }}" class="day-link">D3 · cuDNN 卷积算法</a>
          <a href="{{ '/infra/week03/infra-w3-workspace-fusion' | relative_url }}" class="day-link">D4 · 算子融合</a>
          <a href="{{ '/infra/week03/infra-w3-memory-layout' | relative_url }}" class="day-link">D5 · NHWC vs NCHW</a>
          <a href="{{ '/infra/week03/infra-w3-profiling' | relative_url }}" class="day-link">D6 · 综合调优</a>
          <a href="{{ '/infra/week03/infra-w3-capstone' | relative_url }}" class="day-link">D7 · 综合实战</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 4 · FlashAttention：IO 感知 Attention</div>
        <div class="day-grid">
          <a href="{{ '/infra/week04/infra-w4-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week04/infra-w4-attention-bottleneck' | relative_url }}" class="day-link">D1 · Attention 瓶颈</a>
          <a href="{{ '/infra/week04/infra-w4-tiling-online-softmax' | relative_url }}" class="day-link">D2 · Tiling &amp; Online Softmax</a>
          <a href="{{ '/infra/week04/infra-w4-flashattn1-cuda' | relative_url }}" class="day-link">D3 · FlashAttn1 CUDA</a>
          <a href="{{ '/infra/week04/infra-w4-flashattn2-improvements' | relative_url }}" class="day-link">D4 · FlashAttn2 改进</a>
          <a href="{{ '/infra/week04/infra-w4-flashattn3-h100' | relative_url }}" class="day-link">D5 · FlashAttn3 &amp; H100</a>
          <a href="{{ '/infra/week04/infra-w4-capstone' | relative_url }}" class="day-link">D6 · Capstone</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 5 · 推理优化：KV Cache / vLLM / 投机解码</div>
        <div class="day-grid">
          <a href="{{ '/infra/week05/infra-w5-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week05/infra-w5-kvcache-fundamentals' | relative_url }}" class="day-link">D1 · KV Cache 基础</a>
          <a href="{{ '/infra/week05/infra-w5-pagedattention' | relative_url }}" class="day-link">D2 · PagedAttention</a>
          <a href="{{ '/infra/week05/infra-w5-continuous-batching' | relative_url }}" class="day-link">D3 · 连续批处理</a>
          <a href="{{ '/infra/week05/infra-w5-speculative-decoding' | relative_url }}" class="day-link">D4 · 投机解码</a>
          <a href="{{ '/infra/week05/infra-w5-medusa-lookahead' | relative_url }}" class="day-link">D5 · Medusa &amp; Lookahead</a>
          <a href="{{ '/infra/week05/infra-w5-capstone' | relative_url }}" class="day-link">D6 · Capstone</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 6 · 分布式训练进阶：3D 并行 / MoE / 专家并行</div>
        <div class="day-grid">
          <a href="{{ '/infra/week06/infra-w6-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week06/infra-w6-megatron-3d-parallel' | relative_url }}" class="day-link">D1 · Megatron 3D 并行</a>
          <a href="{{ '/infra/week06/infra-w6-sequence-parallel' | relative_url }}" class="day-link">D2 · 序列并行</a>
          <a href="{{ '/infra/week06/infra-w6-pipeline-schedule' | relative_url }}" class="day-link">D3 · Pipeline 调度</a>
          <a href="{{ '/infra/week06/infra-w6-moe-architecture' | relative_url }}" class="day-link">D4 · MoE 架构</a>
          <a href="{{ '/infra/week06/infra-w6-expert-parallel' | relative_url }}" class="day-link">D5 · 专家并行</a>
          <a href="{{ '/infra/week06/infra-w6-capstone' | relative_url }}" class="day-link">D6 · Capstone</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 7 · 量化与压缩：INT8 / INT4 / AWQ / GPTQ</div>
        <div class="day-grid">
          <a href="{{ '/infra/week07/infra-w7-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week07/infra-w7-quantization-basics' | relative_url }}" class="day-link">D1 · 量化基础</a>
          <a href="{{ '/infra/week07/infra-w7-smoothquant' | relative_url }}" class="day-link">D2 · SmoothQuant</a>
          <a href="{{ '/infra/week07/infra-w7-gptq' | relative_url }}" class="day-link">D3 · GPTQ</a>
          <a href="{{ '/infra/week07/infra-w7-awq' | relative_url }}" class="day-link">D4 · AWQ</a>
          <a href="{{ '/infra/week07/infra-w7-practical-quantization' | relative_url }}" class="day-link">D5 · 实战工具链</a>
          <a href="{{ '/infra/week07/infra-w7-capstone' | relative_url }}" class="day-link">D6 · Capstone</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 8 · MLOps 与系统设计：监控 / 故障恢复 / 面试</div>
        <div class="day-grid">
          <a href="{{ '/infra/week08/infra-w8-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/infra/week08/infra-w8-training-monitoring' | relative_url }}" class="day-link">D1 · 训练监控</a>
          <a href="{{ '/infra/week08/infra-w8-fault-tolerance' | relative_url }}" class="day-link">D2 · 故障恢复</a>
          <a href="{{ '/infra/week08/infra-w8-cost-optimization' | relative_url }}" class="day-link">D3 · 成本优化</a>
          <a href="{{ '/infra/week08/infra-w8-system-design-llm' | relative_url }}" class="day-link">D4 · LLM 系统设计</a>
          <a href="{{ '/infra/week08/infra-w8-interview-questions' | relative_url }}" class="day-link">D5 · 面试精讲</a>
          <a href="{{ '/infra/week08/infra-w8-capstone' | relative_url }}" class="day-link">D6 · 全面复习</a>
        </div>
      </div>
    </div>
  </div>
  <div class="phase c-infra"><div class="phase-header"><div class="phase-left"><div class="phase-icon" style="background:rgba(255,123,114,0.1)">🚂</div><div><div class="phase-title">Phase 2 · 训练系统</div><div class="phase-meta">W9–W16 · DeepSpeed/Megatron/ZeRO</div></div></div><div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div></div><div class="phase-body"><p class="coming">🔜 即将开启…</p></div></div>
  <div class="phase c-infra"><div class="phase-header"><div class="phase-left"><div class="phase-icon" style="background:rgba(255,123,114,0.1)">⚡</div><div><div class="phase-title">Phase 3–6 · 推理/MLOps/大规模/前沿架构</div><div class="phase-meta">W17–W52</div></div></div><div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div></div><div class="phase-body"><p class="coming">🔜 即将开启…</p></div></div>
</div>
