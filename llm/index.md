---
layout: default
title: 大模型 LLM
---

<div class="track-hero c-llm">
  <div class="eyebrow" style="color:#bc8cff">🤖 大模型 LLM · @🥚🥚2号:llm 主讲</div>
  <h1>从 Transformer 到前沿研究</h1>
  <p>深入大模型的每一个细节——架构原理、预训练、微调对齐、推理部署、应用工程到前沿研究</p>
  <div class="track-stats">
    <div class="track-stat"><div class="num" style="color:#bc8cff">6</div><div class="label">学习阶段</div></div>
    <div class="track-stat"><div class="num" style="color:#bc8cff">52</div><div class="label">周学习</div></div>
    <div class="track-stat"><div class="num" style="color:#bc8cff">364</div><div class="label">天内容</div></div>
  </div>
</div>

<div class="phases c-llm">

  <div class="phase c-llm">
    <div class="phase-header">
      <div class="phase-left">
        <div class="phase-icon" style="background:rgba(188,140,255,0.1)">🏗️</div>
        <div>
          <div class="phase-title">Phase 1 · 架构基础</div>
          <div class="phase-meta">W1–W8 · Transformer/Attention/位置编码/Tokenizer/Scaling Law</div>
        </div>
      </div>
      <div class="phase-right"><span class="phase-progress">8 / 8 周</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body">
      <div class="week-block">
        <div class="week-label">Week 1 · Transformer 总览</div>
        <div class="day-grid">
          <a href="{{ '/llm/week01/llm-w1-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week01/llm-w1-transformer-overview' | relative_url }}" class="day-link">D1 · Transformer全景</a>
          <a href="{{ '/llm/week01/llm-w1-tokenization-embedding' | relative_url }}" class="day-link">D2 · 输入处理</a>
          <a href="{{ '/llm/week01/llm-w1-self-attention' | relative_url }}" class="day-link">D3 · 自注意力机制</a>
          <a href="{{ '/llm/week01/llm-w1-ffn-layernorm' | relative_url }}" class="day-link">D4 · FFN与LayerNorm</a>
          <a href="{{ '/llm/week01/llm-w1-encoder-decoder' | relative_url }}" class="day-link">D5 · 编码器解码器</a>
          <a href="{{ '/llm/week01/llm-w1-training-practice' | relative_url }}" class="day-link">D6 · 训练流程实战</a>
          <a href="{{ '/llm/week01/llm-w1-review' | relative_url }}" class="day-link">D7 · 综合复习</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 2 · Attention 机制深度推导</div>
        <div class="day-grid">
          <a href="{{ '/llm/week02/llm-w2-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week02/llm-w2-attention-intuition' | relative_url }}" class="day-link">D1 · Attention直觉</a>
          <a href="{{ '/llm/week02/llm-w2-scaled-dot-product' | relative_url }}" class="day-link">D2 · Scaled Dot-Product</a>
          <a href="{{ '/llm/week02/llm-w2-multi-head-attention' | relative_url }}" class="day-link">D3 · 多头注意力</a>
          <a href="{{ '/llm/week02/llm-w2-attention-complexity' | relative_url }}" class="day-link">D4 · 复杂度分析</a>
          <a href="{{ '/llm/week02/llm-w2-attention-variants' | relative_url }}" class="day-link">D5 · Attention变体</a>
          <a href="{{ '/llm/week02/llm-w2-attention-code' | relative_url }}" class="day-link">D6 · 代码实现</a>
          <a href="{{ '/llm/week02/llm-w2-capstone' | relative_url }}" class="day-link">D7 · 综合实战</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 3 · 数学基础 II</div>
        <div class="day-grid">
          <a href="{{ '/llm/week03/llm-w3-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week03/llm-w3-derivatives' | relative_url }}" class="day-link">D1 · 导数与链式法则</a>
          <a href="{{ '/llm/week03/llm-w3-backprop-math' | relative_url }}" class="day-link">D2 · 反向传播数学</a>
          <a href="{{ '/llm/week03/llm-w3-probability' | relative_url }}" class="day-link">D3 · 概率基础</a>
          <a href="{{ '/llm/week03/llm-w3-information-theory' | relative_url }}" class="day-link">D4 · 信息论</a>
          <a href="{{ '/llm/week03/llm-w3-mle' | relative_url }}" class="day-link">D5 · 最大似然估计</a>
          <a href="{{ '/llm/week03/llm-w3-pytorch-autograd' | relative_url }}" class="day-link">D6 · PyTorch Autograd</a>
          <a href="{{ '/llm/week03/llm-w3-capstone' | relative_url }}" class="day-link">D7 · 综合实战</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 4 · 预训练（数据/Tokenizer/GPT/Scaling Law）</div>
        <div class="day-grid">
          <a href="{{ '/llm/week04/llm-w4-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week04/llm-w4-data-processing' | relative_url }}" class="day-link">D1 · 预训练数据处理</a>
          <a href="{{ '/llm/week04/llm-w4-tokenizer' | relative_url }}" class="day-link">D2 · Tokenizer 实战</a>
          <a href="{{ '/llm/week04/llm-w4-gpt-pretraining' | relative_url }}" class="day-link">D3 · GPT 预训练目标</a>
          <a href="{{ '/llm/week04/llm-w4-scaling-law' | relative_url }}" class="day-link">D4 · Scaling Law</a>
          <a href="{{ '/llm/week04/llm-w4-rope-position' | relative_url }}" class="day-link">D5 · RoPE 位置编码</a>
          <a href="{{ '/llm/week04/llm-w4-capstone' | relative_url }}" class="day-link">D6 · 手写 mini-GPT</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 5 · SFT 与对齐（LoRA/QLoRA/RLHF/DPO）</div>
        <div class="day-grid">
          <a href="{{ '/llm/week05/llm-w5-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week05/llm-w5-sft' | relative_url }}" class="day-link">D1 · 指令微调 SFT</a>
          <a href="{{ '/llm/week05/llm-w5-lora' | relative_url }}" class="day-link">D2 · LoRA & QLoRA</a>
          <a href="{{ '/llm/week05/llm-w5-rlhf' | relative_url }}" class="day-link">D3 · RLHF 全流程</a>
          <a href="{{ '/llm/week05/llm-w5-dpo' | relative_url }}" class="day-link">D4 · DPO 直接偏好优化</a>
          <a href="{{ '/llm/week05/llm-w5-alignment-practice' | relative_url }}" class="day-link">D5 · 对齐综合实战</a>
          <a href="{{ '/llm/week05/llm-w5-capstone' | relative_url }}" class="day-link">D6 · 端到端对齐实战</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 6 · 推理与部署（KV Cache/量化/vLLM/投机采样）</div>
        <div class="day-grid">
          <a href="{{ '/llm/week06/llm-w6-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week06/llm-w6-kv-cache' | relative_url }}" class="day-link">D1 · KV Cache</a>
          <a href="{{ '/llm/week06/llm-w6-quantization' | relative_url }}" class="day-link">D2 · 量化 INT4/GPTQ/AWQ</a>
          <a href="{{ '/llm/week06/llm-w6-vllm' | relative_url }}" class="day-link">D3 · vLLM PagedAttention</a>
          <a href="{{ '/llm/week06/llm-w6-speculative' | relative_url }}" class="day-link">D4 · 投机采样</a>
          <a href="{{ '/llm/week06/llm-w6-flash-attention' | relative_url }}" class="day-link">D5 · Flash Attention</a>
          <a href="{{ '/llm/week06/llm-w6-capstone' | relative_url }}" class="day-link">D6 · 生产级推理服务</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 7 · RAG 全链路（向量库/Embedding/Rerank/评估）</div>
        <div class="day-grid">
          <a href="{{ '/llm/week07/llm-w7-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week07/llm-w7-vector-db' | relative_url }}" class="day-link">D1 · 向量数据库 & ANN</a>
          <a href="{{ '/llm/week07/llm-w7-embedding' | relative_url }}" class="day-link">D2 · Embedding 模型</a>
          <a href="{{ '/llm/week07/llm-w7-retrieval' | relative_url }}" class="day-link">D3 · 混合检索策略</a>
          <a href="{{ '/llm/week07/llm-w7-reranker' | relative_url }}" class="day-link">D4 · Reranker 精排</a>
          <a href="{{ '/llm/week07/llm-w7-rag-eval' | relative_url }}" class="day-link">D5 · RAG 评估体系</a>
          <a href="{{ '/llm/week07/llm-w7-capstone' | relative_url }}" class="day-link">D6 · 企业级 RAG 系统</a>
        </div>
      </div>
      <div class="week-block">
        <div class="week-label">Week 8 · 应用工程（Prompt/Function Calling/安全）</div>
        <div class="day-grid">
          <a href="{{ '/llm/week08/llm-w8-week-plan' | relative_url }}" class="day-link plan">📋 周规划总览</a>
          <a href="{{ '/llm/week08/llm-w8-prompt-engineering' | relative_url }}" class="day-link">D1 · Prompt 工程</a>
          <a href="{{ '/llm/week08/llm-w8-function-calling' | relative_url }}" class="day-link">D2 · Function Calling & Agent</a>
          <a href="{{ '/llm/week08/llm-w8-structured-output' | relative_url }}" class="day-link">D3 · 结构化输出</a>
          <a href="{{ '/llm/week08/llm-w8-agent' | relative_url }}" class="day-link">D4 · LLM Agent 设计</a>
          <a href="{{ '/llm/week08/llm-w8-safety' | relative_url }}" class="day-link">D5 · LLM 安全防护</a>
          <a href="{{ '/llm/week08/llm-w8-capstone' | relative_url }}" class="day-link">D6 · 完整应用 Capstone</a>
        </div>
      </div>
    </div>
  </div>

  <div class="phase c-llm">
    <div class="phase-header">
      <div class="phase-left"><div class="phase-icon" style="background:rgba(188,140,255,0.1)">📚</div>
        <div><div class="phase-title">Phase 2 · 主流模型</div><div class="phase-meta">W9–W16 · BERT/GPT/LLaMA/Qwen/MoE/长上下文</div></div>
      </div>
      <div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body"><p class="coming">🔜 即将开启…</p></div>
  </div>

  <div class="phase c-llm">
    <div class="phase-header">
      <div class="phase-left"><div class="phase-icon" style="background:rgba(188,140,255,0.1)">🔧</div>
        <div><div class="phase-title">Phase 3 · 微调与对齐</div><div class="phase-meta">W17–W26 · SFT/LoRA/QLoRA/RLHF/DPO</div></div>
      </div>
      <div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body"><p class="coming">🔜 即将开启…</p></div>
  </div>

  <div class="phase c-llm">
    <div class="phase-header">
      <div class="phase-left"><div class="phase-icon" style="background:rgba(188,140,255,0.1)">⚡</div>
        <div><div class="phase-title">Phase 4 · 推理与部署</div><div class="phase-meta">W27–W34 · KV Cache/量化/vLLM/模型并行</div></div>
      </div>
      <div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body"><p class="coming">🔜 即将开启…</p></div>
  </div>

  <div class="phase c-llm">
    <div class="phase-header">
      <div class="phase-left"><div class="phase-icon" style="background:rgba(188,140,255,0.1)">🛠️</div>
        <div><div class="phase-title">Phase 5 · 应用工程</div><div class="phase-meta">W35–W44 · Prompt/RAG/Function Call/LLMOps</div></div>
      </div>
      <div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body"><p class="coming">🔜 即将开启…</p></div>
  </div>

  <div class="phase c-llm">
    <div class="phase-header">
      <div class="phase-left"><div class="phase-icon" style="background:rgba(188,140,255,0.1)">🔬</div>
        <div><div class="phase-title">Phase 6 · 前沿研究</div><div class="phase-meta">W45–W52 · o1推理/World Model/多模态大模型</div></div>
      </div>
      <div class="phase-right"><span class="phase-progress">即将开启</span><span class="phase-arrow">▶</span></div>
    </div>
    <div class="phase-body"><p class="coming">🔜 即将开启…</p></div>
  </div>

</div>
