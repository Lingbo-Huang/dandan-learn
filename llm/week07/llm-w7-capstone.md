---
layout: default
title: "D6 · Capstone：端到端 RAG 系统"
render_with_liquid: false
---

# D6 · Capstone：构建企业级 RAG 系统

> **目标**：构建一个完整的 RAG 系统，支持 PDF 解析、分块、向量索引、混合检索、Rerank、生成和评估。

---

## 一、系统架构

```
PDF/文档 → 解析 → 分块 → Embedding → 向量数据库
                                          ↓
用户提问 → Embedding → 混合检索 → Reranker → LLM → 答案
                                          ↑
                                   BM25 索引
```

---

## 二、文档处理流水线

```python
# document_processor.py
from pathlib import Path
import re

class DocumentProcessor:
    """文档处理器：PDF/Text → 分块"""
    
    def __init__(
        self,
        chunk_size: int = 512,       # 每块约 512 个字符
        chunk_overlap: int = 64,     # 重叠 64 个字符（防止截断信息）
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """提取 PDF 文本"""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text
        except ImportError:
            # 备选：PyPDF2
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
            return text
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 删除多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        # 删除页眉页脚（启发式）
        lines = text.split('\n')
        cleaned_lines = [l for l in lines if len(l.strip()) > 10]
        return '\n'.join(cleaned_lines)
    
    def split_by_sentence(self, text: str) -> list[str]:
        """按句子分割（中文友好）"""
        import re
        sentences = re.split(r'(?<=[。！？\n])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str) -> list[dict]:
        """
        创建文本块（滑动窗口）
        保留 metadata（来源、位置）
        """
        sentences = self.split_by_sentence(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            if current_length + len(sent) > self.chunk_size and current_chunk:
                # 保存当前块
                chunk_text = ''.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'char_start': sum(len(c) for c in chunks),
                })
                
                # 重叠：保留最后几个句子
                overlap_text = ""
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) <= self.chunk_overlap:
                        overlap_text = s + overlap_text
                    else:
                        break
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(sent)
            current_length += len(sent)
        
        # 最后一块
        if current_chunk:
            chunks.append({
                'text': ''.join(current_chunk),
                'chunk_id': len(chunks),
                'char_start': 0,
            })
        
        return chunks
    
    def process_file(self, file_path: str) -> list[dict]:
        """处理单个文件"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            text = self.load_pdf(file_path)
        else:
            with open(file_path, encoding='utf-8') as f:
                text = f.read()
        
        text = self.clean_text(text)
        chunks = self.create_chunks(text)
        
        for chunk in chunks:
            chunk['source'] = path.name
        
        print(f"处理完成: {path.name} → {len(chunks)} 个文档块")
        return chunks
```

---

## 三、完整 RAG 系统

```python
# rag_system.py
import asyncio
from pathlib import Path

class EnterpriseRAG:
    """企业级 RAG 系统"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-large",
        llm_url: str = "http://localhost:8000",
        data_dir: str = "./data",
        index_dir: str = "./index",
    ):
        from hybrid_retriever import HybridRetriever
        from cross_encoder import CrossEncoderReranker
        from llm_client import LLMClient
        
        self.processor = DocumentProcessor()
        self.retriever = HybridRetriever(embedding_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.llm = LLMClient(llm_url)
        
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
    
    def build_index(self, doc_paths: list[str] = None) -> None:
        """构建知识库索引"""
        if doc_paths is None:
            doc_paths = list(self.data_dir.glob("*.pdf")) + \
                       list(self.data_dir.glob("*.txt")) + \
                       list(self.data_dir.glob("*.md"))
        
        all_chunks = []
        for path in doc_paths:
            chunks = self.processor.process_file(str(path))
            all_chunks.extend(chunks)
        
        # 构建检索索引
        documents = [c['text'] for c in all_chunks]
        self.retriever.build_index(documents)
        
        # 保存元数据
        import json
        with open(self.index_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"索引构建完成，共 {len(all_chunks)} 个文档块")
    
    async def answer(
        self,
        query: str,
        recall_k: int = 20,
        rerank_k: int = 5,
        return_sources: bool = True,
    ) -> dict:
        """回答问题"""
        # 1. 检索
        recalled = self.retriever.search(query, k=recall_k)
        docs = [r['doc'] for r in recalled]
        
        # 2. 重排
        reranked = self.reranker.rerank(query, docs, top_k=rerank_k)
        
        # 3. 构建上下文
        context_parts = []
        for i, r in enumerate(reranked, 1):
            context_parts.append(f"[参考文档{i}]\n{r['doc']}")
        context = "\n\n".join(context_parts)
        
        # 4. 生成答案
        system_prompt = """你是一个专业的知识助手。请严格根据提供的参考文档回答用户问题。
要求：
1. 只使用参考文档中的信息
2. 如果文档中没有相关信息，明确说明"根据提供的文档，无法回答此问题"
3. 回答要准确、简洁、有条理
4. 必要时引用文档内容"""
        
        user_prompt = f"""参考文档：
{context}

用户问题：{query}

请根据以上文档回答问题："""
        
        answer = await self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.3,  # 事实性问答用低温度
        )
        
        result = {
            'query': query,
            'answer': answer,
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'text': r['doc'][:200],
                    'relevance_score': r['score'],
                    'rank': i + 1,
                }
                for i, r in enumerate(reranked)
            ]
        
        return result
    
    async def interactive_chat(self) -> None:
        """交互式问答"""
        print("RAG 知识库助手已就绪！（输入 'quit' 退出）\n")
        
        while True:
            query = input("问题: ").strip()
            if query.lower() == 'quit':
                break
            if not query:
                continue
            
            print("检索中...")
            result = await self.answer(query)
            
            print(f"\n答案：{result['answer']}")
            
            if 'sources' in result:
                print(f"\n参考来源：")
                for src in result['sources'][:3]:
                    print(f"  [{src['rank']}] (相关度:{src['relevance_score']:.3f}) {src['text'][:60]}...")
            print()


# 启动脚本
async def main():
    rag = EnterpriseRAG(
        embedding_model="BAAI/bge-large-zh-v1.5",
        reranker_model="BAAI/bge-reranker-large",
        llm_url="http://localhost:8000",
    )
    
    # 构建索引（首次运行）
    rag.build_index(["./docs/llm_tech.pdf", "./docs/rag_tutorial.md"])
    
    # 单次查询
    result = await rag.answer("什么是 Transformer 的注意力机制？")
    print(result['answer'])
    
    # 交互式对话
    # await rag.interactive_chat()

asyncio.run(main())
```

---

## 四、Week 7 总结

```
RAG 系统技术栈：

文档处理：
  PDF → pdfplumber → 分块（512 chars，64 重叠）

索引：
  稠密：FAISS + BGE-M3 Embedding
  稀疏：BM25（rank-bm25 库）
  持久化：Chroma（原型）/ Milvus（生产）

检索：
  一阶段：混合检索（dense 0.7 + sparse 0.3）
  二阶段：Cross-Encoder Reranker
  高级：HyDE、多路召回

生成：
  Prompt 工程（严格基于上下文）
  低温度（事实性任务）

评估：
  检索：Recall@k, NDCG
  生成：RAGAS（Faithfulness + Answer Relevancy）
```

### 面试综合题

**Q: 如何设计一个支持百万文档的 RAG 系统？**

A:
1. **文档处理**：并行处理，语义感知分块（按段落/标题而非固定长度）
2. **向量存储**：Milvus 分布式集群，IVF+PQ 压缩索引
3. **检索优化**：
   - 混合检索（BM25 + 稠密）
   - 元数据过滤（时间、类别、权限）
   - Prefix Caching（vLLM 缓存常见 context）
4. **工程优化**：
   - Embedding 并发批处理
   - Reranker 轻量化（BGE-reranker-v2-minicpm）
   - 异步 LLM 请求
   - 结果缓存（Redis，相同 query 命中）
