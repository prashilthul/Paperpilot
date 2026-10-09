# Paper Pilot

Paper Pilot is a research paper Q&A platform built for grounded document reasoning. It parses PDFs into structural section hierarchies, runs hybrid vector + lexical search fused with cross-encoder reranking, and streams answers with interactive inline citations.

Built with operational transparency in mind, every query features **end-to-end request traceability**—recording granular execution spans across embedding, retrieval, reranking, and generation stages—paired with real-time quality metrics and waterfall trace inspection.

---

## Interface Screenshots

### Grounded Research Chat
![Grounded Research Chat](photos/chat.png)

### Metrics Dashboard
![Operational Metrics Dashboard](photos/dashboard.png)

### Trace Inspector
![Trace Waterfall Inspector](photos/trace.png)

---

## Architecture

```mermaid
graph TD
    User([User / Browser]) -->|HTTP / SSE| FE[Next.js Frontend]
    FE -->|REST API & SSE Streams| BE[FastAPI Backend]

    subgraph Storage & Telemetry Layer
        BE -->|Async SQLAlchemy / pgvector| DB[(PostgreSQL + pgvector)]
    end

    subgraph AI Model Gateway
        BE -->|Embeddings / Reranking / LLM| API[OpenRouter API]
    end
```

### RAG Pipeline Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Next.js UI
    participant BE as FastAPI Backend
    participant DB as Postgres + pgvector
    participant API as OpenRouter API

    User->>FE: Submit question in chat box
    FE->>BE: POST /api/chat (stream=true)
    BE->>API: 1. Embed query text (1024d vector)
    API-->>BE: Return query vector
    BE->>DB: 2. Hybrid vector (<=>) + lexical (tsvector) search
    DB-->>BE: Return top 20 candidate chunks
    BE->>API: 3. Cross-encoder rerank candidate chunks
    API-->>BE: Return top 5 reranked chunks
    BE->>API: 4. Stream grounded generation with context
    API-->>BE: Stream token chunks
    BE-->>FE: Server-Sent Events (event: token, event: done)
    FE-->>User: Render live response with inline citations [1]
    BE->>DB: 5. Store message history & execution trace spans
```

---

## Stack Choices & Rationale

* **FastAPI (Backend)**  
  Async-native Python framework with built-in Pydantic schema validation. Handles concurrent streaming responses and NLP pipeline tasks cleanly without blocking.

* **Next.js & TailwindCSS (Frontend)**  
  Provides fast client-side rendering, SSE stream consumption, and customizable component styling without bundle bloat.

* **PostgreSQL + `pgvector` (Database & Vector Store)**  
  Keeps relational metadata, trace logs, and vector embeddings in a single database. Avoids the operational overhead of running and syncing a separate vector DB cluster (e.g. Pinecone/Qdrant).

* **PyMuPDF (`fitz`) (PDF Parsing)**  
  Extremely fast PDF text extraction that preserves font sizes and layout metadata. Enables accurate section heading identification based on font weight and size hierarchy.

* **Hybrid Search + Reciprocal Rank Fusion (RRF) + Reranking**  
  Combines dense vector similarity search with Postgres full-text lexical search (`to_tsvector`), merges ranks using RRF, and filters results through a cross-encoder reranker before feeding context to the LLM. This prevents missing keyword hits and reduces hallucination.

* **Server-Sent Events (SSE)**  
  Used for streaming tokens from LLM to frontend over standard HTTP. Simpler and lighter than WebSockets for single-directional response generation.

---

## Advanced RAG Engineering & Evaluation Strategy

Paper Pilot incorporates industry-standard production RAG patterns designed to eliminate common failure modes (such as retrieval vocabulary gaps, hallucinated citations, and context dilution).

### 1. Multi-Stage Retrieval Architecture
* **Structure-Aware Section Chunking**: PDF layout font metadata is extracted via PyMuPDF to construct section hierarchies. Content is chunked along section boundaries with parent-child metadata rather than naive token splitting.
* **Hybrid Search + Reciprocal Rank Fusion (RRF)**: Executes dense vector similarity search (`pgvector` cosine ops) alongside sparse lexical full-text search (`to_tsvector`), fusing ranks via RRF ($k=60$) to capture both semantic meaning and specific keyword terms.
* **Cross-Encoder Reranking**: Re-scores top 20 candidate chunks down to top 5 using a cross-encoder model before context construction, significantly boosting context precision.
* **Multi-Turn Query Rewriting**: Evaluates user conversation history to rewrite ambiguous follow-up questions before executing retrieval.

### 2. Evaluation System (Online & Batch)
The platform includes dual evaluation pipelines to track grounding and retrieval performance:

* **Online LLM-as-a-Judge (`online_eval.py`)**: Evaluates responses live in production asynchronously (5% background sampling, 100% coverage on errors):
  * **Faithfulness**: Measures if generated claims are 100% supported by the retrieved context.
  * **Citation Accuracy**: Verifies that each inline citation badge `[N]` points to a chunk containing the true factual evidence.
* **Batch RAGAS Evaluation (`scripts/ragas_eval.py`)**: Automated test suite measuring standard RAG metrics:
  * **Faithfulness**: Claim verification ratio against retrieved context.
  * **Answer Relevancy**: Direct semantic alignment between query intent and response.
  * **Context Precision**: Signal-to-noise ratio in top-ranked chunks.
  * **Context Recall**: Ground-truth coverage across retrieved chunks.

#### Retrieval Ablation (measured)

Ranking quality of the retrieval stages, scored against a hand-labelled gold set of 20 questions over 4 arXiv papers (Attention, ResNet, BERT, and arXiv 2010.05425, "FPRAS via MCMC"). Gold chunks are matched by anchor phrases in chunk content, not by section headings. Scoring is pure ranking arithmetic: no generation and no LLM judge.

| Config | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Dense only (pgvector) | 0.391 | 0.603 | 0.741 | 0.618 |
| Hybrid (dense + FTS, RRF k=60) | 0.474 | 0.660 | 0.721 | 0.637 |
| Hybrid + cross-encoder rerank | **0.522** | **0.695** | **0.762** | **0.684** |

- Hybrid over dense: +8.3 points Recall@5.
- Rerank over hybrid: +4.8 points Recall@5 and +4.7 points nDCG@10.
- Rerank over dense: +13.1 points Recall@5, a 33% relative gain.

Caveats:
- Small corpus (about 360 chunks across 4 papers) and 20 questions. Treat the deltas as directional, not as a benchmark score.
- These numbers predate the PDF parser fixes (heading detection and NUL-byte handling), so they reflect the earlier chunking. A re-run on the current corpus is pending.
- Reranker and embeddings use OpenRouter free-tier models. Daily quota exhaustion silently falls back to hash embeddings, so results are only valid from runs made under quota.

Reproduce (inside the backend container, after ingesting the papers in `backend/evals/papers/`):

```bash
docker compose exec backend sh -c "PYTHONPATH=. uv run python scripts/eval_retrieval.py --configs vector_only,hybrid,hybrid_rerank --top-k 20"
```

### 3. Telemetry & Execution Tracing
Every request records OpenTelemetry-style execution spans (`embed`, `query_rewrite`, `vector_search`, `lexical_search`, `rerank`, `generate`, `judge`). Spans capture timing, token counts, and retrieval scores, feeding directly into the **Trace Inspector** waterfall view and operational **Dashboard KPIs**.

---

## Key Features

* **Structure-Aware Chunking**: Chunks papers by section headings rather than arbitrary token boundaries, preserving topical context.
* **Interactive Citations**: Answers include clickable `[1]`, `[2]` citation badges mapping directly to section headers and chunk IDs.
* **Paper Inspector**: Collapsible side drawer to review abstract, metadata, and full document section breakdown while chatting.
* **Operational Telemetry & Traces**: Built-in trace waterfall viewer to inspect latency, embedding dimensions, and search scores for every query.
* **Chat Session Management**: History drawer with session persistence and idempotent session deletion.

---

## Quick Start

### Prerequisites
* Docker & Docker Compose
* OpenRouter API Key

### 1. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your `OPENROUTER_API_KEY` in `.env`.

### 2. Launch with Docker Compose
```bash
docker compose up --build
```

### 3. Open Application
Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

---

## Core API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/papers/upload` | Upload and index a PDF paper |
| `GET` | `/api/papers` | List indexed papers with metadata |
| `POST` | `/api/chat` | Query RAG pipeline (supports `stream=true`) |
| `GET` | `/api/chat/sessions` | List active chat history sessions |
| `DELETE` | `/api/chat/sessions/{id}` | Delete a chat session |
| `GET` | `/api/metrics/summary` | Fetch operational health & latency telemetry |
