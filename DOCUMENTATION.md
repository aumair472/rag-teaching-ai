# RAG Teaching Assistant — Technical Documentation

**Author:** Umair Ali  
**Date:** March 2026  
**Version:** 1.0.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Component Deep Dive](#6-component-deep-dive)
7. [Data Pipeline](#7-data-pipeline)
8. [API Specification](#8-api-specification)
9. [Voice Agent](#9-voice-agent)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Deployment](#11-deployment)
12. [Performance Considerations](#12-performance-considerations)
13. [Future Enhancements](#13-future-enhancements)
14. [References](#14-references)

---

## 1. Executive Summary

The **RAG Teaching Assistant** is a production-grade Retrieval-Augmented Generation (RAG) microservice designed to function as an AI-powered teaching assistant. The system ingests educational content from multiple formats (lecture videos, PDFs, PowerPoint presentations), processes them into a searchable knowledge base, and provides accurate, citation-backed answers to student questions.

### Key Capabilities

- **Multi-format Ingestion**: Supports PDF documents, PowerPoint slides, and lecture videos
- **Hybrid Retrieval**: Combines bi-encoder vector search with cross-encoder re-ranking
- **Guardrails**: Prevents hallucination by refusing to answer questions outside the knowledge base
- **Voice Agent**: Complete speech-to-text and text-to-speech pipeline for hands-free interaction
- **Streaming Responses**: Real-time token-by-token response delivery via Server-Sent Events (SSE)
- **Production-Ready**: Redis caching, rate limiting, health checks, and Docker deployment

---

## 2. Problem Statement

### Challenges in Educational Settings

1. **Scalability**: Instructors cannot provide 1:1 support to hundreds of students
2. **Availability**: Students need answers outside office hours
3. **Consistency**: Answer quality varies depending on who responds
4. **Accessibility**: Traditional text-based interfaces exclude some learners

### Why Existing Solutions Fall Short

| Solution | Limitation |
|----------|------------|
| Generic ChatGPT | Hallucinates; not grounded in specific course material |
| Search Engines | Return documents, not answers; require manual reading |
| FAQ Systems | Static; don't understand natural language |
| Human TAs | Limited availability; expensive to scale |

---

## 3. Solution Overview

The RAG Teaching Assistant addresses these challenges through a sophisticated pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INGESTION     │───▶│   RETRIEVAL     │───▶│   GENERATION    │
│                 │    │                 │    │                 │
│ • PDF Extract   │    │ • Bi-encoder    │    │ • GPT-4o        │
│ • PPT Extract   │    │ • FAISS Search  │    │ • Guardrails    │
│ • Video/Whisper │    │ • Cross-encoder │    │ • Citations     │
│ • Chunking      │    │ • Threshold     │    │ • Streaming     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Design Principles

1. **Grounded Responses**: Every answer must be traceable to source material
2. **Transparency**: Provide source citations with page/slide/timestamp
3. **Fail-Safe**: Admit when information is not in the knowledge base
4. **Low Latency**: Cache frequent queries; stream responses
5. **Modular Architecture**: Each component is swappable and testable

---

## 4. System Architecture

### High-Level Architecture

```
                                    ┌──────────────────────────────────────┐
                                    │           VOICE AGENT                │
                                    │  ┌─────────┐      ┌─────────┐       │
                                    │  │ Whisper │      │ OpenAI  │       │
                                    │  │   STT   │      │   TTS   │       │
                                    │  └────┬────┘      └────▲────┘       │
                                    │       │               │             │
                                    │       ▼               │             │
┌─────────────┐    ┌────────────────┼───────────────────────┼─────────────┤
│   Client    │    │                │    FastAPI Backend    │             │
│             │    │                │                       │             │
│ • Streamlit │───▶│  ┌──────────┐  │  ┌─────────────────┐  │             │
│ • cURL      │    │  │  /ask    │─────▶│  RAG Service   │  │             │
│ • Browser   │    │  │  /ingest │  │  │                 │  │             │
│             │◀───│  │  /voice  │◀────│  • Cache Check  │  │             │
│             │    │  │  /health │  │  │  • Retrieval    │  │             │
│             │    │  └──────────┘  │  │  • Generation   │  │             │
└─────────────┘    │                │  │  • Memory       │  │             │
                   │                │  └────────┬────────┘  │             │
                   │                │           │           │             │
                   └────────────────┼───────────┼───────────┼─────────────┘
                                    │           │           │
                   ┌────────────────┼───────────┼───────────┼─────────────┐
                   │                │           ▼           │             │
                   │  ┌─────────┐   │   ┌─────────────┐     │  ┌───────┐  │
                   │  │  FAISS  │◀──────│  Retrieval  │     │  │ Redis │  │
                   │  │  Index  │   │   │   Service   │     │  │ Cache │  │
                   │  └─────────┘   │   └─────────────┘     │  └───────┘  │
                   │                │           │           │             │
                   │                │           ▼           │             │
                   │                │   ┌─────────────┐     │             │
                   │                │   │Cross-Encoder│     │             │
                   │                │   │  Re-ranker  │     │             │
                   │                │   └─────────────┘     │             │
                   │                │      DATA LAYER       │             │
                   └────────────────┴───────────────────────┴─────────────┘
```

### Request Flow

```
1. User submits question
       │
       ▼
2. Redis cache lookup ────────────────────────────────┐
       │                                              │
       │ MISS                                    HIT  │
       ▼                                              │
3. Embed query (all-MiniLM-L6-v2)                     │
       │                                              │
       ▼                                              │
4. FAISS vector search (Top-K=10)                     │
       │                                              │
       ▼                                              │
5. Similarity threshold filter (≥0.3)                 │
       │                                              │
       ▼                                              │
6. Cross-encoder re-ranking (Top-N=5)                 │
       │                                              │
       ▼                                              │
7. Guardrail check ──────────── No results? ──────▶ "Not covered" message
       │
       │ Has context
       ▼
8. Construct prompt with history + context
       │
       ▼
9. GPT-4o generation (stream or sync)
       │
       ▼
10. Cache response + Update memory
       │
       ▼
11. Return answer with source citations ◀─────────────┘
```

---

## 5. Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Backend Framework** | FastAPI | 0.109+ | Async REST API with OpenAPI docs |
| **ASGI Server** | Uvicorn | 0.27+ | High-performance async server |
| **LLM** | OpenAI GPT-4o | Latest | Answer generation |
| **Embeddings** | sentence-transformers | 2.3+ | all-MiniLM-L6-v2 (384-dim) |
| **Re-ranker** | CrossEncoder | 2.3+ | ms-marco-MiniLM-L-6-v2 |
| **Vector Store** | FAISS | 1.7+ | Similarity search |
| **Cache** | Redis | 7+ | Response caching |
| **STT** | OpenAI Whisper | whisper-1 | Speech transcription |
| **TTS** | OpenAI TTS | tts-1-hd | Speech synthesis |
| **Containerization** | Docker | 24+ | Deployment |

### Python Dependencies

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
openai>=1.10.0
sentence-transformers>=2.3.0
faiss-cpu>=1.7.4
redis>=5.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6
slowapi>=0.1.9
httpx>=0.26.0
PyMuPDF>=1.23.0
python-pptx>=0.6.21
pydub>=0.25.1
ragas>=0.1.0
structlog>=24.1.0
```

---

## 6. Component Deep Dive

### 6.1 Embedding Service

**File:** `app/services/embedding_service.py`

The embedding service converts text into dense vector representations using sentence-transformers.

**Model:** `all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Max Sequence Length:** 256 tokens
- **Training:** Distilled from MiniLM, trained on 1B+ sentence pairs
- **Performance:** Fast inference with good semantic understanding

**Key Features:**
- Batch encoding for efficiency
- L2 normalization for cosine similarity
- Thread-safe singleton pattern

```python
class EmbeddingService:
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Returns normalized embeddings of shape (n, 384)"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
```

### 6.2 Vector Store (FAISS)

**File:** `app/db/vector_store.py`

FAISS provides fast approximate nearest-neighbor search.

**Index Type:** `IndexFlatIP` (Inner Product)
- Used with L2-normalized vectors → equivalent to cosine similarity
- Exact search (not approximate) for smaller datasets
- Can swap to `IndexIVFFlat` for larger scale

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│                 FAISSVectorStore                │
├─────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────────┐    │
│  │  FAISS Index  │    │  Metadata (JSON)  │    │
│  │  (vectors)    │    │  • source_name    │    │
│  │               │◀──▶│  • source_type    │    │
│  │  index.faiss  │    │  • page/slide     │    │
│  │               │    │  • text           │    │
│  └───────────────┘    └───────────────────┘    │
│           │                     │              │
│           └─────────┬───────────┘              │
│                     ▼                          │
│            Disk Persistence                    │
│         (data/faiss_index.bin)                 │
└─────────────────────────────────────────────────┘
```

### 6.3 Retrieval Service

**File:** `app/services/retrieval_service.py`

Implements a two-stage hybrid retrieval pipeline:

**Stage 1: Bi-Encoder (Fast)**
```
Query → Embed → FAISS Top-K Search → Candidates
```

**Stage 2: Cross-Encoder (Accurate)**
```
Candidates → Cross-Encoder Re-rank → Top-N Results
```

**Why Two Stages?**

| Stage | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| Bi-encoder | ~1ms | Good | Filter millions to hundreds |
| Cross-encoder | ~50ms | Excellent | Re-rank hundreds to top results |

**Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Trained on MS MARCO passage ranking
- Takes (query, passage) pairs and outputs relevance score
- More accurate than bi-encoder but slower

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 10 | Candidates from vector search |
| `TOP_N_RERANK` | 5 | Results after re-ranking |
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum cosine similarity |

### 6.4 LLM Service

**File:** `app/services/llm_service.py`

Handles all GPT-4o interactions with a carefully crafted system prompt.

**System Prompt Design:**

```
You are an AI Teaching Assistant. Your role is to help students 
understand course material by providing clear, accurate answers.

STRICT RULES:
1. Answer ONLY from the provided context below. Do NOT use external knowledge.
2. If the context does not contain sufficient information, respond with:
   "This topic is not covered in the course material."
3. Always cite your sources using [Source: <name>, <location>]
4. Be concise but thorough.
5. Never fabricate, hallucinate, or speculate.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}
```

**Streaming Implementation:**

```python
async def generate_stream(self, question: str, context: str) -> AsyncGenerator[str, None]:
    stream = await self.client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 6.5 RAG Orchestrator

**File:** `app/services/rag_service.py`

The central coordinator that ties all services together:

```python
class RAGService:
    async def ask(self, question: str, session_id: str) -> AskResponse:
        # Step 1: Cache lookup
        if cached := self.cache.get(question):
            return cached

        # Step 2: Retrieve relevant chunks
        chunks = self.retrieval_service.retrieve(question)

        # Step 3: Guardrail check
        if not chunks:
            return AskResponse(answer=GUARDRAIL_MESSAGE)

        # Step 4: Get conversation history
        history = self.memory_service.get_history(session_id)

        # Step 5: Generate answer
        context = self._format_context(chunks)
        answer = await self.llm_service.generate(question, context, history)

        # Step 6: Extract and attach sources
        sources = self._extract_sources(chunks)

        # Step 7: Cache and update memory
        self.cache.set(question, {"answer": answer, "sources": sources})
        self.memory_service.add(session_id, question, answer)

        return AskResponse(answer=answer, sources=sources)
```

### 6.6 Memory Service

**File:** `app/services/memory_service.py`

Maintains conversation history per session for contextual follow-ups.

**Storage:** In-memory dictionary (can be extended to Redis)

**Format:**
```python
{
    "session_123": [
        {"role": "user", "content": "What is gradient descent?"},
        {"role": "assistant", "content": "Gradient descent is..."},
    ]
}
```

**Window Size:** Last 10 messages per session

---

## 7. Data Pipeline

### 7.1 Ingestion Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    PDF                    PPT                    VIDEO               │
│     │                      │                       │                 │
│     ▼                      ▼                       ▼                 │
│  ┌──────┐              ┌──────┐              ┌──────────┐           │
│  │PyMuPDF│             │pptx  │              │  ffmpeg  │           │
│  └───┬───┘             └───┬──┘              └────┬─────┘           │
│      │                     │                      │                 │
│      ▼                     ▼                      ▼                 │
│  Page-aware            Slide-aware           Audio extract          │
│  extraction            extraction                 │                 │
│      │                     │                      ▼                 │
│      │                     │               ┌───────────┐            │
│      │                     │               │  Whisper  │            │
│      │                     │               │    API    │            │
│      │                     │               └─────┬─────┘            │
│      │                     │                     │                  │
│      │                     │                     ▼                  │
│      │                     │              Timestamped segments      │
│      │                     │                     │                  │
│      └──────────┬──────────┴─────────────────────┘                  │
│                 │                                                   │
│                 ▼                                                   │
│         ┌─────────────────┐                                         │
│         │  MetadataAware  │                                         │
│         │    Chunker      │                                         │
│         └────────┬────────┘                                         │
│                  │                                                  │
│                  ▼                                                  │
│      ┌───────────────────────┐                                      │
│      │   DocumentChunk       │                                      │
│      │   • text              │                                      │
│      │   • source_name       │                                      │
│      │   • source_type       │                                      │
│      │   • page / slide      │                                      │
│      │   • timestamp         │                                      │
│      └───────────┬───────────┘                                      │
│                  │                                                  │
│                  ▼                                                  │
│         ┌─────────────────┐                                         │
│         │   Embed chunks  │                                         │
│         │ (batch encode)  │                                         │
│         └────────┬────────┘                                         │
│                  │                                                  │
│                  ▼                                                  │
│         ┌─────────────────┐                                         │
│         │  FAISS Index    │                                         │
│         │  + Metadata     │                                         │
│         └─────────────────┘                                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Chunking Strategy

**File:** `ingestion/chunk.py`

The `MetadataAwareChunker` splits text while preserving source information:

**Strategy:**
1. Split on paragraph boundaries (`\n\n`)
2. If paragraph > chunk_size, split on sentence boundaries (`.!?`)
3. As last resort, split at character boundaries
4. Apply overlap for context continuity

**Parameters:**

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `CHUNK_SIZE` | 512 | Fits embedding model context (256 tokens ≈ 512 chars) |
| `CHUNK_OVERLAP` | 50 | Prevents information loss at boundaries |

**Output Schema:**

```python
class DocumentChunk(BaseModel):
    text: str                    # Chunk content
    source_name: str             # "Lecture 1 - Introduction"
    source_type: SourceType      # PDF | PPT | VIDEO
    page: Optional[int]          # Page number (PDF)
    slide: Optional[int]         # Slide number (PPT)
    timestamp: Optional[str]     # "00:05:30" (Video)
```

### 7.3 Extractors

#### PDF Extractor
**File:** `ingestion/extract_pdf.py`

Uses PyMuPDF (fitz) for text extraction:
- Preserves page numbers
- Handles multi-column layouts
- Extracts text from images via OCR (optional)

#### PPT Extractor
**File:** `ingestion/extract_ppt.py`

Uses python-pptx:
- Extracts text from shapes and text boxes
- Preserves slide numbers
- Handles speaker notes

#### Video Transcriber
**File:** `ingestion/transcribe.py`

Pipeline:
1. **ffmpeg**: Extract audio track as MP3
2. **Whisper API**: Transcribe with timestamps
3. **Chunker**: Create timestamped segments

---

## 8. API Specification

### 8.1 Endpoints

#### POST /ask

Ask a question to the teaching assistant.

**Request:**
```json
{
    "question": "What is gradient descent?",
    "session_id": "student-001",
    "stream": false
}
```

**Response (JSON):**
```json
{
    "answer": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent...",
    "sources": [
        {
            "source_name": "ML Lecture 1",
            "source_type": "PDF",
            "page": 15,
            "text_snippet": "Gradient descent is used to minimize...",
            "similarity_score": 0.89,
            "rerank_score": 0.95
        }
    ],
    "cached": false,
    "session_id": "student-001",
    "latency_ms": 1234.56
}
```

**Response (Streaming SSE):**
```
data: Gradient
data:  descent
data:  is
data:  an
data:  optimization
...
data: [DONE]
```

#### POST /ingest

Ingest a document into the knowledge base.

**Request:**
```json
{
    "file_path": "/data/lectures/ml_intro.pdf",
    "source_type": "pdf",
    "source_name": "ML Lecture 1 - Introduction"
}
```

**Response:**
```json
{
    "message": "Document ingested successfully",
    "chunks_created": 45,
    "source_name": "ML Lecture 1 - Introduction"
}
```

#### POST /voice

Submit audio question, receive audio answer.

**Request:** `multipart/form-data`
- `audio`: WAV/MP3/WebM file
- `session_id`: Session identifier

**Response:** `audio/mpeg` (MP3 file)

**Headers:**
```
X-Voice-Metadata: {"transcription": "...", "answer": "...", "sources": [...]}
```

#### GET /health

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "components": {
        "vector_store": "healthy (1,234 vectors)",
        "redis": "healthy",
        "llm": "healthy"
    }
}
```

### 8.2 Rate Limiting

| Endpoint | Limit | Rationale |
|----------|-------|-----------|
| `/ask` | 20/minute | Prevent abuse; LLM costs |
| `/ingest` | 5/minute | Resource-intensive |
| `/voice` | 10/minute | API costs (STT + TTS) |
| `/health` | Unlimited | Monitoring needs |

---

## 9. Voice Agent

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VOICE AGENT                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌──────┐ │
│   │  Audio  │────▶│ Whisper │────▶│   RAG   │────▶│ TTS  │ │
│   │  Input  │     │   STT   │     │Pipeline │     │      │ │
│   └─────────┘     └─────────┘     └─────────┘     └──┬───┘ │
│                        │               │              │     │
│                        │               │              │     │
│                        ▼               ▼              ▼     │
│                   Transcribed      Answer         MP3 Audio │
│                      Text          + Sources       Output   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 STT Service

**File:** `app/services/stt_service.py`

**Model:** OpenAI Whisper (`whisper-1`)

**Supported Formats:** WAV, MP3, M4A, WebM, OGG, FLAC

**Limits:** 25 MB max file size

```python
async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
    response = await self.client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes),
        response_format="text",
    )
    return response
```

### 9.3 TTS Service

**File:** `app/services/tts_service.py`

**Model:** `tts-1-hd` (high-definition)

**Voices Available:**

| Voice | Description |
|-------|-------------|
| `alloy` | Neutral, balanced |
| `echo` | Warm, conversational |
| `fable` | Expressive, narrative |
| `onyx` | Deep, authoritative |
| `nova` | Friendly, energetic |
| `shimmer` | Clear, professional |

**Limits:** 4,096 characters per request

```python
async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
    response = await self.client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
        response_format="mp3",
    )
    return response.content
```

---

## 10. Evaluation Framework

### 10.1 RAGAS Metrics

**File:** `evaluation/evaluator.py`

The system uses RAGAS (Retrieval Augmented Generation Assessment) for quality evaluation:

| Metric | Description | Range |
|--------|-------------|-------|
| **Faithfulness** | Is the answer faithful to the context? (No hallucination) | 0-1 |
| **Answer Relevancy** | Is the answer relevant to the question? | 0-1 |
| **Context Precision** | Is the most relevant context ranked highest? | 0-1 |

### 10.2 Evaluation Pipeline

```python
from evaluation.evaluator import RAGEvaluator
from app.models.schemas import EvalSample

evaluator = RAGEvaluator()

samples = [
    EvalSample(
        question="What is gradient descent?",
        answer="Gradient descent is an optimization algorithm...",
        contexts=["Gradient descent is used to minimize..."],
        ground_truth="Gradient descent is an iterative optimization algorithm."
    )
]

result = evaluator.evaluate(samples)
# result.faithfulness = 0.95
# result.answer_relevancy = 0.92
# result.context_precision = 0.88
```

### 10.3 Results Logging

Evaluation results are persisted to `logs/eval_metrics.json`:

```json
{
    "timestamp": "2026-03-08T12:00:00Z",
    "num_samples": 50,
    "faithfulness": 0.94,
    "answer_relevancy": 0.91,
    "context_precision": 0.87
}
```

---

## 11. Deployment

### 11.1 Docker Deployment

**docker-compose.yml:**

```yaml
version: '3.8'
services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./data:/app/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

**Start Command:**
```bash
docker compose up --build
```

### 11.2 Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit UI (separate terminal)
cd streamlit-ui
streamlit run app.py
```

### 11.3 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | LLM model |
| `OPENAI_TEMPERATURE` | No | `0.2` | Sampling temperature |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Embedding model |
| `CROSS_ENCODER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranker |
| `TOP_K` | No | `10` | Vector search candidates |
| `TOP_N_RERANK` | No | `5` | Results after re-ranking |
| `SIMILARITY_THRESHOLD` | No | `0.3` | Minimum similarity |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection |
| `CACHE_TTL_SECONDS` | No | `3600` | Cache TTL |
| `RATE_LIMIT` | No | `20/minute` | API rate limit |
| `WHISPER_MODEL` | No | `whisper-1` | STT model |
| `TTS_MODEL` | No | `tts-1-hd` | TTS model |
| `TTS_VOICE` | No | `alloy` | TTS voice |

---

## 12. Performance Considerations

### 12.1 Latency Breakdown

| Stage | Typical Latency | Optimization |
|-------|-----------------|--------------|
| Cache lookup | ~1ms | Redis in-memory |
| Query embedding | ~5ms | Sentence-transformers (CPU) |
| FAISS search | ~2ms | IndexFlatIP (exact) |
| Cross-encoder | ~50ms | Batch scoring |
| LLM generation | ~1-3s | Streaming reduces perceived latency |
| **Total (cache miss)** | ~1.5-3.5s | |
| **Total (cache hit)** | ~5ms | |

### 12.2 Scaling Strategies

| Bottleneck | Solution |
|------------|----------|
| Vector search | IndexIVFFlat or IndexHNSW for 1M+ vectors |
| Embedding | GPU-accelerated encoding |
| LLM API | Response caching; request batching |
| Memory | Redis for session storage |
| Throughput | Horizontal scaling with load balancer |

### 12.3 Cost Optimization

| Component | Cost Driver | Mitigation |
|-----------|-------------|------------|
| GPT-4o | Token usage | Caching; concise prompts |
| Whisper | Audio minutes | Client-side compression |
| TTS | Characters | Summarize long answers |
| Embeddings | Self-hosted | No API cost (local model) |

---

## 13. Future Enhancements

### 13.1 Planned Features

| Feature | Priority | Description |
|---------|----------|-------------|
| Multi-language support | High | Support for non-English content |
| Image/diagram understanding | Medium | Process figures in PDFs |
| User authentication | High | Per-user session management |
| Analytics dashboard | Medium | Usage statistics and insights |
| Fine-tuned embeddings | Low | Domain-specific embedding model |

### 13.2 Architecture Improvements

| Improvement | Benefit |
|-------------|---------|
| Pinecone/Qdrant integration | Managed vector DB; horizontal scaling |
| Async embedding | Parallel ingestion |
| GraphRAG | Better multi-hop reasoning |
| Hybrid search (BM25 + dense) | Improved recall |

---

## 14. References

### Papers

1. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
3. Nogueira & Cho (2019). "Passage Re-ranking with BERT"

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [RAGAS Documentation](https://docs.ragas.io/)

---

## Author

**Umair Ali**  
Machine Learning Engineer

- LinkedIn: [umair-machine-learning-engineer](https://www.linkedin.com/in/umair-machine-learning-engineer/)
- Email: aumair472@gmail.com
- GitHub: [aumair472](https://github.com/aumair472)

---

*Document Version: 1.0.0 | Last Updated: March 2026*
