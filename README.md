# Journalist Agent - TFM VIU

AI Agent for automating journalistic workflows - Master's Thesis project (VIU 2026).

## Description

System that automates the collection, filtering, clustering, and generation of journalistic content (summaries, headlines, coverage angles) from external news sources. Built with LangGraph for agent orchestration and PostgreSQL + pgvector for persistence and vector search.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| Local LLM | LM Studio (GPT-OSS-20b) |
| Fallback LLM | DeepSeek API |
| Embeddings | LM Studio (nomic-embed-text-v1.5, 768d) |
| Clustering | HDBSCAN + UMAP |
| Database | PostgreSQL 16 + pgvector 0.8.0 |
| News Sources | NewsAPI, GNews, RSS |
| RAG | Advanced RAG with batch reranking, source citations |
| Query Routing | LLM-based Agent Router (prompt-based classification) |
| Bot | Telegram (aiogram 3.x) with RAG integration |
| API | FastAPI REST API |

---

## Quick Start

### Prerequisites

You need these services running:

| Service | Purpose | How to Start |
|---------|---------|--------------|
| PostgreSQL | Database | `docker-compose up -d` or local install |
| LM Studio | Local LLM + Embeddings | Open LM Studio app, load model |

### Installation

```bash
# Clone repository
git clone <repo-url>
cd VIU_TFM_Code

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials

# Setup database
poetry run alembic upgrade head
```

### Verify Installation

```bash
# Test LLM connection
poetry run python scripts/test_llm.py

# Test news connectors
poetry run python scripts/test_connectors.py
```

---

## Running the System

The system has multiple components that can run independently or together.

### Terminal Layout

For full functionality, you need **3 terminals**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         TERMINAL 1                               │
│                      PostgreSQL + LM Studio                      │
│  (Keep LM Studio app open with model loaded)                    │
├─────────────────────────────────────────────────────────────────┤
│                         TERMINAL 2                               │
│                         REST API                                 │
│  $ poetry run uvicorn src.api.main:app --reload --port 8000     │
├─────────────────────────────────────────────────────────────────┤
│                         TERMINAL 3                               │
│                       Telegram Bot                               │
│  $ poetry run python -m src.bot.main                            │
└─────────────────────────────────────────────────────────────────┘
```

### Option 1: Only the API

```bash
# Terminal 1: Start API server
poetry run uvicorn src.api.main:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/news/latest?limit=5
curl "http://localhost:8000/news/search?q=economia&limit=10"
```

### Option 2: Only the Telegram Bot

```bash
# Terminal 1: Start bot
poetry run python -m src.bot.main

# Bot will show:
# Bot @YourBotName started. Press Ctrl+C to stop.
```

### Option 3: Full System (API + Bot)

```bash
# Terminal 1: Start API
poetry run uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start Bot
poetry run python -m src.bot.main
```

### Option 4: Run Agent Manually (CLI)

```bash
# Basic execution
poetry run python scripts/test_agent.py

# With search query
poetry run python scripts/test_agent.py --query "economía"

# With specific sources
poetry run python scripts/test_agent.py --sources rss,newsapi,gnews

# Export results to JSON
poetry run python scripts/test_agent.py --output results.json
```

---

## System Flows

### Flow 1: Agent Router (Query Classification)

Every user query (both `/buscar` and free text) is first classified by the LLM-based Agent Router:

```
User query
    │
    ▼
Agent Router (LLM, temperature=0.1)
    │
    ├── LOCAL_RAG ──────────► Only search local DB (pgvector)
    │                         For: general topics, historical queries
    │
    ├── EXTERNAL_SEARCH ────► Go directly to external APIs (NewsAPI, GNews)
    │                         For: "hoy", "última hora", emerging topics
    │
    └── COMBINED ───────────► Local first, external if low confidence
                              For: ambiguous queries, broad topics
                              Also used as fallback if router fails
```

### Flow 2: Telegram Bot Search (`/buscar`)

```
User: /buscar Anthropic
         │
         ▼
    Agent Router classifies query
         │
         ├── LOCAL_RAG ────────────► Vector search → filter → return (~1 sec)
         │
         ├── EXTERNAL_SEARCH ──────► Fetch from NewsAPI/GNews → save → return
         │
         └── COMBINED ─────────────► Vector search in DB
                                          │
                                          ├── 3+ articles ──► Process & return (~1 sec)
                                          │
                                          └── 0-2 articles ─► Run journalist agent
                                                                     │
                                                                     ▼
                                                              Fetch + process (~166 sec)
                                                                     │
                                                                     ▼
                                                              Return combined results
```

### Flow 3: RAG Question Answering (free text)

```
User: ¿Qué está pasando con la economía?
         │
         ▼
    Agent Router classifies query
         │
         ├── LOCAL_RAG ────────► RAG only (vector search → rerank → generate)
         │
         ├── EXTERNAL_SEARCH ──► Fetch external → save → RAG over all articles
         │
         └── COMBINED ─────────► RAG first
                                    │
                                    ├── confidence > 0.3 ──► Return response
                                    │
                                    └── confidence ≤ 0.3 ──► Fetch external → retry RAG
                                                                    │
                                                                    ▼
                                                             Return best response
```

### Flow 4: Agent Pipeline

```
fetch → check_existing → filter → embed → load_similar → cluster → deduplicate → generate → quality → save
  │           │            │        │          │            │            │           │          │       │
  │           │            │        │          │            │            │           │          │       │
  ▼           ▼            ▼        ▼          ▼            ▼            ▼           ▼          ▼       ▼
NewsAPI    Skip if      Quality  Nomic     pgvector     HDBSCAN     Union-Find   LLM        Check   PostgreSQL
GNews      in DB        checks   embed     similarity   clustering  algorithm    summaries  length
RSS                                                                              headlines  format
                                                                                 angles     clickbait
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with DB and LLM status |
| `/news/latest` | GET | Latest processed articles (paginated) |
| `/news/search` | GET | Vector similarity search |
| `/news/{id}` | GET | Article by ID |
| `/news/{id}/processed` | GET | Processed article with generated content |
| `/ask` | POST | RAG-powered Q&A with citations |
| `/agent/run` | POST | Trigger journalist agent pipeline |
| `/agent/sessions` | GET | List agent sessions |

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# Get latest news
curl http://localhost:8000/news/latest?limit=5

# Search by topic
curl "http://localhost:8000/news/search?q=inteligencia%20artificial&limit=10"

# Ask a question (RAG)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué está pasando con la economía?", "max_sources": 5}'

# Run agent
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "tecnología", "max_articles": 10}'
```

---

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Show available commands |
| `/ultimas [N]` | Latest N news (default 5) |
| `/buscar <tema>` | Search news by topic |
| `/digest` | Daily summary |
| `/categorias` | List available categories |
| *free text* | RAG-powered Q&A with citations |

### Bot Examples

```
/ultimas 10          → Show last 10 news
/buscar Petro        → Search news about Petro
/buscar economía     → Search news about economy

¿Qué decisiones ha tomado el gobierno?  → RAG answer with sources
Cuéntame sobre inteligencia artificial  → RAG answer with sources
```

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=lm_studio
LM_STUDIO_BASE_URL=http://192.168.68.57:1234/v1
LM_STUDIO_MODEL=openai/gpt-oss-20b
DEEPSEEK_API_KEY=your_key  # Fallback

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/tfm_db

# News APIs
NEWSAPI_KEY=your_key
GNEWS_API_KEY=your_key

# Embeddings
USE_LM_STUDIO_EMBEDDINGS=true
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
EMBEDDING_DIMENSION=768

# Processing thresholds
SIMILARITY_THRESHOLD=0.65
DUPLICATE_THRESHOLD=0.95

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_from_botfather
```

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download models:
   - **LLM**: `gpt-oss-20b` or similar
   - **Embeddings**: `nomic-embed-text-v1.5`
3. Start the local server (default port: 1234)
4. Update `LM_STUDIO_BASE_URL` in `.env` if needed

---

## Project Structure

```
src/
├── agent/              # LangGraph agent
│   ├── graph.py        # Main workflow graph
│   ├── state.py        # Agent state definitions
│   └── nodes/          # Pipeline nodes (fetch, filter, embed, etc.)
├── router/             # LLM-based query routing
│   └── query_router.py # Agent Router (LOCAL_RAG / EXTERNAL_SEARCH / COMBINED)
├── api/                # FastAPI REST API
│   ├── main.py         # App initialization
│   ├── routes/         # Endpoint handlers
│   └── schemas.py      # Pydantic models
├── bot/                # Telegram bot
│   ├── main.py         # Bot initialization
│   ├── handlers/       # Command handlers (use Agent Router)
│   └── config.py       # Messages and constants
├── rag/                # RAG module
│   ├── retriever.py    # Vector search
│   ├── generator.py    # Citation generation
│   └── engine.py       # RAG orchestrator
├── llm/                # LLM provider abstraction
├── connectors/         # News source connectors
├── processing/         # NLP processing (embeddings, clustering)
├── generation/         # Content generation (summaries, headlines)
└── storage/            # Database layer (repositories)

scripts/                # CLI scripts for testing
docs/                   # Documentation and metrics
alembic/                # Database migrations
```

---

## Performance

| Scenario | Time | Description |
|----------|------|-------------|
| Agent Router classification | ~1-2 seconds | LLM classifies query into route |
| LOCAL_RAG search | ~1 second | Articles already in database |
| EXTERNAL_SEARCH (agent) | ~166 seconds | Fetch + process new articles |
| COMBINED (local hit) | ~2-3 seconds | Router + local RAG |
| COMBINED (external fallback) | ~168 seconds | Router + local + external |

The bottleneck is LLM content generation (~14 sec/article). Once processed, articles are served instantly from cache. The Agent Router adds ~1-2s overhead but avoids unnecessary external searches for queries answerable locally.

---

## Development

```bash
# Run tests
pytest
pytest --cov=src

# Code quality
black src/
ruff check src/
mypy src/

# Create new migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head
```

---

## License

This project is part of a Master's Thesis and is for academic purposes.

## Author

Daniel Zapata Grajales - Master in Artificial Intelligence, VIU 2026
