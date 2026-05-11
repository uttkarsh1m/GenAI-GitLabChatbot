# 🦊 GitBot — GitLab AI Assistant

> An intelligent RAG-based chatbot for GitLab's Handbook and Direction pages, powered by Google Gemini, FAISS, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

GitBot allows employees and aspiring employees to easily access information from GitLab's [Handbook](https://handbook.gitlab.com) and [Direction](https://about.gitlab.com/direction/) pages through a conversational interface. It uses **Retrieval-Augmented Generation (RAG)** to provide accurate, source-cited answers grounded in GitLab's actual documentation.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           3-Layer Guardrails            │
│  Layer 1: Harmful content filter        │
│  Layer 2: Off-topic detection           │
│  Layer 3: Post-retrieval confidence     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         Query Processing                │
│  • Typo normalization                   │
│  • Abbreviation expansion               │
│  • Follow-up context resolution         │
│  • Vague query detection                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         Hybrid Retrieval                │
│  Stage 1a: FAISS semantic search        │
│  Stage 1b: BM25 keyword search          │
│  Stage 2:  Reciprocal Rank Fusion       │
│  Stage 3:  Cross-encoder reranking      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         Generation & Verification       │
│  • Gemini 2.5 Flash (streaming)         │
│  • Hallucination detection              │
│  • Answer quality verification          │
│  • Source citation                      │
└─────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `paraphrase-MiniLM-L3-v2` |
| Reranker | `ms-marco-TinyBERT-L-2-v2` |
| Vector Store | FAISS (IndexFlatIP) |
| Keyword Search | BM25 (rank-bm25) |
| Web Scraping | BeautifulSoup + requests |
| Model Server | Unix socket IPC |

---

## Project Structure

```
gitbot/
├── app.py                  # Streamlit application
├── ingest.py               # Data ingestion pipeline (run to rebuild knowledge base)
├── requirements.txt
├── .env.example            # Copy to .env and add your API key
│
├── src/
│   ├── chatbot.py          # RAG pipeline, guardrails, Gemini integration
│   ├── embeddings.py       # FAISS, BM25, hybrid search, reranker
│   ├── scraper.py          # GitLab page scraper with retry logic
│   ├── chunker.py          # Text chunking and deduplication
│   └── model_server.py     # Background model server (Unix socket)
│
├── data/
│   ├── faiss_index.bin     # Pre-built vector index (included in repo)
│   ├── chunks.json         # Pre-processed text chunks (included in repo)
│   └── gitlab_content.json # Raw scraped content — generated locally, not committed
│
├── .devcontainer/
│   └── devcontainer.json   # GitHub Codespaces configuration
│
└── .streamlit/
    └── config.toml         # Theme configuration
```

> **Note:** `faiss_index.bin` and `chunks.json` are committed to the repository so the app works immediately after cloning without needing to run `ingest.py`. `gitlab_content.json` is excluded from the repo (it is large and regeneratable).

---

## Prerequisites

- Python 3.9 or higher
- A free [Google Gemini API key](https://aistudio.google.com/app/apikey) (1,500 requests/day on free tier)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/GenAI-GitLabChatbot.git
cd GenAI-GitLabChatbot
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

---

## Running the App

The knowledge base (`faiss_index.bin` and `chunks.json`) is already included in the repository — no scraping needed to get started.

**Option A — With model server (recommended, faster startup)**

```bash
# Start model server in background — loads AI models once (~10s)
python src/model_server.py &
sleep 12

# Start the app
streamlit run app.py
```

**Option B — Without model server (simpler)**

```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

> First load takes ~15–20 seconds while AI models initialize. Subsequent queries are fast (~0.5s).

---

## Stopping the Server

```bash
pkill -f "streamlit run app.py"
pkill -f "model_server.py"
```

---

## Rebuilding the Knowledge Base

The included index covers GitLab's handbook as of the last commit. To re-scrape with the latest content:

```bash
rm data/faiss_index.bin data/chunks.json data/gitlab_content.json
python ingest.py
```

Expected output:
```
✅ Scraped 37 pages
✅ Created 609 chunks from 28 sources
✅ Built index with 609 vectors
✅ Data ingestion complete!
```

---

## Key Features

### Intelligent Query Handling
- **Typo correction** — `"gitlab ccore values"` → `"gitlab core values"`
- **Abbreviation expansion** — `"gitlab eng"` → `"gitlab engineering"`
- **Follow-up resolution** — `"elaborate it"`, `"explain each"`, `"why is that"` are understood as continuations of the previous exchange
- **Vague query detection** — ambiguous queries like `"gitlab uses"` return a structured overview with a prompt to clarify

### Retrieval Pipeline
- **Hybrid search** combines BM25 keyword matching with FAISS semantic search, fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** re-scores candidates jointly with the query for higher precision
- **Deduplication** removes exact and near-duplicate chunks before indexing (reduced 816 → 609 chunks)

### Answer Quality
- **Hallucination detection** flags responses containing hedging language (`"I believe"`, `"typically"`, etc.)
- **Answer verification** checks length, query coverage, and citation presence
- **Streaming responses** — tokens appear as they are generated
- **Source citations** — every response links to the exact handbook section

### Guardrails
- **Layer 1** — blocks harmful or sensitive queries (passwords, confidential data, exploitation)
- **Layer 2** — redirects off-topic queries (geography, cooking, cryptocurrency, etc.)
- **Layer 3** — rejects queries where retrieved chunks have insufficient relevance

---

## Knowledge Base

GitBot indexes **37 pages** across two categories:

**Handbook (27 pages):** Values, Mission, Culture, OKRs, History, Engineering, Development, Infrastructure, Architecture, Product, Product Principles, UX, People Group, Hiring, Total Rewards, Leadership, Marketing, Sales, Finance, Legal, Security, Support, Communication, IT, Business Technology, Customer Success, Alliances

**Direction (10 pages):** Overview, Dev, Ops, Sec, Data Stores, AI/ML, ModelOps, Analytics, Verify, Plan

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Knowledge base not found` | The `data/` files may be missing. Run `python ingest.py` to rebuild them. |
| `API quota exceeded` | Create a new API key from a new project at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Slow startup (~20s) | Start `python src/model_server.py &` before the app and wait 12 seconds |
| Changes not reflected after edit | `src/` changes require a full server restart — hot-reload only applies to `app.py` |
| UI key not taking effect | Paste the key in the sidebar Settings panel and press Enter — it overrides the default key immediately |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---
