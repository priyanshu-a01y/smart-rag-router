# Smart RAG Router

A local-first FastAPI application that routes questions to Ollama models, reuses semantically similar answers from a SQLite cache, and exposes request history, metrics, and a simple benchmark. A browser dashboard is served by FastAPI at `/`.

> **Local development project.** The API has no authentication. Keep it bound to `127.0.0.1` and do not expose it to an untrusted network.

## Features

- Semantic answer cache backed by SQLite and sentence embeddings.
- Heuristic routing between a smaller and a larger Ollama model.
- Automatic retry with the other configured model when the selected model fails.
- Request history and measured cache, latency, fallback, and error metrics.
- Cache-versus-direct-inference benchmark endpoint.
- Obsidian-themed dashboard, served directly by FastAPI without a frontend build step.
- Additive SQLite schema migration for request metadata. The application does not include a database reset command.

## Architecture and request flow

`dashboard.html` is served by `app.py`. The browser calls the same-origin JSON API. For a question, the API lazily loads the configured Sentence Transformers embedding model, embeds the question, and compares it with up to the 5,000 newest cache entries using cosine similarity. At or above `CACHE_THRESHOLD` (default `0.88`), it returns the saved answer and records a cache hit. Otherwise, a small set of complexity signals selects `SMALL_MODEL` or `LARGE_MODEL`, then Ollama generates the answer. If that model call fails, the other configured model is attempted once. Successful answers and request metadata are written to SQLite.

The router is heuristic; this is not document retrieval over an indexed corpus. The cache is a nearest-neighbor scan over the newest 5,000 rows, not a vector database. The first uncached request may take longer while the embedding model initializes (and downloads its model if it is not already present). SQLite stores questions, answers, embeddings, and request history locally.

## Requirements

- Python 3.10 or newer (the project is tested with the included virtual environment).
- [Ollama](https://ollama.com/download) installed and running for uncached inference.
- Disk space and network access for the first Sentence Transformers model download unless already cached locally.

## Setup

Open PowerShell in the project directory.

### 1. Create and install the Python environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Install and prepare Ollama models

Install Ollama using its official installer, start the Ollama service, then pull both configured models:

```powershell
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama list
```

The default model names are `llama3.2:1b` (small) and `llama3.2:3b` (large). Both must be available for the full primary-plus-fallback behavior. You can override them with `SMALL_MODEL` and `LARGE_MODEL`. The Ollama HTTP API is expected at `http://127.0.0.1:11434` by default.

### 3. Configure and start the application

`.env.example` contains safe defaults and no secrets. Copy it if you want a local configuration file:

```powershell
Copy-Item .env.example .env
```

The application reads environment variables; it does **not** load `.env` automatically. Load the simple `KEY=value` example into the current PowerShell session before starting:

```powershell
Get-Content .env | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    Set-Item -Path "Env:$name" -Value $value
}
```

For a first run that keeps the repository's existing `rag_router.db` untouched, set a separate local database path (the directory must already exist):

```powershell
$env:DB_PATH = ".\local-test.db"
```

Start the API and dashboard on the requested local address:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Stop the foreground server with `Ctrl+C`. Opening `dashboard.html` directly from disk is unsupported; use the FastAPI URL so its API calls work.

## Configuration

All settings are optional environment variables with defaults in `app.py` and `.env.example`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DB_PATH` | `rag_router.db` | SQLite database path. Use a separate path to isolate local runs. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server base URL. |
| `SMALL_MODEL` | `llama3.2:1b` | Default model for simpler questions and one fallback target. |
| `LARGE_MODEL` | `llama3.2:3b` | Model for questions matching complexity heuristics and the other fallback target. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers model used for cache matching. |
| `CACHE_THRESHOLD` | `0.88` | Cosine similarity required for cache reuse; must be between 0 and 1. |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Timeout for Ollama generation calls. |
| `NUM_PREDICT` | `384` | Maximum generated tokens passed to Ollama. |
| `OLLAMA_KEEP_ALIVE` | `5m` | Ollama model keep-alive setting. |
| `MAX_CONCURRENT_INFERENCES` | `1` | Maximum concurrent local inference calls. |
| `SMALL_MODEL_ESTIMATED_COST` | `0` | Optional cost estimate recorded per small-model request. |
| `LARGE_MODEL_ESTIMATED_COST` | `0` | Optional cost estimate recorded per large-model request. |

Cost values are estimates only; local Ollama inference has no per-request API charge. Never put credentials or private values in a committed configuration file. `.env` is ignored by Git.

## API

Interactive OpenAPI documentation is available at `/docs`; the schema is at `/openapi.json`.

| Method and path | Purpose |
| --- | --- |
| `GET /` | Serves the dashboard HTML. |
| `GET /health` or `GET /api/health` | Process-level health response. |
| `GET /ready` or `GET /api/ready` | Checks SQLite readiness. Ollama is not contacted by this check. |
| `POST /api/ask` | Embed, cache lookup, route, infer if needed, and record the request. Body: `{"question":"..."}` (3–4,000 characters). |
| `GET /api/history?limit=20` | Newest request records; `limit` range is 1–100. |
| `GET /api/metrics` | Aggregated request, cache, latency, fallback, failure, model-use, and estimated-cost data. |
| `POST /api/benchmark?runs=3` | Compare cache lookup with direct model inference; `runs` range is 1–10. Body uses the same question schema. |

`/api/ask` returns answer text, cache status and similarity, requested and actual model, fallback status, routing signals, stage timings, and estimated cost. Unavailable embeddings or failed inference return HTTP 503. Invalid questions return validation errors. The benchmark prepares a cache entry if needed, measures repeated cache lookups and direct inference, and excludes its benchmark inference from normal request history. Its small run counts are useful for local comparison, not statistically rigorous performance claims.

## Testing

The repository's pytest suite uses an isolated temporary SQLite database and a fake embedder; it does not require Ollama or write to `rag_router.db`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

If the default temporary directory is restricted in your environment, give pytest an in-project temporary directory (ignored by Git):

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp
```

Compile the application and tests without running the server:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py tests
```

For a manual local inference check, start the server with a separate `DB_PATH`, confirm `GET /ready`, then submit `POST /api/ask` and repeat the same question to observe a cache hit. This check requires Ollama, both configured model tags for fallback coverage, and the embedding model. The automated suite mocks generation and focuses on API logic, caching, routing, fallback, metrics, and benchmark behavior.

## Repository contents

```text
.
├── .env.example
├── .gitignore
├── README.md
├── app.py
├── dashboard.html
├── requirements.txt
└── tests/
    └── test_app.py
```

Local virtual environments, `.env`, SQLite databases, logs, Python bytecode, pytest caches, and temporary files are excluded by `.gitignore`. No database file should be committed.
