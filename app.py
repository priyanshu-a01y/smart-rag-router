"""Local Smart RAG Router API and lightweight dashboard."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

APP_NAME = "Smart RAG Router"
DB_PATH = os.getenv("DB_PATH", "rag_router.db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
SMALL_MODEL = os.getenv("SMALL_MODEL", "llama3.2:1b").strip()
LARGE_MODEL = os.getenv("LARGE_MODEL", "llama3.2:3b").strip()
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.88"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "384"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
ESTIMATED_COSTS = {SMALL_MODEL: float(os.getenv("SMALL_MODEL_ESTIMATED_COST", "0")),
                   LARGE_MODEL: float(os.getenv("LARGE_MODEL_ESTIMATED_COST", "0"))}

if not 0 <= CACHE_THRESHOLD <= 1 or REQUEST_TIMEOUT <= 0 or NUM_PREDICT < 1:
    raise ValueError("CACHE_THRESHOLD must be 0..1; request timeout and NUM_PREDICT must be positive")
if not SMALL_MODEL or not LARGE_MODEL:
    raise ValueError("SMALL_MODEL and LARGE_MODEL must be non-empty")

@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title=APP_NAME, version="2.0.0", description="Local semantic cache and Ollama model router.", lifespan=lifespan)
_embedding_model: Any = None
_embedding_lock = threading.Lock()
_inference_lock = threading.BoundedSemaphore(max(1, int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))))
_db_lock = threading.RLock()


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS cache (
          id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
          embedding TEXT NOT NULL, model TEXT NOT NULL, created_at REAL NOT NULL,
          hits INTEGER NOT NULL DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS requests (
          id TEXT PRIMARY KEY, question TEXT NOT NULL, model TEXT, cache_hit INTEGER NOT NULL,
          latency_ms REAL NOT NULL, estimated_cost REAL NOT NULL, created_at REAL NOT NULL,
          error TEXT)""")
        cols = {r["name"] for r in con.execute("PRAGMA table_info(requests)")}
        migrations = {
            "fallback_used": "INTEGER NOT NULL DEFAULT 0", "requested_model": "TEXT",
            "embedding_ms": "REAL NOT NULL DEFAULT 0", "cache_lookup_ms": "REAL NOT NULL DEFAULT 0",
            "inference_ms": "REAL NOT NULL DEFAULT 0", "error_type": "TEXT",
        }
        for name, declaration in migrations.items():
            if name not in cols:
                con.execute(f"ALTER TABLE requests ADD COLUMN {name} {declaration}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at DESC)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at DESC)")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


def get_embedder():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _embedding_model


def encode(question: str) -> tuple[list[float], float]:
    start = time.perf_counter()
    encoded = get_embedder().encode(question, normalize_embeddings=True)
    vector = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    return vector, (time.perf_counter() - start) * 1000


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("embedding dimensions do not match")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def find_cache(question: str, vector: list[float] | None = None):
    if vector is None:
        vector, embed_ms = encode(question)
    else:
        embed_ms = 0.0
    started = time.perf_counter()
    with db() as con:
        rows = con.execute("SELECT * FROM cache ORDER BY created_at DESC LIMIT 5000").fetchall()
    best = None
    for row in rows:
        try:
            saved = json.loads(row["embedding"])
            if not isinstance(saved, list) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in saved):
                continue
            score = cosine(vector, saved)
            if best is None or score > best[0]:
                best = (score, row)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return best, embed_ms, (time.perf_counter() - started) * 1000


def choose_model(question: str) -> tuple[str, dict[str, Any]]:
    q = question.lower()
    markers = ("compare", "analyze", "analyse", "step by step", "why", "trade-off", "tradeoff",
               "multiple documents", "design", "architecture", "recommend", "evaluate", "versus", "pros and cons")
    signals = {"analytical_terms": [m for m in markers if m in q],
               "multiple_questions": question.count("?") > 1,
               "long_context": len(question.split()) > 35,
               "multi_step_language": any(m in q for m in ("first", "then", "finally", "step by step"))}
    complex_q = bool(signals["analytical_terms"] or signals["multiple_questions"] or signals["long_context"] or signals["multi_step_language"])
    return (LARGE_MODEL if complex_q else SMALL_MODEL), {"strategy": "heuristic_complexity_v1", "signals": signals}


def estimated_cost(model: str) -> float:
    return ESTIMATED_COSTS.get(model, 0.0)


def classify_error(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout): return "timeout"
    if isinstance(exc, requests.ConnectionError): return "connection_error"
    message = str(exc).lower()
    if "not found" in message or "pull model" in message: return "model_not_found"
    if isinstance(exc, requests.RequestException): return "ollama_http_error"
    return "model_error"


def ask_ollama(question: str, model: str) -> str:
    prompt = ("Answer clearly and accurately. If context is missing, say so. Keep the response concise.\n\nQuestion: " + question)
    try:
        with _inference_lock:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model, "prompt": prompt, "stream": False,
                "keep_alive": KEEP_ALIVE, "options": {"num_predict": NUM_PREDICT}}, timeout=REQUEST_TIMEOUT)
        if response.status_code == 404:
            raise RuntimeError(f"Model '{model}' is unavailable in Ollama. Install it with: ollama pull {model}")
        response.raise_for_status()
        data = response.json()
        answer = str(data.get("response", "")).strip()
        if not answer: raise RuntimeError(f"Model '{model}' returned an empty answer")
        return answer
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request for '{model}' failed ({classify_error(exc)}): {exc}") from exc


def generate_with_fallback(question: str, requested_model: str):
    fallback = LARGE_MODEL if requested_model == SMALL_MODEL else SMALL_MODEL
    started = time.perf_counter()
    try:
        return ask_ollama(question, requested_model), requested_model, False, (time.perf_counter()-started)*1000
    except Exception as primary:
        if fallback == requested_model:
            raise RuntimeError(f"Selected model failed and no distinct fallback is configured: {primary}") from primary
        try:
            answer = ask_ollama(question, fallback)
            return answer, fallback, True, (time.perf_counter()-started)*1000
        except Exception as secondary:
            raise RuntimeError(f"Primary '{requested_model}' failed ({primary}); fallback '{fallback}' failed ({secondary})") from secondary


def log_request(question, model, hit, latency, cost, error=None, fallback=False, requested=None,
                embedding_ms=0, lookup_ms=0, inference_ms=0, error_type=None):
    with _db_lock, db() as con:
        con.execute("""INSERT INTO requests
          (id,question,model,cache_hit,latency_ms,estimated_cost,created_at,error,fallback_used,requested_model,embedding_ms,cache_lookup_ms,inference_ms,error_type)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (str(uuid.uuid4()), question, model, int(hit), float(latency), float(cost), time.time(), error,
           int(fallback), requested, embedding_ms, lookup_ms, inference_ms, error_type))


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "http_error"}})


@app.exception_handler(413)
async def too_large(request: Request, exc: Exception):
    return JSONResponse(status_code=413, content={"error": {"message": "Request body exceeds 16 KiB limit", "type": "request_too_large"}})


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": "internal_error"}})


@app.get("/", response_class=HTMLResponse)
def home():
    dashboard = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(dashboard)


@app.post("/api/ask")
def ask(body: AskRequest):
    started = time.perf_counter(); question = body.question
    try:
        vector, embed_ms = encode(question)
        cached, _, lookup_ms = find_cache(question, vector)
    except Exception as exc:
        raise HTTPException(503, f"Embedding service unavailable: {exc}")
    if cached and cached[0] >= CACHE_THRESHOLD:
        score, row = cached
        latency = (time.perf_counter()-started)*1000
        with _db_lock, db() as con:
            con.execute("UPDATE cache SET hits=hits+1 WHERE id=?", (row["id"],))
        log_request(question, row["model"], True, latency, 0, requested=row["model"], embedding_ms=embed_ms, lookup_ms=lookup_ms)
        return {"answer": row["answer"], "cache_hit": True, "similarity": score,
                "model": row["model"], "requested_model": row["model"], "fallback_used": False,
                "routing": {"strategy": "cache_reuse", "reason": "similarity_threshold_met"},
                "latency_ms": latency, "embedding_ms": embed_ms, "cache_lookup_ms": lookup_ms,
                "inference_ms": 0, "estimated_cost": 0}
    requested, routing = choose_model(question); infer_start = time.perf_counter()
    try:
        answer, actual, fell_back, inference_ms = generate_with_fallback(question, requested)
    except Exception as exc:
        inference_ms = (time.perf_counter()-infer_start)*1000
        log_request(question, requested, False, (time.perf_counter()-started)*1000, 0, str(exc),
                    requested=requested, embedding_ms=embed_ms, lookup_ms=lookup_ms,
                    inference_ms=inference_ms, error_type=classify_error(exc))
        raise HTTPException(503, f"{exc}. Check that Ollama is running and configured models are installed.")
    cost = estimated_cost(actual)
    with _db_lock, db() as con:
        con.execute("INSERT INTO cache(id,question,answer,embedding,model,created_at,hits) VALUES(?,?,?,?,?,?,0)",
                    (str(uuid.uuid4()), question, answer, json.dumps(vector), actual, time.time()))
    latency = (time.perf_counter()-started)*1000
    log_request(question, actual, False, latency, cost, fallback=fell_back, requested=requested,
                embedding_ms=embed_ms, lookup_ms=lookup_ms, inference_ms=inference_ms)
    return {"answer": answer, "cache_hit": False, "similarity": cached[0] if cached else 0,
            "model": actual, "requested_model": requested, "fallback_used": fell_back,
            "routing": routing, "latency_ms": latency, "embedding_ms": embed_ms,
            "cache_lookup_ms": lookup_ms, "inference_ms": inference_ms, "estimated_cost": cost}


@app.post("/api/benchmark")
def benchmark(body: AskRequest, runs: int = Query(default=3, ge=1, le=10)):
    question = body.question
    # Ensure a successful cached answer exists without counting it as a benchmark request.
    vector, embedding_ms = encode(question)
    cached, _, lookup_ms = find_cache(question, vector)
    if not cached or cached[0] < CACHE_THRESHOLD:
        requested, _ = choose_model(question)
        try: answer, model, _, _ = generate_with_fallback(question, requested)
        except Exception as exc: raise HTTPException(503, f"Could not prepare benchmark cache entry: {exc}")
        with db() as con:
            con.execute("INSERT INTO cache(id,question,answer,embedding,model,created_at,hits) VALUES(?,?,?,?,?,?,0)",
                        (str(uuid.uuid4()), question, answer, json.dumps(vector), model, time.time()))
    cache_times=[]; direct_times=[]; direct_models=[]
    for _ in range(runs):
        t=time.perf_counter(); match, _, _ = find_cache(question, vector); cache_times.append((time.perf_counter()-t)*1000)
        if not match or match[0] < CACHE_THRESHOLD: raise HTTPException(500, "Cached answer was not retrievable at configured threshold")
        requested, _=choose_model(question); t=time.perf_counter()
        try: _, actual, _, _=generate_with_fallback(question, requested)
        except Exception as exc: raise HTTPException(503, f"Direct model benchmark failed: {exc}")
        direct_times.append((time.perf_counter()-t)*1000); direct_models.append(actual)
    median_cache=sorted(cache_times)[len(cache_times)//2]; median_direct=sorted(direct_times)[len(direct_times)//2]
    return {"runs": runs, "cache_median_ms": median_cache, "direct_median_ms": median_direct,
            "cache_samples_ms": cache_times, "direct_samples_ms": direct_times,
            "speedup": median_direct/median_cache if median_cache > 0 else None,
            "cache_hit": True, "direct_model": direct_models[-1], "preparation_embedding_ms": embedding_ms,
            "preparation_lookup_ms": lookup_ms,
            "note": "Measured local timings. A small number of runs is not statistically reliable. Benchmark inference is excluded from request history."}


@app.get("/api/metrics")
def metrics():
    with db() as con:
        r=con.execute("""SELECT COUNT(*) total,COALESCE(SUM(cache_hit),0) hits,
          COALESCE(SUM(estimated_cost),0) cost,COALESCE(AVG(latency_ms),0) avg_latency,
          COALESCE(SUM(fallback_used),0) fallbacks,COALESCE(SUM(error IS NOT NULL),0) failures,
          COALESCE(AVG(embedding_ms),0) embedding,COALESCE(AVG(cache_lookup_ms),0) lookup,
          COALESCE(AVG(inference_ms),0) inference FROM requests""").fetchone()
        models=[dict(x) for x in con.execute("SELECT model,COUNT(*) count FROM requests WHERE model IS NOT NULL AND error IS NULL GROUP BY model")]
        latencies=[x[0] for x in con.execute("SELECT latency_ms FROM requests ORDER BY latency_ms")]
    n=r["total"]
    median=(latencies[(n-1)//2]+latencies[n//2])/2 if n else 0
    return {"total_requests": n,"cache_hits": r["hits"],"cache_misses": n-r["hits"],
      "cache_hit_rate": r["hits"]/n if n else 0,"estimated_cost":r["cost"],
      "average_latency_ms":r["avg_latency"],"median_latency_ms":median,"fallback_count":r["fallbacks"],
      "failed_requests":r["failures"],"average_embedding_ms":r["embedding"],
      "average_cache_lookup_ms":r["lookup"],"average_inference_ms":r["inference"],"model_usage":models,
      "cost_note":"Configurable estimates only; local Ollama has no per-request API charge."}


@app.get("/api/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    with db() as con:
        rows=con.execute("SELECT question,model,requested_model,cache_hit,latency_ms,estimated_cost,created_at,error,error_type,fallback_used,embedding_ms,cache_lookup_ms,inference_ms FROM requests ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
    return {"items":[dict(x) for x in rows]}


@app.get("/health")
@app.get("/api/health")
def health(): return {"status":"ok","app":APP_NAME,"version":"2.0.0"}


@app.get("/ready")
@app.get("/api/ready")
def ready():
    try:
        with db() as con: con.execute("SELECT 1").fetchone()
        return {"status":"ready","database":"ok","ollama":"not_checked"}
    except sqlite3.Error:
        raise HTTPException(503,"Database is not ready")



