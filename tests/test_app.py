import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import app as service


class FakeEmbedder:
    def encode(self, text, normalize_embeddings=True):
        return [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(service, "_embedding_model", FakeEmbedder())
    service.init_db()


def test_health_readiness_and_safe_migration(isolated_db):
    assert service.health()["status"] == "ok"
    assert service.ready()["status"] == "ready"
    with service.db() as con:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(requests)")}
    assert {"fallback_used", "requested_model", "embedding_ms", "cache_lookup_ms", "inference_ms", "error_type"} <= cols


@pytest.mark.parametrize("question", ["", "  ", "ok", "x" * 4001])
def test_invalid_question_validation(question):
    with pytest.raises(ValidationError):
        service.AskRequest(question=question)


def test_cache_miss_then_genuine_hit(isolated_db, monkeypatch):
    calls = []
    monkeypatch.setattr(service, "generate_with_fallback", lambda q, m: (calls.append(q) or "model answer", m, False, 4.0))
    first = service.ask(service.AskRequest(question="apple question"))
    second = service.ask(service.AskRequest(question="apple follow-up"))
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["answer"] == "model answer"
    assert len(calls) == 1
    assert second["similarity"] >= service.CACHE_THRESHOLD


def test_similarity_below_threshold_does_not_reuse(isolated_db, monkeypatch):
    monkeypatch.setattr(service, "generate_with_fallback", lambda q, m: ("answer", m, False, 1.0))
    service.ask(service.AskRequest(question="apple question"))
    response = service.ask(service.AskRequest(question="orange question"))
    assert response["cache_hit"] is False
    assert response["similarity"] < service.CACHE_THRESHOLD


def test_corrupt_cache_entry_is_ignored(isolated_db, monkeypatch):
    with service.db() as con:
        con.execute("INSERT INTO cache VALUES(?,?,?,?,?,?,0)", ("bad", "broken", "bad answer", "{bad", "m", 1.0))
    monkeypatch.setattr(service, "generate_with_fallback", lambda q, m: ("fresh", m, False, 1.0))
    response = service.ask(service.AskRequest(question="new question"))
    assert response["cache_hit"] is False
    assert response["answer"] == "fresh"


@pytest.mark.parametrize("question,expected", [
    ("hello there", service.SMALL_MODEL),
    ("Compare these approaches and explain the trade-off", service.LARGE_MODEL),
    ("Do A? Then do B?", service.LARGE_MODEL),
])
def test_router(question, expected):
    model, metadata = service.choose_model(question)
    assert model == expected
    assert metadata["strategy"] == "heuristic_complexity_v1"
    assert "signals" in metadata


def test_primary_model_success(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "ask_ollama", lambda q, m: calls.append(m) or "ok")
    answer, actual, fallback, _ = service.generate_with_fallback("q", service.SMALL_MODEL)
    assert (answer, actual, fallback) == ("ok", service.SMALL_MODEL, False)
    assert calls == [service.SMALL_MODEL]


def test_primary_failure_falls_back_once(monkeypatch):
    calls = []
    def responder(q, model):
        calls.append(model)
        if model == service.SMALL_MODEL:
            raise RuntimeError("offline primary")
        return "fallback answer"
    monkeypatch.setattr(service, "ask_ollama", responder)
    answer, actual, fallback, _ = service.generate_with_fallback("q", service.SMALL_MODEL)
    assert answer == "fallback answer" and actual == service.LARGE_MODEL and fallback
    assert calls == [service.SMALL_MODEL, service.LARGE_MODEL]


def test_both_models_fail(monkeypatch):
    calls = []
    def fail(q, model):
        calls.append(model)
        raise RuntimeError("unavailable")
    monkeypatch.setattr(service, "ask_ollama", fail)
    with pytest.raises(RuntimeError, match="fallback"):
        service.generate_with_fallback("q", service.SMALL_MODEL)
    assert calls == [service.SMALL_MODEL, service.LARGE_MODEL]


def test_primary_unavailable_and_distinct_fallback(monkeypatch):
    calls = []
    def fail_first(q, model):
        calls.append(model)
        if model == "missing-model": raise RuntimeError("model not found")
        return "ok"
    monkeypatch.setattr(service, "SMALL_MODEL", "missing-model")
    monkeypatch.setattr(service, "LARGE_MODEL", "available-model")
    monkeypatch.setattr(service, "ask_ollama", fail_first)
    result = service.generate_with_fallback("q", "missing-model")
    assert result[1:3] == ("available-model", True)
    assert calls == ["missing-model", "available-model"]


def test_failed_request_is_recorded_and_not_cached(isolated_db, monkeypatch):
    monkeypatch.setattr(service, "generate_with_fallback", lambda q, m: (_ for _ in ()).throw(RuntimeError("both failed")))
    with pytest.raises(HTTPException) as error:
        service.ask(service.AskRequest(question="failure question"))
    assert error.value.status_code == 503
    assert service.metrics()["failed_requests"] == 1
    with service.db() as con:
        assert con.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0


def test_history_and_metrics_match_records(isolated_db, monkeypatch):
    monkeypatch.setattr(service, "generate_with_fallback", lambda q, m: ("answer", m, True, 12.0))
    service.ask(service.AskRequest(question="some unique ordinary question"))
    history = service.history(20)["items"]
    metrics = service.metrics()
    assert len(history) == metrics["total_requests"] == 1
    assert history[0]["fallback_used"] == 1
    assert metrics["fallback_count"] == 1
    assert metrics["cache_misses"] == 1


def test_benchmark_uses_measured_calls_and_skips_history(isolated_db, monkeypatch):
    monkeypatch.setattr(service, "generate_with_fallback", lambda q, m: ("answer", m, False, 2.0))
    result = service.benchmark(service.AskRequest(question="benchmark question"), runs=2)
    assert result["runs"] == 2
    assert len(result["cache_samples_ms"]) == len(result["direct_samples_ms"]) == 2
    assert result["cache_hit"] is True
    assert result["direct_median_ms"] >= 0
    assert result["speedup"] >= 0
    assert service.history(20)["items"] == []

