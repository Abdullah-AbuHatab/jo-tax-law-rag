"""Abuse and cost guards: daily budgets, per-client daily limit, concurrency gate, answer cache."""

import threading
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.answering import LOW_RETRIEVAL_SCORE, AnswerService, ServiceError
from app.embeddings import FakeEmbedder
from app.guards import AnswerCache, ConcurrencyGate, DailyBudget, GateFull, PerKeyDailyLimit, SingleFlight
from app.llm import FakeLLM
from app.schemas import AskResponse

QUESTION = "ما هي نسبة الضريبة العامة على استيراد أي سلعة أو خدمة أو توريد أي منهما؟"
OTHER_QUESTION = "ما هو الإعفاء الشخصي للشخص الطبيعي المقيم؟"


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def ask(client, question=QUESTION, ip="203.0.113.9", **body):
    return client.post("/ask", json={"question": question, **body}, headers={"CF-Connecting-IP": ip})


def guarded_client(make_client, **settings):
    base = {"client_ip_header": "cf-connecting-ip", "rate_limit_per_minute": 1000}
    return make_client(**{**base, **settings})


# ---- DailyBudget -------------------------------------------------------------


def test_daily_budget_resets_at_utc_midnight():
    clock = Clock(datetime(2026, 10, 25, 23, 59, 59, tzinfo=UTC))
    budget = DailyBudget(limit=2, clock=clock)
    assert budget.try_consume() and budget.try_consume()
    assert not budget.try_consume()
    assert budget.remaining() == 0
    assert budget.seconds_until_reset() == 1

    # 02:30 in Amman (+03:00) is still 23:30 UTC of the same day: no reset yet
    clock.now = datetime(2026, 10, 26, 2, 30, tzinfo=timezone(timedelta(hours=3)))
    assert budget.remaining() == 0

    clock.now = datetime(2026, 10, 26, 0, 0, 1, tzinfo=UTC)
    assert budget.remaining() == 2
    assert budget.try_consume(2)
    assert not budget.try_consume()


def test_daily_budget_never_negative_under_threads():
    budget = DailyBudget(limit=10)
    start = threading.Barrier(50)
    results: list[bool] = []
    lock = threading.Lock()

    def worker():
        start.wait()
        outcome = budget.try_consume()
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 10
    assert budget.remaining() == 0
    assert not budget.try_consume(5)  # an oversized request is refused whole, not partly spent
    with pytest.raises(ValueError):
        budget.try_consume(0)


# ---- per-client daily limit ------------------------------------------------------


def test_per_key_daily_limit_resets_daily_and_stays_bounded():
    clock = Clock(datetime(2026, 10, 25, 12, 0, tzinfo=UTC))
    limit = PerKeyDailyLimit(limit=2, clock=clock, max_keys=3)
    assert limit.allow("a") and limit.allow("a")
    assert not limit.allow("a")
    for key in ("b", "c", "d"):  # "a" is the oldest and is evicted at the 4th key
        assert limit.allow(key)
    assert len(limit) == 3
    clock.now += timedelta(days=1)
    assert limit.allow("b") and limit.allow("b")
    assert not limit.allow("b")


def test_per_client_daily_limit_returns_429(make_client):
    client, llm, _ = guarded_client(make_client)  # default DAILY_ASK_LIMIT_PER_CLIENT=30
    for _ in range(30):
        assert ask(client).status_code == 200
    response = ask(client)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "DAILY_LIMIT_REACHED"
    assert response.json()["error"]["message"]
    assert 1 <= int(response.headers["Retry-After"]) <= 86400
    assert ask(client, ip="203.0.113.10").status_code == 200  # other clients are unaffected
    assert len(llm.calls) == 1  # the 30 repeats were cache hits


# ---- answer cache ------------------------------------------------------------------


def test_cached_answer_still_counts_against_rate_limit(make_client):
    client, llm, _ = guarded_client(make_client, rate_limit_per_minute=2)
    assert ask(client).status_code == 200
    assert ask(client).status_code == 200  # cache hit
    response = ask(client)
    assert response.status_code == 429 and response.json()["error"]["code"] == "RATE_LIMITED"
    assert len(llm.calls) == 1

    client, llm, _ = guarded_client(make_client, daily_ask_limit_per_client=2)
    assert ask(client).status_code == 200
    assert ask(client).status_code == 200  # cache hit
    assert ask(client).json()["error"]["code"] == "DAILY_LIMIT_REACHED"


def test_same_question_hits_cache_one_llm_call(make_client):
    client, llm, _ = guarded_client(make_client)
    first = ask(client)
    # the same question with diacritics, a hamza variant and extra spaces is the same cache key
    second = ask(client, question="  ما هيَ نسبة الضريبة العامة على استيراد اي سلعة أو خدمة أو توريد أي منهما؟ ")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(llm.calls) == 1
    # law_id and top_k are part of the key
    assert ask(client, law_id="sales_tax_6_1994").status_code == 200
    assert ask(client, top_k=3).status_code == 200
    assert ask(client, top_k=5).status_code == 200  # 5 is the default top_k: same key as the first ask
    assert len(llm.calls) == 3


def test_errors_are_not_cached(make_client):
    client, llm, _ = guarded_client(make_client, responder=lambda m: "not json")
    for _ in range(2):
        data = ask(client).json()
        assert data["refused"] is True and data["error_code"] == "LLM_INVALID_OUTPUT"
    assert len(llm.calls) == 2


def test_answer_cache_expires_and_evicts_oldest():
    now = [0.0]
    cache = AnswerCache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    fields = {"refused": False, "refusal_reason": None, "citations": [], "dropped_citations": [], "retrieved": []}
    value = AskResponse(answer="x", error_code=None, model="m", **fields)
    errored = value.model_copy(update={"error_code": "LLM_INVALID_OUTPUT"})
    cache.put("errored", errored)
    assert cache.get("errored") is None
    cache.put("a", value)
    cache.put("b", value)
    cache.put("c", value)
    assert cache.get("a") is None and cache.get("b") is value and cache.get("c") is value
    now[0] = 10.0
    assert cache.get("b") is None
    assert AnswerCache.key_for("سؤال", None, 5) != AnswerCache.key_for("سؤال", "sales_tax_6_1994", 5)
    assert AnswerCache.key_for("سؤال", None, 5) != AnswerCache.key_for("سؤال", None, 3)
    disabled = AnswerCache(max_entries=0)
    disabled.put("a", value)
    assert disabled.get("a") is None


# ---- concurrency gate -------------------------------------------------------------


def test_concurrency_gate_unit():
    gate = ConcurrencyGate(max_in_flight=1)
    assert gate.acquire_nowait()
    assert not gate.acquire_nowait()
    gate.release()
    with gate, pytest.raises(GateFull), gate:
        pass
    assert gate.acquire_nowait()  # the outer block released its slot
    gate.release()


def test_concurrency_gate_returns_503_busy(make_client):
    entered, release = threading.Event(), threading.Event()

    class SlowLLM(FakeLLM):
        def complete(self, messages, timeout_seconds=None):
            entered.set()
            release.wait(10)
            return super().complete(messages, timeout_seconds)

    client, _, _ = guarded_client(make_client, llm=SlowLLM(), max_concurrent_asks=1)
    holder: dict[str, int] = {}
    thread = threading.Thread(target=lambda: holder.update(status=ask(client).status_code))
    thread.start()
    try:
        assert entered.wait(10)
        busy = ask(client, question=OTHER_QUESTION, ip="203.0.113.10")
        assert busy.status_code == 503
        assert busy.json()["error"]["code"] == "BUSY"
        assert busy.headers["Retry-After"] == "5"
    finally:
        release.set()
        thread.join(10)
    assert holder["status"] == 200
    assert ask(client, question=OTHER_QUESTION, ip="203.0.113.10").status_code == 200  # slot released


# ---- daily embedding / LLM budgets -----------------------------------------------------


def test_llm_budget_not_consumed_on_low_score_refusal(laws, filled_store):
    budget = DailyBudget(limit=1)
    llm = FakeLLM()
    gated = AnswerService(FakeEmbedder(), filled_store, llm, laws, min_similarity=0.99, llm_budget=budget)
    result = gated.ask(QUESTION)
    assert result.refused and result.refusal_reason == LOW_RETRIEVAL_SCORE
    assert budget.remaining() == 1 and llm.calls == []

    answering = AnswerService(FakeEmbedder(), filled_store, llm, laws, min_similarity=0.1, llm_budget=budget)
    assert not answering.ask(QUESTION).refused
    assert budget.remaining() == 0
    with pytest.raises(ServiceError) as info:
        answering.ask(OTHER_QUESTION)
    assert (info.value.code, info.value.http_status) == ("DAILY_BUDGET_EXHAUSTED", 503)
    assert len(llm.calls) == 1


def test_llm_budget_exhausted_returns_503(make_client):
    client, llm, _ = guarded_client(make_client, daily_llm_budget=1)
    assert ask(client).status_code == 200
    response = ask(client, question=OTHER_QUESTION)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DAILY_BUDGET_EXHAUSTED"
    assert int(response.headers["Retry-After"]) >= 1
    assert len(llm.calls) == 1
    assert ask(client).status_code == 200  # cached answers are still served


def test_embedding_budget_exhausted_returns_503(make_client):
    client, llm, _ = guarded_client(make_client, daily_embedding_budget=1)
    assert ask(client).status_code == 200
    response = ask(client, question=OTHER_QUESTION)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DAILY_BUDGET_EXHAUSTED"
    assert len(llm.calls) == 1


# ---- round 2: fairness (refunds) and single-flight ------------------------------------


def test_per_key_daily_limit_refund_gives_back_one_unit():
    limit = PerKeyDailyLimit(limit=2)
    assert limit.allow("a") and limit.allow("a")
    assert not limit.allow("a")
    limit.refund("a")
    assert limit.allow("a")
    assert not limit.allow("a")
    limit.refund("never-seen")  # no error, no negative counts
    assert limit.allow("never-seen") and limit.allow("never-seen") and not limit.allow("never-seen")


def test_budget_exhausted_refunds_the_daily_count(make_client):
    for budget in ({"daily_llm_budget": 1}, {"daily_embedding_budget": 1}):
        client, _, _ = guarded_client(make_client, daily_ask_limit_per_client=2, **budget)
        assert ask(client).status_code == 200  # 1 of 2 used
        exhausted = ask(client, question=OTHER_QUESTION)
        assert exhausted.json()["error"]["code"] == "DAILY_BUDGET_EXHAUSTED", budget  # refunded
        assert ask(client).status_code == 200, budget  # cached: 2 of 2 (a 429 without the refund)
        assert ask(client).json()["error"]["code"] == "DAILY_LIMIT_REACHED", budget


def test_busy_refunds_the_daily_count(make_client):
    entered, release = threading.Event(), threading.Event()

    class SlowLLM(FakeLLM):
        def complete(self, messages, **kwargs):
            entered.set()
            release.wait(10)
            return super().complete(messages, **kwargs)

    client, _, _ = guarded_client(make_client, llm=SlowLLM(), max_concurrent_asks=1, daily_ask_limit_per_client=1)
    holder: dict[str, int] = {}
    thread = threading.Thread(target=lambda: holder.update(status=ask(client).status_code))
    thread.start()
    try:
        assert entered.wait(10)
        busy = ask(client, question=OTHER_QUESTION, ip="203.0.113.10")
        assert busy.json()["error"]["code"] == "BUSY"
    finally:
        release.set()
        thread.join(10)
    assert holder["status"] == 200
    # the BUSY answer did not use up 203.0.113.10's only daily question
    assert ask(client, question=OTHER_QUESTION, ip="203.0.113.10").status_code == 200


def test_single_flight_unit():
    flights = SingleFlight(max_wait_seconds=0.2, max_waiters=1)
    assert flights.lead("k") is True
    assert flights.lead("k") is False  # someone is already computing "k"
    assert flights.lead("other") is True
    started = time.monotonic()
    assert flights.wait("k") is False  # the leader never finished: the wait is bounded
    assert time.monotonic() - started < 2
    flights.finish("k")
    assert flights.wait("k") is True  # nothing in flight any more: no wait at all
    assert flights.lead("k") is True
    flights.finish("k")
    flights.finish("other")


def test_identical_concurrent_questions_wait_for_the_first_answer(make_client):
    entered, release = threading.Event(), threading.Event()

    class SlowLLM(FakeLLM):
        def complete(self, messages, **kwargs):
            entered.set()
            release.wait(10)
            return super().complete(messages, **kwargs)

    client, llm, _ = guarded_client(make_client, llm=SlowLLM(), max_concurrent_asks=1)
    flights = client.app.state.single_flight
    results: dict[str, object] = {}
    first = threading.Thread(target=lambda: results.update(first=ask(client)))
    first.start()
    try:
        assert entered.wait(10)
        # same question from another client: with a gate of 1 this would be BUSY without single-flight
        second = threading.Thread(target=lambda: results.update(second=ask(client, ip="203.0.113.10")))
        second.start()
        deadline = time.monotonic() + 5
        while flights.waiting() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert flights.waiting() == 1
    finally:
        release.set()
        first.join(10)
        second.join(10)
    assert results["first"].status_code == results["second"].status_code == 200
    assert results["first"].json() == results["second"].json()
    assert len(llm.calls) == 1  # one upstream computation for both


# ---- round 2: defaults sized to the providers' free tiers -----------------------------------


def test_default_budgets_fit_the_free_tiers_and_render_uses_them():
    import yaml

    from app.config import REPO_ROOT, Settings

    defaults = Settings.from_env({})
    assert defaults.daily_embedding_budget == 800  # Gemini free tier: about 1,000 embedding requests a day
    assert defaults.daily_llm_budget == 45  # OpenRouter free models: about 50 requests a day without credits
    assert defaults.daily_ask_limit_per_client == 30
    assert defaults.daily_llm_calls_per_client == 5  # one client cannot use up the LLM budget alone
    blueprint = yaml.safe_load((REPO_ROOT / "render.yaml").read_text(encoding="utf-8"))
    env = {item["key"]: item.get("value") for item in blueprint["services"][0]["envVars"]}
    assert env["DAILY_EMBEDDING_BUDGET"] == "800" and env["DAILY_LLM_BUDGET"] == "45"
    assert env["DAILY_ASK_LIMIT_PER_CLIENT"] == "30" and env["DAILY_LLM_CALLS_PER_CLIENT"] == "5"
    assert env["TRUSTED_PROXY_HOPS"] == "0"  # header missing: prefer the peer over an X-Forwarded-For entry
    assert "TURNSTILE_SECRET_KEY" not in env and "TURNSTILE_SITE_KEY" not in env  # optional, off by default


# ---- round 3: a per-client daily allowance of LLM calls -------------------------------------

OFF_TOPIC = "What is the weather like on Mars today?"  # scores below MIN_SIMILARITY with the fake embedder


def _variant(i: int) -> str:
    return f"{QUESTION} رقم {i}"  # a different cache key, still on topic


def test_answer_service_counts_llm_calls_per_client(laws, filled_store):
    llm, budget, per_client = FakeLLM(), DailyBudget(limit=10), PerKeyDailyLimit(limit=2)
    svc = AnswerService(FakeEmbedder(), filled_store, llm, laws, 0.1, llm_budget=budget, llm_per_client=per_client)
    svc.ask(QUESTION, client_key="a")
    svc.ask(QUESTION, client_key="a")
    with pytest.raises(ServiceError) as info:
        svc.ask(QUESTION, client_key="a")
    assert (info.value.code, info.value.http_status) == ("DAILY_LIMIT_REACHED", 429)
    assert info.value.retry_after and info.value.retry_after > 0
    assert len(llm.calls) == 2 and budget.remaining() == 8  # refused before the call: nothing spent
    svc.ask(QUESTION, client_key="b")  # another client still can
    for _ in range(3):
        svc.ask(QUESTION)  # no client key (eval, tests): unlimited
    assert len(llm.calls) == 6

    gated = AnswerService(FakeEmbedder(), filled_store, llm, laws, 0.99, llm_budget=budget, llm_per_client=per_client)
    for _ in range(5):
        assert gated.ask(QUESTION, client_key="c").refusal_reason == LOW_RETRIEVAL_SCORE  # no LLM, not counted
    svc.ask(QUESTION, client_key="c")
    svc.ask(QUESTION, client_key="c")


def test_global_budget_exhausted_gives_the_per_client_llm_call_back(laws, filled_store):
    llm, per_client = FakeLLM(), PerKeyDailyLimit(limit=2)
    empty = AnswerService(
        FakeEmbedder(), filled_store, llm, laws, 0.1, llm_budget=DailyBudget(limit=1), llm_per_client=per_client
    )
    empty.ask(QUESTION, client_key="a")  # per-client 1 of 2, global 1 of 1
    with pytest.raises(ServiceError) as info:
        empty.ask(QUESTION, client_key="a")
    assert info.value.code == "DAILY_BUDGET_EXHAUSTED"
    fresh = AnswerService(
        FakeEmbedder(), filled_store, llm, laws, 0.1, llm_budget=DailyBudget(limit=10), llm_per_client=per_client
    )
    fresh.ask(QUESTION, client_key="a")  # 2 of 2: the failed call was given back
    with pytest.raises(ServiceError):
        fresh.ask(QUESTION, client_key="a")


def test_one_client_cannot_spend_more_than_five_llm_calls_a_day(make_client):
    client, llm, _ = guarded_client(make_client, daily_ask_limit_per_client=30, daily_llm_calls_per_client=5)
    for _ in range(3):
        assert ask(client, question=OFF_TOPIC).json()["refusal_reason"] == LOW_RETRIEVAL_SCORE  # not counted
    for i in range(5):
        assert ask(client, question=_variant(i)).status_code == 200, i
    over = ask(client, question=_variant(5))
    assert over.status_code == 429 and over.json()["error"]["code"] == "DAILY_LIMIT_REACHED"
    assert int(over.headers["retry-after"]) > 0
    assert len(llm.calls) == 5
    assert ask(client, question=_variant(0)).status_code == 200  # cached answers cost no LLM call
    assert ask(client, question=OFF_TOPIC).status_code == 200  # nor do low-score refusals
    assert ask(client, question=_variant(5), ip="203.0.113.10").status_code == 200  # other clients still can
    assert len(llm.calls) == 6


def test_llm_allowance_429_gives_the_daily_ask_back(make_client):
    client, _, _ = guarded_client(make_client, daily_ask_limit_per_client=3, daily_llm_calls_per_client=1)
    assert ask(client, question=_variant(0)).status_code == 200  # asks 1, LLM 1
    assert ask(client, question=_variant(1)).json()["error"]["code"] == "DAILY_LIMIT_REACHED"  # refunded
    assert ask(client, question=_variant(0)).status_code == 200  # asks 2 (cached)
    assert ask(client, question=_variant(0)).status_code == 200  # asks 3: a 429 without the refund
    assert ask(client, question=_variant(0)).json()["error"]["code"] == "DAILY_LIMIT_REACHED"
