# jo-tax-law-rag

Arabic question answering over Jordanian tax laws. Every answer cites the article it comes from, and the
system refuses when the laws it has do not answer the question.

**Live demo:** https://jo-tax-law-rag.onrender.com (API docs at `/docs`). Free hosting: the first request after
15 idle minutes takes about a minute, and an answer takes 10–25 s on the free models.

> **Not legal advice.** This is a portfolio project that returns automatically extracted information from
> public law texts. The texts may be outdated. Check the official text or ask a qualified professional before
> relying on anything it says.

Laws covered (official ISTD consolidated texts; see [`data/SOURCES.md`](data/SOURCES.md)):

- قانون ضريبة الدخل رقم (34) لسنة 2014 وتعديلاته (Income Tax Law, as amended by Law 38/2018): 82 articles
- قانون الضريبة العامة على المبيعات رقم (6) لسنة 1994 وتعديلاته (General Sales Tax Law, consolidated Sept. 2022):
  78 articles

## The problem

People ask tax questions in Arabic: "كم الإعفاء الشخصي؟", "ما نسبة ضريبة المبيعات؟". A plain LLM answers
fluently, often wrongly, and without a source. In tax and legal work, an answer without the article number
behind it is not worth much. This service:

1. answers **only** from the text of the laws,
2. cites the article for every answer and checks those citations in code,
3. **refuses** when retrieval finds nothing relevant, or when the model cannot ground its answer.

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Data preparation (tools/, once)"]
        RAW["data/raw<br/>official .docx / PDF"] --> FIX["repair PDF text layer<br/>(font cmap + GSUB)"]
        FIX --> CLEAN["data/clean/*.txt<br/>one heading per article"]
    end
    subgraph Ingest["python -m app.ingest"]
        CLEAN --> PARSE["parse articles"] --> CHUNK["1 chunk per article<br/>split only long ones"]
        CHUNK --> HASH{"content hash<br/>changed?"}
        HASH -- yes --> EMB["Gemini embeddings<br/>RETRIEVAL_DOCUMENT, 768-d"] --> PG[("Neon Postgres<br/>pgvector HNSW")]
        HASH -- no --> SKIP["skip"]
    end
    subgraph Ask["POST /ask"]
        Q["question"] --> QE["Gemini embedding<br/>RETRIEVAL_QUERY"] --> SEARCH["top-k cosine search"]
        PG --> SEARCH
        SEARCH --> GATE{"best score ≥<br/>MIN_SIMILARITY?"}
        GATE -- no --> REFUSE["refuse<br/>(no LLM call)"]
        GATE -- yes --> LLM["OpenRouter chat<br/>fenced context, JSON reply<br/>models: primary + fallbacks"]
        LLM --> VALIDATE["parse + validate JSON<br/>drop citations not retrieved"]
        VALIDATE --> ANSWER["answer + citations<br/>or refusal"]
    end
```

| Path | Responsibility |
|---|---|
| `app/parsing.py` | Clean file → `Law` / `Article` (handles `المادة (12)`, Arabic-Indic digits, ordinal words, `مكرر`) |
| `app/arabic.py` | Normalization for matching (diacritics, tatweel, alef/hamza, ta marbuta, ya, digits) |
| `app/chunking.py` | Article chunks, deterministic ids, content hashes, Arabic-aware splitting of long articles |
| `app/embeddings.py` | `Embedder` protocol, `GeminiEmbedder` (REST, batching, retries), `FakeEmbedder` |
| `app/store.py` | `VectorStore` protocol, `PgVectorStore`, `InMemoryStore` (same interface, for tests) |
| `app/db.py` | Lazy connection pool, plain-SQL migrations under an advisory lock |
| `app/llm.py` | `OpenRouterClient` (REST, fallbacks, stable error codes), `FakeLLM` |
| `app/prompts.py` | System prompt; retrieved articles fenced as untrusted data |
| `app/answering.py` | Gate → generate → validate ("the model proposes, code decides") |
| `app/main.py` | FastAPI app, error mapping, security headers, body cap, rate limit, static UI |
| `app/rate_limit.py` | Per-minute sliding-window limiter, client IP (Cloudflare header first), pseudonymous `client_key` |
| `app/guards.py` | Daily budgets (embeddings, LLM), per-client daily limit, concurrency gate, answer cache |
| `app/ingest.py`, `app/eval.py` | CLIs |
| `tools/` | Data preparation from the official files (not needed at runtime) |

## Design decisions

**Why article-level chunks.** Lawyers cite articles, so the retrieval unit is the citation unit. Each article is
one chunk. The 24 articles longer than 1,500 characters are split with LangChain's
`RecursiveCharacterTextSplitter`, using Arabic-aware separators (paragraph markers, `. ` `؟` `؛` `،`) and a
150-character overlap. Every piece keeps its law id and article number in its metadata and in its embedded
header line, so a piece of article 4 can never be cited as anything other than article 4.

**Why refuse before calling the LLM.** If the closest chunk is still far from the question
(`MIN_SIMILARITY`), there is nothing to ground an answer on. Calling the model at that point only invites a
hallucination and burns rate limit. The threshold check is deterministic, costs nothing, and is easy to test.

**Why validate citations in code.** The model's JSON is a proposal. Each cited `(law_id, article)` must
match a chunk that was actually retrieved for this question. Anything else is dropped and reported in
`dropped_citations`. If no valid citation is left, the answer becomes a refusal. Invalid JSON also becomes a
refusal, with error code `LLM_INVALID_OUTPUT`. The model never gets the final say on what counts as grounded.

**Why fenced context.** Retrieved text and the question are untrusted input. All instructions live in the system
prompt. Articles go into the user message inside `<reference_articles>` / `<article ...>` tags, labelled as
untrusted data. `<`, `>` and `&` are escaped, so text inside a law or a question cannot close a tag and pose as
instructions.

**Why the data took real work.** ISTD publishes the sales tax law only as a PDF whose text layer is garbled, a
Word 2010 ToUnicode bug. Instead of retyping or guessing, `tools/repair_pdf_tounicode.py` rebuilds the
glyph→letter maps from the embedded fonts' own `cmap`/`GSUB` tables. The result agrees with an independent web
copy on 96.9% of words, and every disagreement is explained by the 2022 amendment or by typos in the web copy.
Details are in [`data/SOURCES.md`](data/SOURCES.md).

Other choices you might ask about in an interview:

- **Idempotent ingest.** Chunk id = `law_id:article:index`, and `content_hash` = sha256(model, dimension,
  embedded text). A re-run embeds only new or changed chunks and deletes stale ones. Changing the embedding model
  re-embeds everything.
- **The app always starts.** No database connection is opened at startup. The first request that needs the
  database opens the pool, so a cold or unreachable Neon instance gives `503 DATABASE_UNAVAILABLE` instead of a
  crash. After a failed attempt, requests fail fast for 5 s (no queue of callers each waiting for the connect
  timeout), then the next one retries. `/health` (liveness, an async route that needs no worker thread) never
  touches the database; `/health/db` does, and it is private (see API).
- **REST, no vendor SDKs.** Gemini and OpenRouter are called with `httpx`. Every call has an explicit timeout.
  Ingest retries Gemini with exponential backoff and jitter on 429/5xx only (up to 4 attempts, honouring
  `Retry-After`). The query embedding of a question makes at most 2 attempts within `GEMINI_TIMEOUT_SECONDS`
  (15 s) in total, and the LLM call gets what is left of `ASK_DEADLINE_SECONDS` (35 s), at most
  `OPENROUTER_TIMEOUT_SECONDS` (20 s) of wall-clock time, so one question cannot hold a worker longer.
  OpenRouter failover happens server-side through its `models` array.
- **Stable error codes.** Clients see a fixed code and message, for example `LLM_TIMEOUT` or `LLM_RATE_LIMITED`.
  Upstream error text is never forwarded, and server logs record only the status and the upstream error code.
- **Rate limiting.** `/ask` uses a per-client sliding window held in process memory (a client is an IPv4
  address, or an IPv6 /64). That is correct for one instance (the Render free plan) and wrong for several, which
  would need Redis. `X-Forwarded-For` is trusted only for the configured number of proxy hops.

## API

| Method | Path | Notes |
|---|---|---|
| `GET`, `HEAD` | `/health` | liveness, no dependencies (Render's health check) |
| `GET` | `/health/db` | private: `404` unless header `X-Health-Token` equals `HEALTH_TOKEN`; then a real `SELECT 1`, `503` when the database is down |
| `POST` | `/ask` | `{question, law_id?, top_k?}`; per-client limits and daily budgets (see [Security & privacy](#security--privacy)); with Turnstile on, header `CF-Turnstile-Response` |
| `GET` | `/config` | `{"turnstile_site_key": …, "min_similarity": 0.60}` |
| `GET` | `/laws` | the available laws, with their Arabic version and coverage labels |
| `GET` | `/laws/{law_id}/articles` | article keys and labels in document order |
| `GET` | `/laws/{law_id}/articles/{n}` | full article text and amendment notes, plus `prev_article` / `next_article` and their labels; `n` = `6`, `4bis` or `4 مكرر` |
| `GET` | `/docs` | OpenAPI UI (Swagger UI pinned to one release); `/redoc` is off |
| `GET`, `HEAD` | `/` | the RTL Arabic page |

Example (shape only; the values come from a fake run):

```json
{
  "answer": "…",
  "refused": false,
  "refusal_reason": null,
  "error_code": null,
  "citations": [{"law_id": "sales_tax_6_1994", "law_name": "قانون الضريبة العامة على المبيعات …",
                 "article": "6", "article_label": "6", "snippet": "أ- تفرض بمقتضى هذا القانون …",
                 "source_url": "https://istd.gov.jo/…", "score": 0.81}],
  "dropped_citations": [],
  "retrieved": [{"chunk_id": "sales_tax_6_1994:6:0", "law_id": "sales_tax_6_1994", "article": "6",
                 "chunk_index": 0, "score": 0.81}],
  "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
  "disclaimer": "هذه الإجابة معلومات عامة … وليست استشارة قانونية …"
}
```

`refusal_reason` is one of `LOW_RETRIEVAL_SCORE`, `MODEL_REFUSED`, `NO_VALID_CITATIONS` or
`LLM_INVALID_OUTPUT`. Errors use the shape `{"error": {"code", "message"}}` with codes `VALIDATION_ERROR`,
`UNKNOWN_LAW`, `RATE_LIMITED` (429), `DAILY_LIMIT_REACHED` (429), `BUSY` (503), `DAILY_BUDGET_EXHAUSTED` (503),
`CAPTCHA_REQUIRED` (403), `CAPTCHA_FAILED` (403, or 503 when Cloudflare cannot be reached), `PAYLOAD_TOO_LARGE`,
`DATABASE_UNAVAILABLE`, `EMBEDDING_UNAVAILABLE`, `LLM_*`, `INTERNAL_ERROR` (500). The 429s, `BUSY`,
`DAILY_BUDGET_EXHAUSTED` and an unreachable Turnstile carry a `Retry-After` header in seconds.

## Run locally

Requirements: Python 3.12. Docker is optional (for pgvector).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1) Fully offline (no keys, no database): fake embedder + fake LLM + in-memory store
EMBEDDER=fake LLM=fake VECTOR_STORE=memory MIN_SIMILARITY=0.15 \
  uvicorn app.main:create_app_from_env --factory --reload
# open http://127.0.0.1:8000  (answers are placeholders: the fake LLM just quotes the top article)

# 2) Real providers + local pgvector
cp .env.example .env        # add GEMINI_API_KEY and OPENROUTER_API_KEY
docker compose up -d db
export DATABASE_URL=postgresql://rag:rag-local-only@localhost:5432/rag
python -m app.ingest        # embeds ~200 chunks once; re-runs skip unchanged ones
uvicorn app.main:create_app_from_env --factory

# or everything in containers
docker compose up --build
docker compose run --rm api python -m app.ingest
```

### Tests

```bash
ruff check . && ruff format --check .
pytest                       # unit + API tests; pgvector tests are skipped
docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/postgres pytest   # adds the pgvector tests
# (they drop law_chunks and create throwaway roles: use a throwaway container only)
```

The tests use the real cleaned law text. They cover article parsing and heading variants, normalization,
chunk ids and hashes, idempotent re-ingest, the refusal gate, citation dropping, JSON-output edge cases, OpenRouter
error mapping and fallback body, the Gemini request shape and retry policy (both with mocked `httpx`), the HTTP
API, and pgvector search against a real database. They also check that every golden-set expectation quotes text
that really is in the cited article.

## Ingestion against Neon

1. Create a Neon project (free tier). Copy the **pooled** connection string (host contains `-pooler`, keep
   `sslmode=require`).
2. Run once from your machine with the **owner** connection string. Ingest always applies the migrations
   (the `vector` extension, the table and the HNSW index); the web service never does:

   ```bash
   export DATABASE_URL='postgresql://…-pooler…neon.tech/neondb?sslmode=require'
   export GEMINI_API_KEY=…
   python -m app.ingest --dry-run   # shows what will be embedded, no calls
   python -m app.ingest
   ```

3. Create the SELECT-only role the web service connects with (once; run it again to rotate its password):

   ```bash
   # Neon: add channel_binding=require to the owner URL (Neon's own connection strings include it)
   OWNER_DATABASE_URL="$DATABASE_URL" python -I tools/create_reader_role.py --plaintext-password
   # prints only "role ready"; the reader URL goes to .env as DATABASE_URL_READER=…
   ```

   **Least privilege.** In one transaction, the tool creates `rag_reader` (login only: no superuser, createdb,
   createrole, replication or bypassrls) with CONNECT on the database, USAGE on schema `public` and SELECT on
   `law_chunks` and `schema_migrations`, nothing else. It limits the role to 20 connections and sets
   `statement_timeout = 5s` and `idle_in_transaction_session_timeout = 10s` on the role itself, so every session
   gets them (the app's pool sends no startup options, which Neon's pooler may reject). It then checks that no
   write or CREATE privilege is left, and refuses, changing nothing, if the role belongs to another role. Paste
   `DATABASE_URL_READER` into Render as `DATABASE_URL`.

   **`--plaintext-password`.** By default the password reaches the server only as a SCRAM-SHA-256 verifier
   computed locally, so its plaintext is never part of any SQL text. Neon rejects pre-hashed passwords, so on
   Neon you need this flag: the password is then sent as a quoted literal. The tool refuses the flag unless the
   owner URL authenticates the server, with `channel_binding=require` or `sslmode=verify-full`.
   `sslmode=require` alone encrypts the connection but does not stop a man in the middle. The literal may appear
   in server-side statement logs or statistics (for example `pg_stat_statements`). If that matters, rotate by
   running the tool again.

4. Run ingest again whenever `data/clean/` changes. Only changed articles are re-embedded. Restart the web
   service afterwards to clear its answer cache.

## Evaluation

`data/eval/golden.jsonl` holds 26 answerable Arabic questions and 6 out-of-scope ones that must be refused
(including a prompt-injection attempt). Each answerable item names its expected article. Its `comment` quotes
that article, and a test checks the quote.

```bash
python -m app.eval --fake        # offline smoke run (pipeline check only)
python -m app.eval               # real: Gemini + OpenRouter + DATABASE_URL, waits 4 s between questions
```

Metrics, written to `eval/results.md`: recall@k and MRR (retrieval, article level), refusal accuracy, citation
validity (the share of model-proposed citations that survived validation), and citation hit rate. The committed
`eval/results.md` is the **real run** from 2026-10-24 (26 answerable and 6 out-of-scope Arabic questions).

Real results (2026-10-24, gemini-embedding-001, top_k = 5, OpenRouter free models, 32 questions):

| metric | value |
|---|---|
| recall@5 | 96.2% (25 of 26) |
| MRR | 0.962 |
| refusal accuracy | 100% (6 of 6 out-of-scope refused, 0 answerable refused) |
| citation validity | 100% (no citation had to be dropped) |
| `MIN_SIMILARITY` chosen | 0.60 |

Threshold analysis: the lowest best-score among answerable questions was 0.704 and the highest among out-of-scope
questions was 0.636. At 0.60, four of the six out-of-scope questions are refused before any LLM call and the model
refuses the other two from the fenced context, so both layers are exercised. Raising it to about 0.66 would refuse
all six before the LLM on this set, but 32 questions are too few to tune that tightly, so 0.60 stays.
The one retrieval miss (`sal-02`, expected article 2, the definitions article) retrieved the articles that apply the
definition instead. To recalibrate, run the eval and compare the two score ranges in `eval/results.md`.

## Deployment (Render + Neon)

1. Push the repository to GitHub, then in Render: **New → Blueprint** → select the repo. `render.yaml` defines
   one free Docker web service in Frankfurt, health-checked on `/health`.
2. Render asks for the `sync: false` secrets: `DATABASE_URL` (the pooled **reader** URL from
   `tools/create_reader_role.py`, not the owner URL), `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and optionally
   `HEALTH_TOKEN` (a long random string; leave it unset and `/health/db` always answers 404). The blueprint also
   sets `RUN_MIGRATIONS=false`, `CLIENT_IP_HEADER=cf-connecting-ip`, `TRUSTED_PROXY_HOPS=0`, the daily budgets and
   the per-client limits.
   Render's health check stays on `/health`, which never touches the database. Do not point an uptime or
   keep-alive pinger at `/health/db`: it would keep Neon's compute awake.
3. Ingest into Neon from your machine (see above). The web service never ingests or migrates.
4. Open the service URL. The first request after idle time is slow: Render free services sleep after inactivity
   and Neon suspends its compute. The app answers `503` until the database wakes, and the UI says so.

CI (`.github/workflows/ci.yml`) installs the runtime from `requirements.lock` (the image's exact set), checks
that the lock matches `requirements.txt`, runs ruff, `pip-audit` on the lock, all tests against a
`pgvector/pgvector:pg16` service container, and the offline eval. Actions are pinned to commit SHAs. The Docker
base image and the pgvector image (in CI and `docker-compose.yml`) are pinned to digests. Dependabot
(`.github/dependabot.yml`, 7-day cooldown) proposes weekly updates, including `tools/requirements.txt`. It does
not update the pgvector digest: bump that by hand.

### Dependencies

`requirements.txt` is the short, human-edited list of direct pins. `requirements.lock` pins the whole closure
(44 packages) with hashes, and the Docker image installs only that file
(`pip install --require-hashes --no-deps`). After editing `requirements.txt`, or when Dependabot bumps it, which
it does without touching the lock, regenerate the lock in the image's own Python:

```bash
docker run --rm -v "$PWD:/work" -w /work python:3.12-slim \
  sh -c "pip install -q pip-tools==7.6.2 && \
         pip-compile --quiet --strip-extras --generate-hashes --output-file=requirements.lock requirements.txt"
```

pip-compile keeps the existing pins and changes only what `requirements.txt` forces. To pick up a fixed
transitive release that `pip-audit` reports, add `--upgrade-package <name>`.

## Security & privacy

The public demo runs on the owner's free-tier keys, so the main risks are quota and cost abuse, not data theft.
The corpus is public law text. To report a problem, see `SECURITY.md`.

### Privacy

- The text of a question is sent to Google Gemini (embedding) and to OpenRouter (the free model that writes the
  answer). Those providers' free tiers may log inputs or use them for training. The page says so above the form
  and asks users not to type personal data.
- This app does not log questions. It does not log client IP addresses either. Each request line carries a
  salted 12-hex `client_key` (salt random per process), the method, the path (percent-encoded, at most 200
  characters), the status and the latency. uvicorn's own access log, which would add the peer address, is off.

### Threat model (audits of 2026-10-25)

| # | Threat | Mitigation | Where |
|---|---|---|---|
| A1 | One client, or many rotating ones, drain the Gemini and OpenRouter quotas and take the demo down | Daily budgets for query embeddings (800) and LLM calls (45), set below the providers' free caps. Each client gets at most 30 questions and 5 LLM calls a day (cache hits and low-score refusals cost no LLM call), so one client can no longer use up the LLM budget alone. Rotating addresses (about 9 clients, or IPv6 /64s) is the remaining path, accepted below. An answer cache and single-flight make repeated or simultaneous identical questions cost one upstream call; Turnstile is optional | `app/guards.py`, `app/answering.py`, `app/captcha.py` |
| A2 | Wrong or forgeable client identity behind Cloudflare → Render | `CLIENT_IP_HEADER=cf-connecting-ip` (overwritten by Cloudflare) wins over `X-Forwarded-For`; every value must parse as one IP; an IPv6 client is its /64 (one host controls a whole /64), an IPv4-mapped address is its IPv4 | `app/rate_limit.py` |
| A3 | Slow upstream calls hold every worker thread | `MAX_CONCURRENT_ASKS` gate (`503 BUSY`); query embedding at most 2 attempts in 15 s; the LLM gets only what is left of a 35 s deadline per question, enforced on wall-clock time; waiting for an identical question is bounded (30 s, 8 waiters) | `app/guards.py`, `app/embeddings.py`, `app/llm.py`, `app/answering.py` |
| A4 | A chunked body (no `Content-Length`) bypasses the 16 KB cap | An ASGI middleware counts the bytes actually received and answers `413` | `app/main.py` |
| A5 | The web process connects as the database owner and runs migrations | It connects as `rag_reader` (SELECT on two tables; login only, no other attributes, no role memberships, 20 connections, 5 s statement timeout, 10 s idle-in-transaction timeout, all checked or set by the tool) with `RUN_MIGRATIONS=false`. The password goes as a SCRAM verifier, or with `--plaintext-password` (Neon) as a literal, and only over an authenticated TLS channel | `app/db.py`, `tools/create_reader_role.py` |
| A6 | Incomplete security headers; third-party script on `/docs` | HSTS, `nosniff`, `no-referrer`, `X-Frame-Options: DENY`, Permissions-Policy, COOP and CORP on every response, including 500s; CSP `default-src 'none'` on API responses, a strict CSP on `/` (same-origin only, no `data:` images, `object-src 'none'`, and Trusted Types with no policy, so `innerHTML`, `eval` and the like throw; only with Turnstile off, since its loader sets `script.src`); `405` keeps its `Allow` header; `/docs` loads Swagger UI 5.33.3 with Subresource Integrity, and its CSP allows only that package path plus the hash of its one inline script; ReDoc off; no `server` header | `app/main.py`, `Dockerfile` |
| A7 | Spoofed-looking, oversized or link-carrying model output | Control, bidi and invisible characters stripped (soft hyphen, zero-width characters, word joiner, BOM, tag characters), at most 3 blank lines in a row, 4000 characters. Links, e-mail addresses and bare domains with a real TLD are replaced by «[رابط محذوف]», also when written with look-alike dots (`．` `。`), unless they are a law's own source URL or its host. The prompt forbids links and contact details; the UI renders text with `textContent` only; a 422 echoes only a plain field path of at most 64 characters | `app/answering.py`, `app/prompts.py`, `app/main.py` |
| A8 | Supply chain | Hashed lock file for the whole runtime closure, checked in CI and audited with `pip-audit`; Actions pinned to commit SHAs; base image pinned to a digest; Dependabot with a 7-day cooldown; `SECURITY.md` | `requirements.lock`, `.github/`, `Dockerfile` |
| A9 | `HEAD /` returned 405; `/favicon.ico` returned 404 | `HEAD` on `/` and `/health`; `/favicon.ico` serves an SVG | `app/main.py` |
| A10 | Anyone can call `/health/db` to keep Neon awake, or pile up requests on a stalled database | `/health/db` is private (`X-Health-Token`, constant-time compare, `404` otherwise); the pool fails fast for 5 s after a failed open and closes the failed pool | `app/main.py`, `app/db.py` |
| A11 | Log forging through the decoded request path | The logged path is percent-encoded (controls, spaces, `=` and bidi characters cannot forge fields) and capped at 200 characters | `app/main.py` |
| A12 | Unhandled errors answered outside the security middleware | Caught inside it: `500 INTERNAL_ERROR` with every header, `X-Request-ID` and a request log line; the traceback goes to the log only | `app/main.py` |
| A13 | Model output (which an injected question can shape) makes parsing or sanitizing take seconds to minutes of CPU (ReDoS) | No regex around the JSON reply (a fence pattern was cubic: 4,000 spaces took about a minute); every repeat in the link pattern is bounded by DNS limits (it was quadratic: `a.` × 3,000 took 4.5 s). Regression tests time adversarial inputs | `app/answering.py` |
| A14 | The page showed an empty answer panel with its "every article was retrieved" badge before any question | CSS classes that set `display` overrode the `hidden` attribute. All rules now sit in cascade layers and one unlayered `[hidden]` rule outranks them; a contract test keeps it that way | `app/static/style.css` |

Already in place before the audits: parameterized SQL everywhere, the prompt context escaped and fenced as
untrusted data, citations validated against what was retrieved, the Gemini key sent in a header, a non-root
container, and upstream error text never forwarded to clients.

### Accepted risks

- **The limits and budgets live in process memory.** They reset when the free instance restarts, which Render
  does after about 15 idle minutes, so an attacker who waits can get a fresh budget. The real ceiling is
  provider-side: the Gemini free tier, OpenRouter's free-model limit, and the key's credit limit ($1). **Keep
  billing disabled on the Gemini project**, so that exhausting the free tier stops the demo instead of costing
  money.
- **Slow-header and slow-body connections** are absorbed by the Cloudflare and Render proxies in front of the
  app. The app adds no read timeouts of its own.
- **Rotating client addresses.** Each client is limited (30 questions and 5 LLM calls a day), but about nine
  addresses, or nine IPv6 /64s, together can still use up the 45 LLM calls of the day. Turnstile (optional,
  below) and the provider caps are the answers to that.
- **Prompt injection** can still shape the answer text as long as the answer cites a retrieved article. Links
  and contact details are removed and the length is capped, and only the person who typed the question sees the
  result.

### How the limits and budgets work

`POST /ask` passes these checks in order. Daily counters reset at 00:00 UTC (03:00 in Amman).

| Step | Setting (default) | When exceeded |
|---|---|---|
| 1. Per-client limit per minute | `RATE_LIMIT_PER_MINUTE` (10) | `429 RATE_LIMITED` |
| 2. Turnstile token, only when both Turnstile keys are set | `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY` (unset) | `403 CAPTCHA_REQUIRED` / `CAPTCHA_FAILED` |
| 3. Per-client limit per day | `DAILY_ASK_LIMIT_PER_CLIENT` (30) | `429 DAILY_LIMIT_REACHED` |
| 4. Answer cache: a hit returns at once, already counted by steps 1 and 3 | `ANSWER_CACHE_SIZE` (500), `ANSWER_CACHE_TTL_SECONDS` (86400) | - |
| 5. Single-flight: an identical question already being answered is waited for | 30 s at most, 8 waiters | then answered normally |
| 6. Requests in flight | `MAX_CONCURRENT_ASKS` (4) | `503 BUSY` |
| 7. Query embeddings per day, all clients | `DAILY_EMBEDDING_BUDGET` (800) | `503 DAILY_BUDGET_EXHAUSTED` |
| 8. LLM calls per client per day, spent only when the LLM is really called | `DAILY_LLM_CALLS_PER_CLIENT` (5) | `429 DAILY_LIMIT_REACHED` |
| 9. LLM calls per day, all clients, spent only when the LLM is really called | `DAILY_LLM_BUDGET` (45) | `503 DAILY_BUDGET_EXHAUSTED` |

`BUSY`, `DAILY_BUDGET_EXHAUSTED` and a used-up LLM allowance (step 8) give the client's daily question back,
because nothing was served. When step 9 says no, step 8's call is given back too. The cache key
is the normalized question (diacritics, hamza forms and spacing folded) plus `law_id` and `top_k`. Only
successful results are cached. Results with an `error_code` and error responses are not. Refusals below
`MIN_SIMILARITY` never call the LLM, so they spend no LLM budget. The two global budgets default to just under
the providers' free caps (Gemini `gemini-embedding-001`: reportedly about 1,000 requests a day; OpenRouter free
models: about 50 a day without purchased credits). Lower them if the key is shared with anything else.

### Optional: Cloudflare Turnstile

Off by default, and not enabled in `render.yaml`. To turn it on, create a Turnstile widget in the Cloudflare
dashboard for the service's hostname, then set both `TURNSTILE_SITE_KEY` and `TURNSTILE_SECRET_KEY` on Render.
Setting only one of them stops the app at start-up. When it is on:

- `GET /config` returns the site key, and the page renders the widget. Only then does the page CSP allow
  `https://challenges.cloudflare.com` (script and frame).
- `/ask` needs a fresh token in the `CF-Turnstile-Response` header. The server verifies it with Cloudflare's
  siteverify (5 s timeout) and fails closed: a missing token is `403 CAPTCHA_REQUIRED`, a bad one
  `403 CAPTCHA_FAILED`, and an unreachable Cloudflare `503 CAPTCHA_FAILED`. Failed checks do not count against
  the daily limit.

### Rotating keys and passwords

- **Gemini**: create a new key in Google AI Studio, set `GEMINI_API_KEY` on Render (saving redeploys the
  service), then delete the old key. Update your local `.env`.
- **OpenRouter**: create a new key (keep a per-key credit limit), set `OPENROUTER_API_KEY` on Render, then delete
  the old key.
- **Reader role**: run `tools/create_reader_role.py` again with `OWNER_DATABASE_URL`, then paste the new
  `DATABASE_URL_READER` into Render as `DATABASE_URL`. New connections with the old password fail at once, so
  expect `503 DATABASE_UNAVAILABLE` until the redeploy is live. The tool refuses, and changes nothing, if
  `rag_reader` was created by another role, is a member of another role, or holds an attribute only a superuser
  can remove. In that case drop the role and run it again.
- **Health token**: set a new `HEALTH_TOKEN` on Render.
- **Neon owner**: reset its password in the Neon console and update your local `.env`. The web service does not
  use it.
- If a key leaks, rotate it first, then check the provider's usage page.

## Limitations

- **Not legal advice**, and not a substitute for reading the law.
- **The laws may be outdated.** These are ISTD consolidations: the income tax text includes amendments up to
  Law 38/2018 (file saved January 2022), and the sales tax text is dated September 2022. Later amendments,
  regulations (أنظمة), instructions (تعليمات) and the sales tax schedules are not included.
- **Consolidated text caveats.** A consolidation is an editorial product, not the Official Gazette. The sales
  tax text was recovered from a PDF; its words are verified, but punctuation next to numbers was repaired by rule
  (see `data/SOURCES.md`).
- **Free model limits.** OpenRouter `:free` models have per-minute and per-day request caps (at the time of
  writing: about 20 per minute, and 50 per day without account credits). They can be slow, and they can
  disappear. Expect `LLM_RATE_LIMITED` under load. The free Gemini tier has its own quotas.
- **Retrieval is dense-only.** There is no keyword or BM25 fusion and no reranker. Questions phrased very
  differently from the law's wording can miss, and then the system refuses rather than guesses.
- **Limits live in process memory.** The rate limits, daily budgets and the answer cache are correct for a
  single instance only, and they reset when the process restarts (a Render free service restarts after it has
  slept). A budget therefore caps one process lifetime, not strictly one calendar day. See "Accepted risks".

## License

MIT for the code (see `LICENSE`). The law texts in `data/` are public legislative texts of Jordan and are not
covered by that license.
