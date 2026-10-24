# jo-tax-law-rag

Arabic question answering over Jordanian tax laws. Every answer cites the article it comes from, and the
system refuses when the laws it has do not answer the question.

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
| `app/main.py` | FastAPI app, error mapping, rate limit, static UI |
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
  crash, and the next request retries. `/health` (liveness) never touches the database; `/health/db` does.
- **REST, no vendor SDKs.** Gemini and OpenRouter are called with `httpx`. Every call has an explicit timeout.
  Gemini is retried with exponential backoff and jitter on 429/5xx only (it honours `Retry-After`). OpenRouter
  failover happens server-side through its `models` array.
- **Stable error codes.** Clients see a fixed code and message, for example `LLM_TIMEOUT` or `LLM_RATE_LIMITED`.
  Upstream error text is never forwarded, and server logs record only the status and the upstream error code.
- **Rate limiting.** `/ask` uses a per-IP sliding window held in process memory. That is correct for one instance
  (the Render free plan) and wrong for several, which would need Redis. `X-Forwarded-For` is trusted only for the
  configured number of proxy hops.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | liveness, no dependencies |
| `GET` | `/health/db` | real `SELECT 1`; `503` when the database is down |
| `POST` | `/ask` | `{question, law_id?, top_k?}`, rate-limited per IP |
| `GET` | `/laws` | the available laws |
| `GET` | `/laws/{law_id}/articles/{n}` | full article text and amendment notes; `n` = `6`, `4bis` or `4 مكرر` |
| `GET` | `/docs` | OpenAPI UI |
| `GET` | `/` | the RTL Arabic page |

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
`UNKNOWN_LAW`, `RATE_LIMITED`, `PAYLOAD_TOO_LARGE`, `DATABASE_UNAVAILABLE`, `EMBEDDING_UNAVAILABLE`, `LLM_*`.

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
```

The tests use the real cleaned law text. They cover article parsing and heading variants, normalization,
chunk ids and hashes, idempotent re-ingest, the refusal gate, citation dropping, JSON-output edge cases, OpenRouter
error mapping and fallback body, the Gemini request shape and retry policy (both with mocked `httpx`), the HTTP
API, and pgvector search against a real database. They also check that every golden-set expectation quotes text
that really is in the cited article.

## Ingestion against Neon

1. Create a Neon project (free tier). Copy the **pooled** connection string (host contains `-pooler`, keep
   `sslmode=require`).
2. Run once from your machine. The migration creates the `vector` extension, the table and the HNSW index:

   ```bash
   export DATABASE_URL='postgresql://…-pooler…neon.tech/neondb?sslmode=require'
   export GEMINI_API_KEY=…
   python -m app.ingest --dry-run   # shows what will be embedded, no calls
   python -m app.ingest
   ```

3. Run it again whenever `data/clean/` changes. Only changed articles are re-embedded.

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
2. Render asks for the `sync: false` secrets: `DATABASE_URL` (Neon pooled string), `GEMINI_API_KEY`,
   `OPENROUTER_API_KEY`.
3. Ingest into Neon from your machine (see above). The web service never ingests.
4. Open the service URL. The first request after idle time is slow: Render free services sleep after inactivity
   and Neon suspends its compute. The app answers `503` until the database wakes, and the UI says so.

CI (`.github/workflows/ci.yml`) runs ruff, all tests against a `pgvector/pgvector:pg16` service container, and
the offline eval.

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
- **Rate limiting is per process.** It is correct for a single instance only.

## License

MIT for the code (see `LICENSE`). The law texts in `data/` are public legislative texts of Jordan and are not
covered by that license.
