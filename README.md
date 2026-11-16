# Project CrossVerify: Dual-Path LLM Verification Engine

[![Architecture: Dual-Path LLM Verification](https://img.shields.io/badge/Architecture-Dual--Path%20Verification-blue.svg)](#architecture)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Redis Streams](https://img.shields.io/badge/Broker-Redis%20Streams-DC382D.svg?logo=redis)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/Telemetry-PostgreSQL%2016-336791.svg?logo=postgresql)](https://www.postgresql.org)
[![Docker Compose](https://img.shields.io/badge/Infra-Docker%20Compose-2496ED.svg?logo=docker)](https://www.docker.com)

CrossVerify is an enterprise-grade, continuous verification and safe-deployment engine designed for production AI systems. It safely evaluates new **Candidate LLMs** against baseline **Champion LLMs** by shadowing live production traffic asynchronously without imposing latency penalties or reliability risks on end users.

---

## 1. Core Architecture

```
                                      +-------------------------+
                                      |   Client / User Traffic |
                                      +------------+------------+
                                                   |
                                                   v
                                      +-------------------------+
                                      |     FastAPI Gateway     |
                                      +----+---------------+----+
                                           |               |
              Synchronous Response Path    |               | Asynchronous Traffic Shadowing
         (Zero Latency Impact on User)     |               | (Mirror Payload + Prompt)
                                           v               v
                             +-------------------+   +-------------------+
                             |   Champion LLM    |   |   Redis Streams   |
                             |   (Production)    |   | (shadow_traffic)  |
                             +---------+---------+   +---------+---------+
                                       |                       |
                                       v                       v
                              [End-User Response]    +-------------------+
                                                     | Worker Consumer   |
                                                     +----+--------------+
                                                          |
                                           +--------------+--------------+
                                           |                             |
                                           v                             v
                                 +-------------------+         +-------------------+
                                 |   Candidate LLM   |         | Discrepancy Engine|
                                 |  (Inference Run)  |         | (Embeddings/FAISS)|
                                 +---------+---------+         +---------+---------+
                                           \                             /
                                            \                           /
                                             v                         v
                                        +------------------------------------+
                                        |    PostgreSQL Telemetry Store      |
                                        +------------------+-----------------+
                                                           |
                                                           v
                                        +------------------------------------+
                                        |      Canary Circuit Breaker        |
                                        |  (Sliding Window Drift / Abort)    |
                                        +------------------------------------+
```

### Key Pillars:
1. **Asynchronous Traffic Shadowing:** Live incoming completion prompts are duplicated to an event stream in Redis without delaying client responses.
2. **Semantic Discrepancy Engine:** Compares Champion and Candidate outputs using dense vector embeddings (`all-MiniLM-L6-v2`) and cosine similarity rather than brittle exact-string heuristics.
3. **Automated Canary Circuit Breaker:** Dynamically computes rolling statistics (drift rate, latency percentiles). If thresholds exceed safety limits, an abort flag trips in Redis to stop shadow routing.

---

## 2. Directory Structure

```
├── docker-compose.yml           # Core infrastructure orchestration
├── pytest.ini                   # Pytest discovery configuration
├── .env                         # Local runtime environment variables
├── .env.example                 # Reference environment template
├── .gitignore                   # Standard ignore rules
├── README.md                    # Architecture & operational runbook
├── gateway/                     # FastAPI Traffic Ingress & Proxy
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # API Gateway entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # /v1/chat/completions endpoint
│   │   └── schemas.py           # OpenAI-compatible Pydantic models
│   ├── clients/
│   │   ├── __init__.py
│   │   └── champion_client.py   # Synchronous Champion model client & simulator
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Settings loaded via pydantic-settings
│   │   └── broker.py            # Async Redis stream publisher stub (Day 3 hook)
│   └── tests/
│       ├── __init__.py
│       └── test_completions.py  # Unit & integration test suite
├── worker/                      # Background Consumer & Evaluator
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # Worker consumer loop
│   ├── clients/
│   │   ├── champion_mock.py     # Champion LLM mock / driver
│   │   └── candidate_mock.py    # Candidate LLM mock / driver
│   ├── evaluator/
│   │   ├── semantic.py          # SentenceTransformers cosine similarity
│   │   └── structural.py        # JSON schema validator
│   └── db/
│       ├── database.py          # Database session pooling
│       └── models.py            # SQLAlchemy telemetry models
├── canary_controller/           # Independent Circuit Breaker Service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                  # Telemetry aggregator & abort trip logic
└── infrastructure/
    └── postgres/
        └── init.sql             # Bootstrap schemas & extensions
```

---

## 3. Quickstart & Local Setup

### Prerequisites
- Docker Engine & Docker Compose (`v2+` or `docker-compose`)
- Python 3.11+ (for local development)

### Environment Configuration
Copy the sample environment file:
```bash
cp .env.example .env
```

### Launch Infrastructure Stack
Start PostgreSQL, Redis, and the API Gateway with Docker Compose:
```bash
docker-compose up --build -d
```

### Verify Service Health & Run Tests
- **API Gateway Root:** `http://localhost:8000/`
- **Gateway Healthcheck:** `http://localhost:8000/health`
- **Chat Completions:** `POST http://localhost:8000/v1/chat/completions`
- **Run Test Suite:**
  ```bash
  pytest
  ```

---

## 4. Implementation Roadmap

- [x] **Day 1: chore: init repo and docker compose stack**
  - Git repository structure, `.gitignore`, `.env.example`, `docker-compose.yml` (PostgreSQL 16, Redis 7, Gateway skeleton), `init.sql`, and architecture documentation.
- [x] **Day 2: feat: implement synchronous champion routing**
  - Build OpenAI-compatible `/v1/chat/completions` endpoint in FastAPI.
  - Implemented request/response Pydantic schemas, synchronous Champion client, latency telemetry headers (`X-Request-ID`, `X-CrossVerify-Latency-Ms`), and comprehensive pytest test suite.
- [x] **Day 3: feat: integrate async redis stream publisher**
  - Mirror incoming requests to Redis Stream (`shadow_traffic`).
  - Implemented resilient asynchronous `TrafficBroker` via Redis Streams (`XADD`), stream length capping (`MAXLEN ~ 100000`), dynamic canary circuit breaker gating, non-blocking fault isolation, broker telemetry, and comprehensive pytest test suite (13 passing tests).
- [x] **Day 4: feat: initialize candidate worker pool**
  - Background worker consuming from Redis consumer groups.
  - Implemented `ShadowConsumer` leveraging Redis consumer groups (`crossverify_workers`), auto-provisioning streams/groups (`XGROUP CREATE ... MKSTREAM`), structured deserialization (`ShadowEvent`), batch reading (`XREADGROUP`), safe message acknowledgement (`XACK`), signal handling, Docker Compose worker service orchestration, and unit tests (22 total passing tests).
- [x] **Day 5: feat: implement sandboxed candidate inference**
  - Execute prompts against candidate models in isolated worker context.
  - Implemented `SandboxedCandidateClient` and `ChampionMockClient` with strict `asyncio.wait_for` timeout sandboxing, complete exception isolation, drift injection support, latency tracking, and integration into `ShadowConsumer` (28 passing tests).
- [x] **Day 6: feat: provision postgres evaluation schema**
  - SQLAlchemy models and persistence for side-by-side evaluations.
  - Implemented async `DatabaseManager` with connection pooling, `EvaluationRecord` ORM schema, incremental SQL migrations (`001_initial_evaluations.sql`, `002_add_model_columns.sql`), fault-tolerant persistence, and in-memory async SQL test coverage (32 passing tests).
- [x] **Day 7: feat: integrate local embedding generation**
  - Local embedding pipeline using SentenceTransformers (`all-MiniLM-L6-v2`).
  - Implemented thread-safe lazy-loaded `EmbeddingEngine` with `asyncio.to_thread` offloading, batch encoding, L2 normalization, empty-string zero-vector guards, and real model verification tests (35 passing tests).
- [x] **Day 8: feat: calculate semantic cosine similarity**
  - Compute semantic similarity scores and classify discrepancies.
  - Implemented `SemanticEvaluator` using FAISS `IndexFlatIP` cosine similarity on normalized 384-d embeddings, threshold-based drift flagging, full end-to-end `ShadowConsumer` pipeline (Candidate inference -> Semantic similarity -> PostgreSQL telemetry persistence -> Redis `XACK`), and integration tests.
- [x] **Day 9: feat: enforce json schema validation**
  - Structural verification for function-calling and structured LLM payloads.
  - Implemented `StructuralEvaluator` supporting raw & fenced JSON extraction, schema validation via `jsonschema`, implicit/explicit schema enforcement, and telemetry persistence to `evaluations.structural_valid` (10 tests).
- [x] **Day 10: feat: build sliding window telemetry aggregator**
  - Continuous telemetry rollups over sliding time windows (5-minute averages).
  - Implemented `TelemetryAggregator` computing rolling 5-minute window metrics from PostgreSQL (drift rate, error rate, p50/p95/p99 latencies, structural pass rate, semantic similarity) with time-cutoff filtering and zero-division safety.
- [ ] **Day 11: feat: implement dynamic circuit breaker**
  - Trip abort flag on Redis if semantic drift exceeds 5% or latency spikes.
- [ ] **Day 12: feat: enforce gateway canary routing**
  - Gateway dynamically honors circuit breaker status to halt shadowing.
- [ ] **Day 13: test: load test async queue backpressure**
  - High-throughput concurrency tests and backpressure benchmarks.
- [ ] **Day 14: feat: add grafana observability dashboard**
  - Visual dashboards for drift metrics, latencies, and circuit breaker status.
- [ ] **Day 15: docs: architect comprehensive readme**
  - Production deployment runbook, trade-offs, and failure mode analysis.
