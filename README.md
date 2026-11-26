# AI Sales Voice Agent (SunNest Solar demo)

A working prototype of an autonomous voice sales agent: it talks to a lead in English / Hindi / Telugu, answers product questions from a knowledge base, qualifies and scores the lead, books a site visit, writes everything to a CRM and produces a call summary plus business insights.

## 1. Business problem

Inbound/outbound sales teams lose leads because calls go unanswered after hours, reps spend time on unqualified leads, and follow-up is slow. For Indian SMEs the language barrier (Hindi/Telugu) makes it worse. Solar is the demo domain (high-ticket, long consideration, strong need for qualification and site-visit booking), but the knowledge base is just markdown files in `knowledge/`, so any business can be swapped in.

Value: 24x7 answering, instant qualification, reps only see warm/hot leads with a summary, and appointments are booked during the first call.

## 2. What is implemented vs. designed

| Requirement | Status | Where |
|---|---|---|
| Real-time voice (STT + TTS) | Working in browser (Web Speech API). Production design: streaming STT/TTS over WebRTC/WebSocket | `static/index.html` |
| Contextual memory | Working: per-call session (slots + history) | `agent.py` SESSIONS |
| Multilingual | Working: en / hi / te detection + replies (templates; LLM path translates freely) | `analysis.py`, `agent.py` |
| Lead qualification + sentiment | Working: explainable additive score 0-100 (hot/warm/cold) with reasons | `analysis.py` |
| RAG | Working: TF-IDF retrieval over markdown KB, swappable for embeddings | `rag.py` |
| Appointment booking | Working: slot search, conflict handling, DB-level double-booking protection | `crm.py` |
| CRM integration | Working local CRM (SQLite) + adapter stub for HubSpot/Salesforce/Zoho | `crm.py` |
| Call summary + insights | Working: summary per call, aggregate dashboard | `agent.end_call`, `/api/insights` |
| Optional LLM | Set `ANTHROPIC_API_KEY` and the LLM phrases replies; unset = deterministic templates | `agent.llm_rephrase` |

## 3. Run it

```bash
pip install starlette uvicorn scikit-learn requests   # numpy comes with scikit-learn
python simulate.py        # scripted end-to-end call in the terminal, no mic needed
python app.py             # open http://localhost:8000 in Chrome/Edge
# optional: export ANTHROPIC_API_KEY=... to enable LLM phrasing
```

Chrome/Edge are best for the mic (Web Speech API). If speech recognition fails, use the text box; the backend is identical.

## 4. Architecture

See `architecture.svg` (and the flow below).

```
Caller -> Voice gateway -> STT -> Dialogue service -> TTS -> Caller
                                   |  NLU (lang, sentiment, intent, entities)
                                   |  RAG retrieve -> POLICY (decides) -> GENERATOR (LLM or template)
                                   +-> Session memory (Redis) / CRM + Calendar (Postgres) / LLM API
End of call -> async queue -> summary, score, CRM sync, WhatsApp confirmation -> analytics
```

Key design decision: **policy decides, the LLM only phrases.** Booking, scoring and CRM writes are deterministic code, so the LLM can't hallucinate an appointment or a score. If the LLM is down or slow, replies fall back to templates and the call continues. This is the reliability story to tell in the interview.

## 5. End-to-end workflow demonstrated

1. Agent greets in the caller's language.
2. Caller gives bill and city, so the lead is partially qualified.
3. Caller asks "how much does a 3 kW system cost?" and RAG answers from the pricing doc.
4. Caller says "that's expensive" and the objection handler pulls the financing/EMI chunk.
5. Caller agrees to a visit and the agent collects name, phone, day and time.
6. Slot is checked and booked. If taken, alternatives are offered (shown in the Hindi scenario in `simulate.py`).
7. Lead is upserted to CRM with score and tier; call ends with summary, next action and aggregate insights.

## 6. Scalability to 100,000 users

Assumptions (state these, they're estimates): 100k registered users, ~10% talking in a month, 3 min average call, ~1% simultaneously active at peak = **about 1,000 concurrent calls**. Voice is bound by concurrency and latency, not by total users.

- **Stateless API pods** behind a load balancer (Kubernetes + HPA on concurrent sessions/CPU). Session state moves from the in-process dict to Redis (TTL 30 min) so any pod serves any turn.
- **Streaming voice path:** WebRTC/WebSocket gateway (LiveKit or telephony provider) with media servers scaled separately from the dialogue service. Target end-to-end latency under ~1 s: streaming STT, LLM token streaming into streaming TTS, VAD for turn-taking and barge-in.
- **Database:** Postgres with read replicas, partition `calls` by month, `bookings.slot` uniqueness (already in prototype) keeps double-booking impossible at any scale. Transcripts to object storage.
- **RAG:** move to pgvector or Qdrant with multilingual embeddings, per-tenant namespaces, cache hot queries.
- **Async post-call work** (summary, CRM sync, WhatsApp) on a queue (SQS/Kafka) so it never blocks a live call.
- **LLM cost/latency control:** small fast model for most turns, larger model only for hard turns, prompt caching of the system prompt/KB context, hard timeout then template fallback.
- **Multi-region/HA:** deploy close to users (India region), health checks, graceful degradation order: LLM -> templates -> hand off to human callback.
- **Load testing:** simulate with scripted callers (like `simulate.py`) at 1,000 concurrent sessions before launch.

## 7. Data security and compliance

- TLS everywhere; encryption at rest for DB and recordings; secrets in a vault, never in code.
- PII minimisation: store phone/name only after the caller provides them; mask phone numbers in logs; configurable retention (e.g. delete recordings after 90 days).
- Consent: announce AI + recording at the start of the call, honour "don't call again" (the prototype already routes it to a DND outcome).
- India's DPDP Act and TRAI telemarketing/DND rules apply to outbound calling; check current requirements with legal before launch.
- Prompt-injection defence: the LLM sees user text as data, can't call tools, and actions are executed only by deterministic code. Rate limiting and input length caps on the API (500 chars/turn in the prototype).
- Auth: API keys / JWT per tenant, role-based access to the CRM dashboard, audit log of exports.
- Tenant isolation: per-tenant KB namespace and row-level security in Postgres.

## 8. Infrastructure, deployment, cost

- **Deploy:** Docker image -> Kubernetes (EKS/GKE) or ECS Fargate; GitHub Actions CI (lint, unit tests for `analysis.py`, a `simulate.py` smoke test), blue/green rollout.
- **Monitoring:** latency p50/p95 per stage (STT, retrieval, LLM, TTS), booking conversion, fallback rate, STT word-error rate per language, cost per call; alert on error spikes.
- **Cost model (rough, verify against current vendor pricing):** per call-minute you pay for telephony, STT, TTS and LLM tokens. A planning range is roughly **US$0.05 to 0.15 per minute** depending on vendors and voice quality. For ~10,000 calls x 3 min = 30,000 min/month that is on the order of **US$1.5k to 4.5k per month** plus infra (a few hundred dollars for a small cluster, DB, Redis). Open-source STT (Whisper) and self-hosted TTS cut vendor cost but add GPU ops.
- **Commercial feasibility:** a human SDR costs far more per qualified lead; pricing options are per-minute usage, per-booked-appointment, or SaaS seat plus minutes. Pilot with one client and measure booking rate and cost per booked visit.

## 9. Known limitations (be upfront about these)

- Browser speech APIs are a demo shortcut; no barge-in and accuracy varies for Telugu.
- Rule-based NLU is fast and explainable but brittle on unusual phrasing; production would use an LLM structured-output call for entities/intent, keeping the deterministic policy.
- Non-English replies without an LLM key use templates, and KB answers stay in English (the LLM path translates them).
- Sessions and SQLite are single-node; see section 6 for the production path.
- Sentiment is lexicon-based; production would use a trained multilingual model or LLM classification.

## 10. Live-coding cheat sheet (where to change things)

| If asked to... | Edit |
|---|---|
| Add a new language | `T` dict in `agent.py`, regex in `detect_language`, `LOCALE` in `index.html` (Telugu `te` is already added) |
| Change lead scoring weights | `score_lead` in `analysis.py` |
| Add business hours / Sunday open | `WORK_HOURS` and `free_slots` in `crm.py` |
| Swap the knowledge base | replace files in `knowledge/` (restart) |
| Plug in a real CRM | implement `push_to_external_crm` in `crm.py`, call it from `agent.handle_turn` |
| Add a new intent | `classify_intent` in `analysis.py` + a branch in `handle_turn` |
| Use embeddings instead of TF-IDF | reimplement `Retriever.search` in `rag.py` (same signature) |
