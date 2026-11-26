# AI Sales Voice Agent: Project Report

**Candidate:** Tirumala Sri |  **Project:** Autonomous AI Sales Voice Agent (Project 1)

## 1. Business problem
Sales teams lose leads when calls go unanswered, spend time on unqualified prospects, and follow up slowly. In India the problem is sharper because customers speak Hindi, Telugu and other languages. The demo domain is rooftop solar (SunNest Solar, a fictional company): a high-value purchase where qualifying the lead and booking a site visit is the key conversion step.

## 2. Solution
A voice agent that talks to a lead in English, Hindi or Telugu, answers product questions from the company's own documents (RAG), qualifies and scores the lead, books a site visit, and writes the lead, booking and a call summary to a CRM. A dashboard shows conversion, sentiment, languages and objections across calls.

## 3. Architecture
Caller -> browser speech-to-text -> dialogue API -> NLU (language, sentiment, intent, entities) -> RAG retrieval -> deterministic policy -> reply generator (LLM if configured, templates otherwise) -> browser text-to-speech. Booking, scoring and CRM writes are plain code; the LLM never decides them. See `architecture.svg`.

## 4. What is implemented
- Voice I/O via the browser Web Speech API (en-IN, hi-IN, te-IN)
- Per-call memory (slots and history), multilingual replies
- Explainable lead score (0-100, hot/warm/cold) and sentiment per turn
- RAG: markdown knowledge base, chunking, TF-IDF retrieval with query cleaning and Hindi/Telugu query expansion
- Appointment booking with conflict handling; database-level double-booking protection
- SQLite CRM (leads, bookings, calls), call summaries, business insights endpoint and dashboard
- Optional LLM phrasing via an API key, with automatic fallback to templates

## 5. Scalability to 100,000 users
100k registered users is about 1,000 concurrent calls at peak (assuming about 1% concurrency). Plan: stateless API pods on Kubernetes with autoscaling on concurrent sessions; Redis for session state; streaming STT/TTS over WebRTC/WebSocket; Postgres with read replicas for CRM; vector database for RAG; async queue for post-call work (summary, CRM sync, WhatsApp confirmation); model routing and prompt caching to control LLM cost; fallback chain LLM -> templates -> human callback.

## 6. Security and compliance
TLS and encryption at rest, secrets in a vault, PII minimisation and masking in logs, configurable retention, recording consent and do-not-call handling, per-tenant isolation, API auth and rate limiting, and prompt-injection defence (LLM output cannot trigger actions; code validates everything).

## 7. Costs and commercial feasibility
Rough planning range of US$0.05 to 0.15 per call-minute across telephony, STT, TTS and LLM (estimate; verify with current vendor pricing). Revenue models: per minute, per booked appointment, or SaaS plus usage. A pilot with one vertical would measure booking rate and cost per booked visit against a human caller.

## 8. Limitations and next steps
Browser speech is a prototype shortcut (no barge-in); NLU is rule-based; single-node SQLite and in-memory sessions. Next: streaming voice pipeline, embedding-based multilingual RAG, LLM tool-calling agent with validated tools, real CRM integration, automated evaluation suite.

## 9. Run
`pip install -r requirements.txt`, then `python simulate.py` (terminal demo) or `python app.py` and open http://localhost:8000 in Chrome.
