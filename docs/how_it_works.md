# How CareCall is built (apart from "call an LLM")

Read top to bottom once; later use it as a map. Every claim names the file that does it.

## 1. The shape of the system

```
 parent's phone ──(phone network)── Twilio ──(WebSocket, audio frames)──► our server (FastAPI)
                                                                              │
                              ┌───────────────────────────────────────────────┤
                              ▼                                               ▼
                   Gnani STT  (WebSocket)                               CallSession (the agent)
                   audio ──► text                                  text ──► LangGraph ──► reply text
                                                                              │
                                                       Gnani TTS (HTTP) ◄─────┘
                                                       text ──► audio ──► back to Twilio ──► parent

   around it:  SQLite (facts + audit events) · Langfuse (traces) · eval harness (personas, judge) · webhooks
```

Two kinds of data move through it, and they need different transports:
- **a continuous audio stream** (the live call): WebSocket
- **discrete request/response** (ask an LLM, fetch a TTS clip, a Twilio webhook): HTTP

## 2. Why a phone call is "live", and what streaming means here

### 2.1 Three ways to move data over a network

| | How it works | Direction | Good for |
|---|---|---|---|
| **Request/response HTTP** | client asks, server computes everything, sends one complete answer, connection done | client -> server -> client, once | normal APIs, a whole TTS clip, a whole LLM answer |
| **HTTP streaming** (chunked transfer / **SSE**, Server-Sent Events) | client asks once; the server keeps the response open and sends pieces as they are produced | server -> client only | LLM tokens, TTS audio as it is generated, live dashboards |
| **WebSocket** | one HTTP request "upgrades" the connection into a permanent two-way pipe; either side sends any time | both ways, continuous | a live call, realtime transcription |

What HTTP streaming looks like on the wire (this is how an LLM "types"): the response never says "done" until the end.

```
POST /chat/completions   {"stream": true, ...}
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"choices":[{"delta":{"content":"Thank"}}]}        <- 0.6 s after the request
data: {"choices":[{"delta":{"content":" you,"}}]}
data: {"choices":[{"delta":{"content":" Vimala."}}]}
...
data: [DONE]
```

### 2.2 The call itself is a WebSocket (not HTTP streaming)

`telephony/bridge.py`, `telephony/persona_bridge.py`, `api.py` (`/twilio/stream`).
- Twilio **Media Streams** opens a WebSocket **to our server** when the parent picks up (TwiML `<Connect><Stream>` in `telephony/twilio_api.py`).
- Every **20 ms** Twilio sends a JSON message containing 160 bytes of audio (base64): that is 8,000 samples/second of **8-bit mu-law**, the telephone codec. We send audio back the same way. So a call is ~50 tiny messages per second in each direction, for as long as the call lasts. That cannot be done with request/response HTTP, and SSE only goes one way.
- Conversions live in `telephony/audio.py`: mu-law <-> 16-bit PCM and resampling (`audioop`). STT wants 16-bit PCM, in **exactly 1024-byte frames, at real-time pace** (Gnani's rule); `GnaniSTT.send_audio` re-chunks Twilio's 160-byte frames into those.
- Control messages on the same socket: `start` (carries the signed per-call token), `media`, `mark` (Twilio echoes it when playback reaches that point), `clear` (stop playing, used for barge-in), `stop`.

### 2.3 Turn-taking: the hard part of "live"

Speech arrives as a flow, but the agent must act on *sentences*. `telephony/bridge.py`:
- **Endpointing**: Gnani's STT runs voice-activity detection and emits one transcript per speech segment after `x-min-silence-ms` of quiet (we use 600 ms). Shorter = faster but it cuts sentences at natural pauses (we measured 3 of 5 sentences split at 300 ms).
- **Barge-in**: if the parent starts talking while we are still playing audio, we send Twilio a `clear` so they are not talked over.
- **Silence timer**: if nobody speaks for 7 s after we finish, we inject an empty utterance so the agent can say "Are you still there?".
- **Persona side** (`persona_bridge.py`): replies only when the line has been quiet for `quiet_s` **and** no STT segment is open **and** a transcript is pending, because an agent turn can span several segments.

### 2.4 Where the code streams today, honestly

| Hop | Transport in the product path today | Streaming available? |
|---|---|---|
| Twilio <-> our server | **WebSocket** (continuous) | yes, in use |
| our server -> Gnani STT | **WebSocket** (continuous) | yes, in use |
| our server -> LLM | HTTP request/response, **waits for the full answer** | **yes (SSE), not used in the live path**; used only to measure time-to-first-token (`evals/latency.py`) |
| our server -> Gnani TTS | HTTP request/response, **waits for the full audio** (`GnaniTTS.synthesize`) | **yes (SSE / WebSocket), not used in the live path**; measured first audio chunk at 0.06 s vs 0.56 s for the whole clip |

That is exactly the optimisation left on the table. Measured turn time (parent stops talking -> agent audio starts):

```
today (everything waits):     STT 0.8  +  LLM 0.9-1.7  +  TTS 0.6-0.8   = 2.5 - 3.3 s
streaming pipeline:           STT 0.4* +  LLM first sentence 0.6  +  TTS first chunk 0.06  = ~1.2 s
                              (* needs the 300 ms silence window, which splits sentences)
```

How the streaming pipeline would work (sentence-level pipelining), because TTS cannot start until it has text:
```python
async for token in llm.stream(...):            # SSE: tokens arrive as they are produced
    buffer += token
    if buffer.endswith((".", "?", "!")):       # a full sentence is ready
        async for audio in tts.stream(buffer): # SSE: audio chunks arrive as they are produced
            await twilio_ws.send(audio)        # parent starts hearing sentence 1 while the LLM writes sentence 2
        buffer = ""
```
Why we have not done it for the **LangGraph** agent: the model only returns a small JSON classification there, and the *words spoken come from templates*, so there is no long LLM text to stream. Streaming pays off for the **model-written** agent (`agent/generative.py`) and for TTS.

## 3. Concurrency model (`asyncio`, threads, processes)

- FastAPI runs on an **asyncio event loop**: one thread juggles many sockets by switching at every `await`. A blocking call would freeze every live call.
- LangGraph, SQLAlchemy and the HTTP LLM client here are **synchronous**, so `bridge.py` runs them with `asyncio.to_thread(session.turn, ...)`: a worker thread does the slow work while the loop keeps pumping audio.
- Inside one call `StreamBridge` runs three cooperating tasks: the **receive loop** (Twilio -> STT), a **pump** (STT events -> an `asyncio.Queue`), a **consumer** (queue -> agent -> TTS -> Twilio). The queue decouples "text arrived" from "agent is busy".
- **Processes** (not threads) for parallel evals (`ProcessPoolExecutor` in `persona_runner.py`): the DB engine is a per-process global, so two threads would corrupt each other's transactions (we hit this).
- A scheduler is not built yet: promised callbacks are *recorded* (`callbacks` table), but nothing dials them.

## 4. The agent: two designs

| | LangGraph agent (`agent/graph.py`) | Model-written agent (`agent/generative.py`) |
|---|---|---|
| Who chooses the next step | code (a graph) | the model |
| Words spoken | templates | generated |
| The model's job | classify the reply into JSON (`agent/interpret.py`) | write the reply |
| Guards | many, in code | a few (pre-check, length cap, leak filter, loop stop) |
| Writes to the DB | yes, with the parent's exact quote | no |

**LangGraph concepts used**: a `StateGraph` over a typed state dict; **nodes** are functions (`safety`, `guardrail`, `interpret`, `validate`, `act`, `advance`, `escalate`); **conditional edges** route (emergency -> escalate, guard hit -> guardrail, else interpret); a **checkpointer** (`MemorySaver`) persists the state per call (`thread_id = call-<id>`) so each parent utterance is one `invoke` that resumes mid-conversation. Why in-memory: a dropped phone call cannot be resumed anyway.

**Key ideas**
- *Evidence-backed facts*: the model must return a verbatim quote; `validate` rejects answers whose quote is not in the parent's words, hedged answers ("maybe", "not sure"), filler "foods", implausible meal times. Models propose, code disposes.
- *Defence in depth*: the emergency check (`agent/interpret.py`) and the guardrails (`agent/guardrails.py`) are regexes that run **before** any model, so a confused or cheap model cannot hide an emergency or be talked into reciting a story.
- *Tools as plain functions* (`agent/tools.py`): the only code that writes facts; every call also appends an **event**.
- *Call-attempt state machine* (`calls/state.py`): legal transitions only (`scheduled -> dialing -> ringing -> in_progress -> completed/...`), so a duplicate or out-of-order Twilio webhook is ignored (**idempotency**). `calls/service.py` adds an **idempotency key** so "start this call" twice never dials twice.
- *Default meal times* (`agent/routines.py`): no time from the parent -> breakfast 08:30, lunch 12:30, dinner 19:30, recorded with `source="default"` so it can be corrected.
- *Per-use-case prompts* (`prompts/`): a shared base + one file per use case; a short hash of the text is the **prompt version**, stamped on traces.

## 4b. Guardrails (`agent/guardrails.py`)

A written scope policy turned into code: a detector with request cues (so "it reminds me of a story" is not a story request), canned replies, strike counting (3 ends the call), once-per-kind family alerts, varied wording on repeats, an output **length cap** (cut at a sentence boundary), and a **leak filter** (instructions, phone numbers, e-mail). Principle: a prompt is a request; a regex is a guarantee. Every guard has tests for what it must catch **and** ordinary speech it must not catch.

## 5. Data and audit

SQLAlchemy models (`models.py`), SQLite locally (`DATABASE_URL` can point to Postgres). Conventions: all timestamps **UTC**; every logged fact stores `source_utterance`; `events` is **append-only** (the audit trail, including every model interpretation and what it returned); `call_summaries`, `notifications` (an outbox), `test_notes`. In tests a **file** DB per test (in-memory SQLite shares one connection across threads and corrupted state).

## 6. Provider abstraction and configuration

`Protocol` interfaces (`STTProvider`, `TTSProvider`, `LLMClient`, `Interpreter`, `NoteWriter`) with real adapters (`telephony/gnani_speech.py`, `llm/openai_compat.py`) and fakes (`ScriptedSTT`, `FakeTTS`, `RuleInterpreter`). That is why all of this tests offline and why swapping Gnani's model in is one setting. Config is `pydantic-settings` reading `.env`; model roles are separate (`LLM_*` fast agent/interpreter, `AGENT_LLM_*`, `PERSONA_LLM_MODEL`, `JUDGE_LLM_MODEL`).

## 7. Security

- Twilio webhooks: **signature validation** (`RequestValidator`).
- The media WebSocket cannot carry auth headers, so each call gets an **HMAC token** in the stream's custom parameters, checked at `start`; forged tokens are closed (1008).
- Gnani's post-call webhook: shared **bearer token**.
- Secrets only in `.env`; keys are redacted in output; prompt-injection and OTP/scam defences in the guardrails.

## 8. Observability (`tracing.py`, Langfuse)

One **trace per call** (created up front, so turns running in different threads attach to the same trace), **spans** for open / each turn / post-call and each graph node and tool, a **generation** for every LLM call (full prompt, output, model), and **scores** (pass/fail, parent turns, ASR fidelity...). `session_id = call-<id>`; tags carry the suite, use case, persona, run id. All helpers are no-ops when Langfuse is down, so tracing can never break a call. Self-hosted via `docker/langfuse/docker-compose.yml` (Postgres, ClickHouse, Redis, MinIO).

## 9. Evaluation (`evals/`)

Scripted scenarios -> 25 **personas** (profile + hidden truth + expected outcome), played by a **scripted** parent or an **LLM** parent -> **deterministic checks** (quotes real, no advice, no leaks, expected logs/alerts) + an **LLM judge** with a rubric derived from each persona's expectations (every verdict must cite transcript turns; code re-checks) + an LLM **note writer** per transcript. Lessons baked in: *a judge needs its own audit* (several "failures" were judge bugs), keep the judge on a stronger model than the agent, and separate the persona model so backstories are respected. Guardrail personas add one off-topic request per persona. `carecall model-bench` and `carecall latency` measure speed/quality.

## 10. Reliability and performance engineering

- **Retries with backoff** on 429/5xx (`llm/openai_compat.py`); **fallback** to the keyword interpreter if the model call fails; per-persona error isolation in parallel runs.
- **IPv4 forcing** (`net.py`): on this network IPv6 was advertised but dead, and every new connection waited ~21 s per address. A real example of why latency work starts with measuring.
- **Connection reuse** (a pooled `httpx.Client`), **non-thinking** models with latency-sorted provider routing, p50/p90 not averages.

## 11. Testing strategy

Offline-first (`tests/`, ~137): fakes everywhere; a **Twilio simulator** plays Twilio's side of the WebSocket protocol; `httpx.MockTransport` checks API request shapes against the docs; regression tests for every bug the live runs found; live runs only for model behaviour. A fix is not done until a test fails without it.

## 12. File map

| Concept | Files |
|---|---|
| live audio, turn-taking | `telephony/bridge.py`, `persona_bridge.py`, `audio.py`, `gnani_speech.py`, `twilio_api.py`, `api.py` |
| agent | `agent/graph.py`, `session.py`, `routines.py`, `interpret.py`, `tools.py`, `context.py`, `generative.py`, `guardrails.py` |
| prompts | `prompts/` (interpreter), `prompts/gnani/` (agent) |
| data, state | `models.py`, `db.py`, `calls/state.py`, `calls/service.py`, `calls/post_call.py` |
| tracing, config | `tracing.py`, `config.py`, `net.py` |
| evals | `evals/*`, `personas/*.yaml` |
| CLI | `cli.py` (`carecall --help`) |
