# CareCall (Phase 0)

A voice agent that calls an elderly parent to track meals, remind about medicines (before/after food) and
deliver one-off reminders, logs what they say, and reports to a caregiver.
Phase 0 is local and hardcoded: one parent, fixed routines, triggered by hand (no clock scheduling yet).

## Quick start

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env                      # everything is optional; defaults run fully offline
carecall seed                               # (or any command: it seeds on first use)
carecall routines                           # what you can trigger
carecall chat --routine lunch_checkin       # you play the parent, by typing
carecall eval                               # scripted scenarios -> runs/<id>/notes/*.md + summary.md
carecall personas                           # 25 simulated parents -> runs/<id>/persona_report.md
pytest                                      # 40 tests
```

Routines: `meal_discovery`, `breakfast_checkin`, `lunch_checkin`, `snack_checkin`, `dinner_checkin`,
`metformin_reminder` (before food, 15 min before lunch), `amlodipine_reminder` (after food), `doctor_reminder`.

## Personas, prompts and the audit loop

- **25 personas, 5 per use case** (`src/carecall/personas/*.yaml`): meal_discovery, meal_checkin, medicine_before_food,
  medicine_after_food, defined_reminder. Each has a profile, traits, hidden *truth* and an expected outcome.
- **A different prompt per use case** (`src/carecall/prompts/`): one shared `base.md` + one file per use case, each with its
  own rules and few-shot examples. Every call records the prompt hash, so a trace tells you which prompt text produced it.
- **Run them:** `carecall personas` (scripted, offline) or `carecall personas --mode llm` (an LLM plays the persona).
  Read `runs/<id>/persona_report.md`. Findings and fixes so far: [docs/persona_audit.md](docs/persona_audit.md).
- **Keys:** paste them into `.env`, then `carecall check-keys`. Gnani publishes Speech APIs (Prisma STT, Timbre TTS) and
  an Agent Builder platform; the reasoning model (Pampa Go/Priority) is documented only as an Agent Builder setting, not a
  standalone endpoint. If you have a direct endpoint, set `LLM_PROVIDER=openai_compat` + `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`.
- **Langfuse:** `carecall langfuse-sync` uploads the prompts and the persona dataset so you can browse them.

## Observability: self-hosted Langfuse

```powershell
cd docker\langfuse ; copy .env.example .env ; docker compose up -d     # needs Docker Desktop running
# UI: http://localhost:3000   login dev@carecall.local / change-me-locally
# project + API keys are pre-created by the LANGFUSE_INIT_* variables
cd ..\.. ; carecall trace-check             # "connected" when keys in .env match
```

One **trace per call** (session `call-<id>`): spans for open / each turn / post-call, nested spans for every graph
node and tool, and a **generation** for every LLM call with its full prompt and output. Evals also attach scores
(`eval_passed`, `note_overall`) to the trace. If Langfuse is down or unconfigured, tracing silently turns off.

## How a call works

```
trigger -> CallSession.open()  -> build context -> LangGraph "open": greeting + first question
parent speaks -> CallSession.turn(text)
   safety (regex emergency words, no model) -> interpret (LLM or rules) -> validate (quote must be verbatim)
   -> act (tools write DB rows + audit events) -> advance (choose next words)
hangup -> CallSession.finish() -> post-call: summary, anomalies, review flag, caregiver notification
```

Key decisions (and the tradeoff behind each):

| Decision | Why | Cost |
|---|---|---|
| Code decides what happens next; the model only *interprets* the parent's reply | Deterministic, testable, safe for health data; works with any model, even without tool-calling | Less free-form chat; new behaviors need code |
| Reminder wording is scripted | Reminders must be exact and consistent | Less natural |
| Emergency detection is regex *before* the model | A model outage or misread can't hide a fall | Keyword list needs maintenance, and it will over-trigger sometimes (that's the right direction to err) |
| Every fact stores `source_utterance`; `validate` downgrades answers whose quote isn't in the parent's words | Prevents invented logs, the worst failure for trust | Cannot catch *semantic* mistakes ("my neighbour took her tablet"); only evals can |
| 'unclear' is a first-class result | Never guess; flag for human review | More "couldn't confirm" than a guessing system |
| LangGraph with in-memory checkpointer, thread = call | A call can't outlive its process anyway | Server restart mid-call loses that call's state |
| Append-only `events` table | Audit trail: what did the agent know and do | More rows |
| Idempotency key + state machine on call attempts | Duplicate triggers/webhooks must not double-dial or re-log | A bit more code than naive updates |

## Testing each layer (no phone needed until the end)

| Layer | How it is tested | Where |
|---|---|---|
| Conversation logic | Scripted-parent scenarios, deterministic checks + invariants | `carecall eval`, `tests/test_flows.py` |
| LLM paths | Stub client returning canned/invalid JSON | `tests/test_llm_paths.py` |
| Audio conversion | mu-law/PCM/resample round trips | `tests/test_audio.py` |
| **Twilio protocol** | A simulator plays Twilio's side of Media Streams (`start`, `media`, `mark`, `stop`) against the real server: opening audio, turn-taking, hangup, barge-in `clear`, silence, forged token, dropped stream, webhooks | `tests/test_twilio_bridge.py` |
| **Speech models (STT/TTS)** | Behind `STTProvider`/`TTSProvider` interfaces; tests use `ScriptedSTT`/`FakeTTS`. Real Gnani adapters are **stubs** until API docs are wired in | `telephony/stt.py`, `tts.py` |

The real-world ladder, in order (stop at the first rung that fails):

1. `carecall call-test`: Twilio rings your phone and reads a sentence (proves creds, from-number, verified number).
2. Tunnel (ngrok) + `carecall serve` + `carecall call --routine lunch_checkin`: webhook + signed TwiML + media stream connect (**needs the Gnani adapters**).
3. Gnani STT: transcripts printed live. Gnani TTS: you hear the greeting.
4. Full conversation; read per-turn latency in Langfuse (target: parent stops talking -> reply starts < ~1.5 s).
5. Interruption, silence, mumbling, background noise, hang-up mid-sentence.

## LLM-written test notes

After each `carecall eval`, every transcript gets a note (`runs/<id>/notes/<scenario>.md`) plus `summary.md`.
Deterministic checks decide pass/fail. The note explains: summary, coverage, a **log audit** (is each log backed by a
quote?), issues with turn numbers, suggested fixes, and a verdict box for you. With `LLM_PROVIDER=fake` a
template writer produces the note; with a real model `LLMNoteWriter` does, and code overrides any log verdict
whose quote is missing and any "pass" when a check failed. Notes are also saved to `run.db` (`test_notes`).
Read every failing and flagged transcript yourself; notes are an index, not a replacement.

Two scenarios tagged `llm_needed` fail on the keyword baseline **on purpose**
(`third_party_not_parent`, `tablet_later_means_no`). Run `carecall eval` with a real model and they should pass.

## Not done yet / needs you

- **Gnani adapters** (`llm/gnani.py`, `telephony/stt.py:GnaniSTT`, `telephony/tts.py:GnaniTTS`) need the API docs.
  If the reasoning model is OpenAI-compatible, set `LLM_PROVIDER=openai_compat` + base URL/key/model instead.
- Langfuse export and the real Twilio path have not been run end to end (Docker Desktop was off; no Twilio creds).
  The Langfuse instrumentation code paths were exercised against the real SDK; the Twilio protocol against a simulator.
- Scheduler, callbacks execution (callbacks are recorded, not yet dialed), multi-user auth, frontend.
- Phase 0 API has no auth: keep `serve` bound to localhost and expose only through the tunnel.
