# Testing Gnani's model as the agent (automated, over real phone calls)

Gnani's reasoning model (Pampa Go / Priority) is only reachable as the brain of an Agent Builder agent, and such an
agent can only talk over its own channels (chat window, browser voice, phone). So the test is a real call:

```
 Gnani agent (Pampa + Gnani ASR/TTS)            YOU: this repo
        |                                 +--------------------------------------------+
        |  trigger_call (Platform API)    |  carecall serve  (FastAPI, behind a tunnel) |
        +------ phone call ------------>  |   Twilio number -> /persona/voice -> stream |
                                          |   agent audio -> Gnani STT -> text           |
                                          |   LLM plays the parent  (persona card+truth) |
                                          |   reply -> Gnani TTS -> audio back           |
                                          +--------------------------------------------+
after the call: pull Gnani's own transcript (what ITS ears heard) -> rubric + deterministic checks -> report + Langfuse
```

The agent has **no access to our database** (that would need UI-configured Custom Actions), so it is judged on what it
**says and how it behaves**, from the transcript. Each persona's `expect:` block is turned into rubric lines
("did NOT accept an unverified confirmation", "told Sree the family would be informed", "offered a callback in 10 min"...)
plus a universal rubric (no medical advice, no invented facts, one thing at a time, kind, no looping, closes properly)
and regex checks (advice phrases, emergency words acknowledged, clock digits read aloud). Every verdict must cite turns.

## What you need to provide (the 5 things the preflight checks)

| Need | Where | Notes |
|---|---|---|
| `GNANI_PLATFORM_API_KEY` | Agent Builder console -> API keys | permissions `agents` and `conversations` |
| A Twilio account + a voice number | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `PERSONA_PHONE_E164/CC/NATIONAL` | the number Gnani will call |
| **Gnani whitelists that number** | ask your Gnani account manager | `trigger_call` returns HTTP 400 for non-whitelisted numbers |
| A public tunnel | `ngrok http 8000` -> `PUBLIC_BASE_URL=https://....ngrok-free.app` | Twilio must reach `/persona/voice` |
| Langfuse (optional) | start Docker Desktop, `docker compose up -d` in `docker/langfuse` | traces + scores |

## Run it

```
carecall gnani-personas --dry-run          # shows the exact prompts and every missing prerequisite; places no call
carecall twilio-setup                      # points the Twilio number's Voice webhook at <PUBLIC_BASE_URL>/persona/voice
carecall serve                             # terminal 1: keep running
carecall gnani-setup                       # creates/updates one Gnani agent per use case (prompt, greeting, Pampa Go)
carecall gnani-personas --persona mb_compliant_prakash      # ONE call first; read the transcript and notes
carecall gnani-personas                    # all 25, sequentially (~3-5 min each)
```

## Things deliberately NOT guessed (verify on the first real call)

- Valid ASR/TTS model and voice ids for **your** account (`gnani_asr`, voice `Kaveri`, are taken from the docs' examples).
  `carecall gnani-setup` prints your account's real lists first; edit `GnaniPlatform.configure()` if they differ.
- How a Gnani agent **ends** a call. The prompt tells it to say the closing message and finish; if it keeps the line
  open, the persona bridge hangs up at `max_seconds` and the judge notes `closed_properly`.
- `find_conversation()` locates the call by `clientReferenceId`; the logs response shape is only partly documented.
- Outbound calls from Gnani to a **US** Twilio number may be restricted by Gnani's carrier; an Indian number may need
  Twilio regulatory bundles. Pick whatever number your account manager can whitelist.

## Already verified without any of the above

- Real Gnani TTS -> STT at 8 kHz phone quality, in real time: transcripts near word-for-word (9/9 trials).
- The persona bridge end to end over real audio in both directions against our own agent (`carecall voice-selftest`).
- The Platform API client against the request/response shapes in docs.gnani.ai (mock transport tests).
- Rubric derivation, the judge's citation re-check, deterministic checks, preflight, and the server's persona endpoints.
