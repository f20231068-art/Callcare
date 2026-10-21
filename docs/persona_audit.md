# Persona audit log

Method: 25 personas (5 per use case) each have a card (profile, traits, ground **truth**) and an **expected outcome**.
A persona talks to the agent; deterministic checks compare what was recorded against the expectation; every
transcript is also read by a person. Pass/fail alone is not the audit: several defects below were in conversations
that "passed".

**Caveat on what these runs prove.** Rounds 1-2 ran *scripted* personas against the *offline rule interpreter*, because no
Gnani reasoning-model endpoint was available yet. They validate the graph, guards, alerts, wording and harness. They do
**not** measure how well the real model understands speech. Run `carecall personas --mode llm` once an LLM is configured.

## Round 1 -> 2: 21/25 -> 24/25 (remaining failure needs a real model: `ma_hinglish_shanta`)

| # | Found in | Severity | What the audit saw | Fix |
|---|---|---|---|---|
| 1 | mb_unsure_which_tablet_sunil | **safety** | "Which tablet is this? ... I took the white one" logged as Metformin **taken** | Hedge/appearance guard in `validate` (code, applies to any model); agent now says the tablet name and dose and asks to check the strip; unresolved -> `unclear` + warning to family |
| 2 | mc_forgetful_dwarka | **safety** | "Maybe. I am not sure" heard as **yes**; then "Hmm." logged as a food item | Hedged yes can never be yes; filler words rejected as foods |
| 3 | mc_forgetful_dwarka | UX + signal | Honest "I do not remember" treated as "did not catch that", re-asked 3 times | New `unsure` intent: recorded as unconfirmed, no re-asking, family told (possible early memory-loss signal) |
| 4 | dr_conflict_sudha | UX | Clear reply re-asked 3 times with identical words | Stuck detector: same words twice -> stop and hand the raw words to the family |
| 5 | mb_refuses_kishore | notification fatigue | One event raised two alerts | All alerts in a turn merged into one |
| 6 | md_skeptical_meena | trust | Suspicious parent got only a first name | Identity reply: automated assistant, who it is for, "never asks for money or bank details" |
| 7 | md_* | voice | Agent said "09:00" (TTS may read digits) | `spoken_time()`: "9 in the morning" |
| 8 | md_* | data quality | Meal times were never read back; they later drive medicine reminders | Read-back confirmation step; "no" flags the family |
| 9 | ma_double_dose_lalitha | safety | "Took it already, shall I take another?" got a generic reply | Referral to doctor/pharmacist, "check before taking any more", alert reason says *possible extra dose* |
| 10 | md_gopal | accessibility | No adaptation when asked to repeat | Speaking-rate hint 0.85 for the rest of the call (passed to TTS `speed`) |
| 11 | all | audit | Summaries showed reasons but not the parent's words | Alert text includes the parent's quote |
| 12 | persona harness | test fidelity | Agent greeted every persona as "Asha" | Seeded parent takes the persona's name |

## Still open (decisions or capabilities, not bugs)

- **Language mirroring.** A Hindi/Hinglish speaker is re-asked in English. Needs language detection and TTS `language`
  switching (Gnani supports it; see `docs.gnani.ai` Language Switch).
- **Real-model behaviour.** Whether the Gnani reasoning model follows the per-use-case prompts (JSON validity, quoting
  verbatim, not over-triggering `unsure`/`medical_question`) is unmeasured.
- **Product decision: the double-dose line.** "Please check with them before taking any more" is deliberately minimal
  safety guidance, not medical advice. A clinician or the product owner should approve the exact wording.
- **Over-triggering.** `dizzy`, `pain`, `weak`, `confused` raise a *warning* but keep the call going; only severe words
  (fall, chest pain, breathless...) end the call. Tune with real calls.
- **STT/TTS quality** (accents, noise, endpointing, latency) is not covered by text personas. See the speech-eval plan.

## How to re-run

```
carecall personas                          # scripted, deterministic (CI runs this: only NEEDS_LLM personas may fail)
carecall personas --mode llm               # an LLM plays each persona (needs LLM_* in .env)
carecall personas --usecase meal_checkin --persona mc_forgetful_dwarka
carecall personas --agent rules            # force the offline baseline even when an LLM is configured
```
Open `runs/<id>/persona_report.md` first, then the per-persona notes. In Langfuse filter by tag `suite:personas`,
`usecase:<x>`, `run:<id>`, or `persona-mode:<m>`; each trace has a `scenario_card` span (who and what was expected), the
turns with every LLM generation, and an `audit` span (checks + note), plus scores.

---

# Live rounds: a real model plays every parent (OpenRouter `openai/gpt-6-luna`)

Setup: the persona is generated turn by turn by the model from its card + hidden truth (temperature 0.7, free text,
not JSON mode); the agent's interpreter and the note writer are also the live model. No line is scripted.
`carecall personas --mode llm --workers 6`.

| Round | Result | What the transcripts showed |
|---|---|---|
| L1 | 23/25 | see below |
| L2 | 24/25 | after fixes 13-19 |
| L3 | 25/25 | after fixes 20-23; 57 interpretations, 0 model errors |

| # | Found by | What the live audit saw | Fix |
|---|---|---|---|
| 13 | ma_compliant_vimala, mc_mild_symptom_hari | Clear answers answered with "Sorry, I did not catch that". Root cause: the model returns `"reason": null`, my schema rejected the whole reply | Null fields are treated as "not provided" |
| 14 | (audit gap) | I could not tell *why* a turn was re-asked | Every interpretation (utterance, intent, evidence, confidence, error, raw reply) is recorded as an event and shown in the report ("Model-call health") |
| 15 | robustness | A provider outage would fail the turn | `LLMInterpreter` falls back to the keyword rules and records `fallback: rules` |
| 16 | mc_happy_lakshmi, mc_mild_symptom_hari | "Yes, rice and dal" still got "What did you have?" | Items stated with the yes are logged directly |
| 17 | mc_mild_symptom_hari | "I'm sorry to hear that... Good to hear." | Generic acknowledgement dropped after a sympathy line |
| 18 | md_chatty_savitri | Nostalgic remarks became 3 alerts | Prompt: concern only if actionable; alerts de-duplicated by theme (loneliness, appetite) |
| 19 | dr_asks_details_bhavani | Parent asked for the exact time, which the reminder does not hold; the same text was read 3 times | After one re-read, the agent says it only has what it read out, acknowledges, and alerts the family to supply details |
| 20 | mb_forgot_then_takes_geeta | "I forgot, I'll go get it now" logged *not taken* + 10 min callback | Prompt: acting-now statements are `wait`; the agent waits and logs what she reports next |
| 21 | dr_sudha | Reply "Thank you." ignored "I can't make it" | Acknowledges and says it will tell the family so it can be rescheduled |
| 22 | md_hard_of_hearing_gopal | "Eh? What was that? About half past seven!" classified as *repeat*; looped 14 turns | An answer inside a repeat request wins (prompt rule + code backstop for times) |
| 23 | dr_repeats_anand | "I think I've got it now" downgraded as a hedge | Hedge words apply to medicine/meal questions, not to "did you hear that" |

**What this does and does not show.** One live run per round; the model is stochastic, so a single 25/25 is evidence, not a
guarantee. Re-run a few times (`carecall personas --mode llm`) and compare. The personas are LLM-written approximations of
elderly speech, not real elderly speech; real STT output (accents, noise, no punctuation) will be messier.

---

# Model-written agent (no templates): `carecall direct-personas`

The agent's replies are written by a model from the per-use-case conversation prompt (`prompts/gnani/*`); the first line is
fixed. **Agent model in these runs: `openai/gpt-6-luna` as a stand-in** (Gnani's model is not reachable directly yet:
set `AGENT_LLM_*` to test it). Parent persona and judge: GPT-6 Luna. Text only; judged from the transcript.

| Run | Result | What changed |
|---|---|---|
| D1 | 17/25 | first run; most failures were the **judge** being wrong (see below) |
| D2 | 19/25 | judge calibrated (agent brief + allowed commitments, 24h-only clock check, transcript-verifiable rubric lines) |
| D3 | 24/25 | 4 real prompt fixes (below) + one more judge fix |

Judge false positives found by reading the failures (a judge needs its own audit):
- The fixed opening ("take it now, about 15 minutes before your lunch", "Metformin 500mg") was flagged as advice/invention.
- "I will let Sree know" / "Sree is your family member" flagged as invention (the prompt prescribes them).
- A safe referral ("check with your doctor or pharmacist before taking any more") flagged as advice.
- "8:30 in the evening" flagged as clock digits (only 24h forms like "09:00" are a speech problem).

Real agent-prompt weaknesses found, and fixed in `prompts/gnani/*` (re-exported by `carecall gnani-export`):
- After re-reading a reminder the model closed the call without waiting for "okay" -> must ask "Did you get that?" and wait.
- The silence rule produced two questions in one turn -> say only "Are you still there?" and wait; tell the family if silence persists.
- The model ignored what the parent had just said (what Hari ate, that Usha had already eaten) -> always acknowledge it first.
- Closing straight after a repeat request.

Still open: `dr_asks_details_bhavani`. The model handled it sensibly (admitted it only had "tomorrow evening", offered to ask Sree, got
a "yes, please"), but the persona never says "got it", so the strict `reminder_ack_true` line fails. That is an expectation/judge
ambiguity, not clearly an agent fault. Decide whether "asked Sree for the exact time" counts as acknowledgement.

Caveat: one run per round, stochastic models, LLM-written parents. Treat 24/25 as "the harness and prompts are in good shape",
not as a measured pass rate for Gnani's model.

---

# Switching to a fast, non-thinking model (qwen/qwen3.7-flash)

Benchmarked 15 OpenRouter models (`carecall model-bench`): thinking off + latency-sorted routing. Quality gate = the 17 scripted
scenarios through the model as interpreter. Qwen3.7-flash vs Qwen3.5-9b (15 interleaved trials): reply p50 1.00 s vs 1.46 s,
first word 0.60 s vs 0.90 s, interpretation 1.59 s vs 1.83 s, both 17/17 and 15/15 valid JSON; ~$0.003 vs ~$0.010 per 100 turns.
Avoid: `ling-3.1-flash` (invalid JSON, 14 s), `nex-n2.5-mini` (14.7 s tail), `glm-5.3-flash` (cannot disable thinking).

| Setup | Typical turn (STT + model + TTS) | LangGraph agent, 25 personas | Model-written agent, 25 personas |
|---|---|---|---|
| gpt-6-luna, thinking on | ~3.5 s | 24/25 | 24/25 |
| qwen3.7-flash, thinking off | ~2.5 s (model-written) / ~3.3 s (LangGraph); ~1.2 s with streaming | **25/25** | 20/25 |

What the cheaper model exposed (all fixed, with tests):
- My "answer inside a repeat request" backstop read "say that **one** more time" as 1 o'clock -> dinner recorded as 13:00.
  Now only explicit clock phrases count, and a meal time outside a plausible window is rejected whatever the model says.
- "I hope Sree can **help me** remember" tripped the emergency check (a false alarm that would end the call and alert the family).
- The agent thanked twice ("Thank you, ..." then "Thank you X. Take care."). The closing no longer starts with "Thank you".

Reading: the LangGraph agent absorbs a weaker model well because code (not the model) owns the flow, the quote guard, the
hedge guard and the time plausibility guard. The model-written agent depends on the model following a long prompt, and the
small model is visibly less reliable there (over-applies the sympathy rule to a hard-of-hearing caller, forgets the read-back,
once claimed the parent "already ate"). If you want the model-written route, use a stronger fast model (claude-haiku-5.5 or
gemini-3.1-flash-lite both scored 17/17 on understanding) and re-run `carecall direct-personas`.
