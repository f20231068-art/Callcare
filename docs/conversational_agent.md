# The conversational agent (replaces the scripted question flow)

## What it is now
The model holds the conversation. It sees the **whole conversation every turn** (memory), writes every word it says, and chooses **tools** itself (log a meal, book a call back, set a reminder, raise an alert, end the call). A tool that refuses is information the model reacts to ("4 won't work because your next reminder is at 3:30; I can do 2 or 3" -> parent: "2pm" -> it tries again).

```
parent speaks -> session.turn(text)  -> graph.invoke({"utterance": text})
   safety   appends the parent's words to the memory; emergency regex may pre-empt the model
   agent    ONE model call: {"say": text|null, "actions": [{"tool","args"}], "end_call": bool}
   tools    runs the actions; logging is fire-and-forget; bookings, lookups and any FAILURE go back to the model
   agent    (again, up to 4 steps)         -> then speaks
   finish   caps/filters the words, assembles the reply, decides if the call is over
```
Code: `agent/converse.py` (graph, prompt, memory), `agent/db_tools.py` (the tools), `agent/session.py` (`mode="converse"` is the default; passing an interpreter selects the old graph).

## What changed
**Added**
- `agent/converse.py`: the tool-calling conversation graph (nodes safety, agent, tools, finish, escalate, give_up), the system prompt built from the call context, memory = the full message history (also persisted as `transcript_turns`), a ring-back prompt that includes the earlier attempt's transcript.
- Tools the model calls directly (in the registry `agent/db_tools.py`): `log_meal`, `log_medicine`, `ack_reminder`, `raise_alert`, `relay_message` (fire-and-forget), plus the existing `book_callback`, `add_reminder`, `get_upcoming_calls`, `get_today_status`. The tool list in the prompt is generated from the registry.
- Tool refusals now explain themselves: a `reason_code` and alternatives (`latest_possible_local`, `next_reminder_at_local`) so the agent can offer something that works.
- `book_callback` writes the `Callback` row itself (it used to be a second step in the graph).
- `llm/openai_compat.chat_json` (JSON-mode chat with history); `config.agent_mode` (`converse` default, `graph` for the old agent); `CallSession(mode=..., llm=...)`; `carecall personas --agent converse`.
- Tests: `tests/test_converse.py` (14 tests, scripted model): memory, one-call logging, quote refusal, de-duplication, negotiation (4 hours refused -> 1 hour accepted), live fillers, emergency pre-empts the model, three silences, forbidden tools, garbage output, caps and leaks, outcome engine keeps the live booking, ring-back prompt.

**Removed from the conversation path (the old graph still exists under `mode="graph"` for comparison)**
- The fixed question list as the thing that drives the dialogue (the opening question is still the routine's exact wording).
- The interpret -> validate -> act pipeline: no per-turn intent classification, no hedge/plausibility/filler-item code rules, no canned replies for off-topic and no strike counter in code.

**Kept in code on purpose (guarantees, not interpretation)**
- Emergency regex pre-check: it can end the model's turn.
- Tool rules: quiet hours, attempt cap, "next reminder first", evidence must be a quote of the parent's real words in this call, no duplicate logs, meal times cannot be changed by the agent.
- Reply length cap, leak filter (instructions, numbers, e-mails), 24-turn limit, three silent turns end the call (and still leave a record).

## Honest costs of this design
- Less predictable: the model may skip a step or word things differently. The persona suite measures it (see below).
- Latency: a tool-using turn can take two model calls (the filler covers the gap on a phone line).
- Guardrails that used to be code are now prompt-driven and judged by an LLM rubric.
