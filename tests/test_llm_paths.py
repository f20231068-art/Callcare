"""The real-LLM code paths, driven by a stub client so no network or API key is needed."""
import json

from carecall.agent.interpret import InterpretRequest, LLMInterpreter
from carecall.agent.session import CallSession
from carecall.calls.service import create_call
from carecall.evals.notes import LLMNoteWriter
from carecall.evals.runner import load_scenarios, run_scenario
from carecall.llm.base import parse_json


class StubClient:
    model = "stub-1"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0)


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"a": 1} Hope that helps') == {"a": 1}


def test_llm_interpreter_parses_model_json():
    reply = json.dumps({"intent": "answer", "yes_no": False, "evidence": "I will take it after some time", "confidence": 0.9})
    it = LLMInterpreter(StubClient(reply))
    out = it.interpret(InterpretRequest(step_kind="yes_no", question="Have you taken it?",
                                        utterance="I will take it after some time"))
    assert out.intent == "answer" and out.yes_no is False


def test_llm_interpreter_garbage_becomes_unclear_not_a_crash():
    it = LLMInterpreter(StubClient("I am sorry, I cannot do that"))
    out = it.interpret(InterpretRequest(step_kind="yes_no", question="q", utterance="whatever"))
    assert out.intent == "unclear"


def test_llm_backed_session_solves_a_scenario_the_rules_cannot():
    reply = json.dumps({"intent": "answer", "yes_no": False, "evidence": "I will take it after some time", "confidence": 0.9})
    cid, _ = create_call("metformin_reminder")
    s = CallSession(cid, LLMInterpreter(StubClient(reply)))
    s.open()
    res = s.turn("I will take it after some time")
    assert res.end_reason == "callback"  # logged not_taken + 10 minute callback


def test_note_writer_cannot_approve_a_log_without_a_quote():
    sc = next(s for s in load_scenarios() if s["id"] == "metformin_taken")
    from carecall.agent.interpret import RuleInterpreter
    from carecall.evals.notes import TemplateNoteWriter
    rec = run_scenario(sc, RuleInterpreter(), TemplateNoteWriter())
    rec["medicine_logs"][0]["source_utterance"] = "something never said"
    lying = json.dumps({"summary": "all good", "overall": "pass", "issues": [],
                        "log_audit": [{"log": "medicine Metformin = taken", "evidence": "something never said", "verdict": "supported"}]})
    note = LLMNoteWriter(StubClient(lying)).write(rec)
    assert note.log_audit[0].verdict == "unsupported"
    assert "overridden by code" in note.log_audit[0].comment


def test_invalid_note_falls_back_to_template():
    sc = next(s for s in load_scenarios() if s["id"] == "metformin_taken")
    from carecall.agent.interpret import RuleInterpreter
    from carecall.evals.notes import TemplateNoteWriter
    rec = run_scenario(sc, RuleInterpreter(), TemplateNoteWriter())
    note = LLMNoteWriter(StubClient("not json at all")).write(rec)
    assert "template note used" in note.judgment_notes
