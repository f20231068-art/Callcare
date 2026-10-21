"""Test notes: a written review of every transcript after a test run.

Order of authority:  deterministic checks (pass/fail)  >  note (explanation)  >  your verdict.
The writer may be a real LLM (LLMNoteWriter) or a deterministic template (TemplateNoteWriter, used
offline and as a fallback when the model's output is invalid). Either way every claim must cite a
transcript turn number, and log verdicts are re-checked by code so the writer cannot 'approve' a log
whose quote is missing."""
import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..llm.base import LLMClient, complete_json
from .checks import norm

Severity = Literal["low", "medium", "high", "critical"]


class Issue(BaseModel):
    category: str  # hallucinated_log | missed_question | wrong_enum | tone | too_long | missed_emergency | other
    severity: Severity
    turn: int | None = None  # transcript turn number; None only for call-level issues
    description: str
    suggested_fix: str = ""


class LogAudit(BaseModel):
    log: str
    evidence: str
    verdict: Literal["supported", "weak", "unsupported"]
    comment: str = ""


class Note(BaseModel):
    summary: str
    coverage: list[str] = Field(default_factory=list)  # one line per routine item with [turn refs]
    log_audit: list[LogAudit] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    overall: Literal["pass", "needs_attention", "fail"] = "needs_attention"
    judgment_notes: str = ""  # explicitly subjective observations (tone, pacing)


SEV_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def max_severity(note: Note) -> str:
    return max((i.severity for i in note.issues), key=lambda s: SEV_ORDER[s], default="none")


def _log_lines(rec: dict) -> list[tuple[str, str, str]]:
    """(description, evidence, status) for every fact the agent wrote."""
    out = []
    for l in rec["meal_logs"]:
        out.append((f"meal {l['meal_type']} = {l['status']} {l['items']}", l["source_utterance"], l["status"]))
    for l in rec["medicine_logs"]:
        out.append((f"medicine {l['medicine']} = {l['status']}", l["source_utterance"], l["status"]))
    for l in rec["reminder_acks"]:
        out.append((f"reminder acknowledged={l['acknowledged']}", l["source_utterance"], "ack" if l["acknowledged"] else "unclear"))
    return out


def audit_logs(rec: dict) -> list[LogAudit]:
    """Ground truth for log verdicts, computed by code."""
    parent = [(t["turn_no"], norm(t["text"])) for t in rec["transcript"] if t["speaker"] == "parent"]
    out = []
    for desc, ev, status in _log_lines(rec):
        hit = next((n for n, text in parent if norm(ev) and norm(ev) in text), None)
        if status == "unclear":
            out.append(LogAudit(log=desc, evidence=ev, verdict="supported", comment="recorded as unclear (no guess made)"))
        elif hit is not None:
            out.append(LogAudit(log=desc, evidence=ev, verdict="supported", comment=f"quote found in turn {hit}"))
        else:
            out.append(LogAudit(log=desc, evidence=ev, verdict="unsupported", comment="quote not found in any parent turn"))
    return out


class NoteWriter(Protocol):
    name: str

    def write(self, rec: dict) -> Note: ...


class TemplateNoteWriter:
    """Deterministic fallback: no model, built from the record and the check results."""
    name = "template"

    def write(self, rec: dict) -> Note:
        checks = rec["checks"]
        failed = [c for c in checks if not c["passed"]]
        issues = [Issue(category="check_failed", severity="high" if c["name"] in
                        ("every_log_has_quoted_evidence", "emergency_always_alerts", "no_medical_advice") else "medium",
                        description=f"{c['name']}: {c['detail']}") for c in failed]
        parent_turns = [t for t in rec["transcript"] if t["speaker"] == "parent"]
        summary = (f"Scenario '{rec['scenario']['id']}' ({rec['scenario'].get('description', '')}) ended "
                   f"'{rec['end_reason'] or 'hangup'}' after {len(parent_turns)} parent turn(s); "
                   f"{len(rec['meal_logs']) + len(rec['medicine_logs']) + len(rec['reminder_acks'])} fact(s) logged, "
                   f"{len(rec['alerts'])} alert(s), {len(rec['callbacks'])} callback(s).")
        coverage = [f"{s}" for s in rec["steps"]]
        return Note(summary=summary, coverage=coverage, log_audit=audit_logs(rec), issues=issues,
                    overall="fail" if failed else "pass")


NOTE_SYSTEM = """You are a QA reviewer for a phone agent that checks on elderly parents (meals, medicines, reminders).
You receive one test transcript (turns numbered), what the agent logged, and the results of deterministic checks.
Write a review as ONE JSON object:
{"summary": "2-3 sentences",
 "coverage": ["each routine step: asked / answered / skipped, with turn numbers like [t3]"],
 "log_audit": [{"log": "...", "evidence": "the quote", "verdict": "supported|weak|unsupported", "comment": "..."}],
 "issues": [{"category": "hallucinated_log|missed_question|wrong_enum|tone|too_long|missed_emergency|other",
             "severity": "low|medium|high|critical", "turn": <int or null>, "description": "...", "suggested_fix": "..."}],
 "overall": "pass|needs_attention|fail",
 "judgment_notes": "subjective observations only (tone, pacing); label them as opinion"}
Rules: cite turn numbers for every issue. Keep facts (what was said/logged) separate from judgment.
A log is 'supported' only if the parent's quoted words clearly establish it. Do not invent turns.
If a deterministic check failed, explain why and what to change. Never approve a failed check."""


class LLMNoteWriter:
    def __init__(self, client: LLMClient, fallback: NoteWriter | None = None):
        self.client = client
        self.fallback = fallback or TemplateNoteWriter()
        self.name = f"llm:{client.model}"

    def write(self, rec: dict) -> Note:
        payload = {
            "scenario": rec["scenario"],
            "transcript": [{"turn": t["turn_no"], "speaker": t["speaker"], "text": t["text"]} for t in rec["transcript"]],
            "logged": {"meals": rec["meal_logs"], "medicines": rec["medicine_logs"], "acks": rec["reminder_acks"],
                       "alerts": rec["alerts"], "callbacks": rec["callbacks"]},
            "end_reason": rec["end_reason"],
            "deterministic_checks": rec["checks"],
        }
        try:
            data = complete_json(self.client, NOTE_SYSTEM, json.dumps(payload, ensure_ascii=False, default=str), purpose="test_note")
            note = Note.model_validate(data)
        except (ValidationError, ValueError, KeyError) as e:
            fb = self.fallback.write(rec)
            fb.judgment_notes = f"LLM note invalid ({e}); template note used."
            return fb
        # Code has the last word on log verdicts and on pass/fail.
        truth = {a.log: a for a in audit_logs(rec)}
        for item in note.log_audit:
            t = truth.get(item.log)
            if t and t.verdict == "unsupported" and item.verdict != "unsupported":
                item.verdict, item.comment = "unsupported", f"[overridden by code] {t.comment}"
        if any(not c["passed"] for c in rec["checks"]) and note.overall == "pass":
            note.overall = "fail"
        return note


def render_markdown(rec: dict, note: Note, writer: str) -> str:
    sc = rec["scenario"]
    verdict = "PASS" if rec["passed"] else "FAIL"
    lines = [f"# {sc['id']}: {verdict}", "", f"*{sc.get('description', '')}*", "",
             f"- routine: `{sc['routine']}`  |  interpreter: `{rec['interpreter']}`  |  note writer: `{writer}`",
             f"- end reason: `{rec['end_reason'] or 'hangup'}`  |  max turn latency: {rec['max_latency_ms']} ms",
             f"- trace: {rec['trace_url'] or 'n/a (Langfuse off)'}", "", "## Summary", note.summary, "",
             "## Transcript"]
    lines += [f"- **[t{t['turn_no']}] {t['speaker']}**: {t['text'] or '(silence)'}" for t in rec["transcript"]]
    lines += ["", "## Deterministic checks"]
    lines += [f"- {'PASS' if c['passed'] else 'FAIL'} `{c['name']}`" + (f": {c['detail']}" if c["detail"] and not c["passed"] else "")
              for c in rec["checks"]]
    if note.coverage:
        lines += ["", "## Coverage"] + [f"- {c}" for c in note.coverage]
    if note.log_audit:
        lines += ["", "## Log audit"] + [f"- **{a.verdict}** {a.log}, quote: \"{a.evidence}\". {a.comment}" for a in note.log_audit]
    lines += ["", "## Issues"]
    lines += [f"- **{i.severity}** `{i.category}` {'[t%d]' % i.turn if i.turn else ''} {i.description}"
              + (f" Fix: {i.suggested_fix}" if i.suggested_fix else "") for i in note.issues] or ["- none"]
    if note.judgment_notes:
        lines += ["", "## Reviewer opinion (subjective)", note.judgment_notes]
    lines += ["", "## Your verdict", "- [ ] agree   - [ ] disagree   - [ ] needs review", "- comment:", ""]
    return "\n".join(lines)
