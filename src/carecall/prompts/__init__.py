"""Prompt registry: one shared base + one prompt per use case, kept as versioned markdown files.

A prompt's `version` is a short hash of its full text, recorded in every trace, so you can always tell
which exact prompt produced a conversation (and diff two runs after you edit a file)."""
import hashlib
from functools import lru_cache
from pathlib import Path

USECASES = ("meal_discovery", "meal_checkin", "medicine_before_food", "medicine_after_food", "defined_reminder")
_DIR = Path(__file__).parent


@lru_cache
def system_prompt(usecase: str) -> tuple[str, str]:
    """Returns (prompt_text, version). Unknown use cases get the base prompt only."""
    base = (_DIR / "base.md").read_text(encoding="utf-8")
    extra_file = _DIR / f"{usecase}.md"
    extra = extra_file.read_text(encoding="utf-8") if extra_file.exists() else ""
    text = base.strip() + ("\n\n" + extra.strip() if extra else "")
    return text, hashlib.sha1(text.encode()).hexdigest()[:8]


def clear_cache():
    system_prompt.cache_clear()
