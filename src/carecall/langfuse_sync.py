"""`carecall langfuse-sync`: put the things you will want to browse into Langfuse.

  Prompts  : one text prompt per use case (interpret/<usecase>), labelled 'latest', so you can read and diff them in the UI.
  Dataset  : 'carecall-personas', one item per persona (input = the persona card, expected_output = what a correct agent records).
Idempotent: prompts become new versions only when text changed; dataset items are keyed by persona id."""
from . import tracing
from .evals.personas import load_personas
from .prompts import USECASES, system_prompt

DATASET = "carecall-personas"


def sync() -> list[str]:
    log = []
    if not tracing.init_tracing():
        return ["Langfuse not reachable: start `docker compose up -d` in docker/langfuse and check .env keys"]
    client = tracing._client
    for uc in USECASES:
        text, version = system_prompt(uc)
        try:
            client.create_prompt(name=f"interpret/{uc}", prompt=text, labels=["latest"], type="text",
                                 config={"content_hash": version}, commit_message=f"content hash {version}")
            log.append(f"prompt interpret/{uc} (hash {version})")
        except Exception as e:
            log.append(f"prompt interpret/{uc} FAILED: {e}")
    try:
        client.create_dataset(name=DATASET, description="Simulated parents (5 per use case) with expected agent outcomes")
    except Exception as e:  # already exists is fine
        log.append(f"dataset create: {str(e)[:120]}")
    for card in load_personas():
        try:
            client.create_dataset_item(
                dataset_name=DATASET, id=card["id"],
                input={k: card[k] for k in ("name", "profile", "traits", "truth", "usecase", "routine")},
                expected_output=card.get("expect", {}), metadata={"usecase": card["usecase"]})
        except Exception as e:
            log.append(f"item {card['id']} FAILED: {str(e)[:120]}")
    log.append(f"dataset {DATASET}: {len(load_personas())} persona items synced")
    tracing.flush()
    return log
