from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from .latent_archive import verify_latent_archive
from .replay_protocol import verify_replay_protocol
from .store import LivingStore, payload_digest


SCHEMA = "NOLANE-PERSONAL-NEXT-USER-CASES-V1"


@dataclass(slots=True)
class PersonalLanguageCase:
    event_id: str
    version: int
    split: str
    context_text: str
    target_text: str
    latent_before: list[float]
    history_event_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _role_line(kind: str, text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if kind == "user_message":
        return f"User: {text}"
    if kind == "assistant_speech":
        return f"Assistant: {text}"
    return None


def build_personal_language_cases(
    store: LivingStore,
    protocol: dict[str, Any],
    latent_archive: dict[str, Any],
    split: str,
    *,
    max_context_events: int = 8,
    max_target_chars: int = 2000,
) -> list[PersonalLanguageCase]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    verify_replay_protocol(store, protocol)
    verify_latent_archive(latent_archive, protocol)
    split_ids = {str(x["event_id"]) for x in protocol["splits"][split]}
    frozen_ids = {
        str(x["event_id"])
        for name in ("train", "dev", "test")
        for x in protocol["splits"][name]
    }
    latent_by_event = {str(x["event_id"]): x for x in latent_archive["entries"]}
    history: deque[tuple[str, str]] = deque(maxlen=max(1, int(max_context_events)))
    cases: list[PersonalLanguageCase] = []

    for record in store.replay_records():
        event = record["event"]
        if event.event_id not in frozen_ids:
            continue
        text = str(event.payload.get("text", "")).strip()
        if event.event_id in split_ids and event.kind == "user_message" and text:
            history_ids = [event_id for event_id, _line in history]
            history_lines = [line for _event_id, line in history]
            prefix = "\n".join(history_lines)
            if prefix:
                prefix += "\n"
            prefix += "User: "
            latent_entry = latent_by_event.get(event.event_id)
            if latent_entry is None:
                raise ValueError(f"missing latent for event {event.event_id}")
            cases.append(
                PersonalLanguageCase(
                    event_id=event.event_id,
                    version=int(record["version"]),
                    split=split,
                    context_text=prefix,
                    target_text=text[: max(1, int(max_target_chars))] + "\n",
                    latent_before=[float(x) for x in latent_entry["latent_before"]],
                    history_event_ids=history_ids,
                )
            )

        line = _role_line(event.kind, text)
        if line is not None:
            history.append((event.event_id, line))

    return cases


def cases_manifest(
    cases: list[PersonalLanguageCase],
    *,
    protocol_sha256: str,
    latent_archive_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "protocol_sha256": protocol_sha256,
        "latent_archive_sha256": latent_archive_sha256,
        "count": len(cases),
        "cases": [case.to_dict() for case in cases],
    }
    payload["cases_sha256"] = payload_digest(payload)
    return payload
