from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .store import LivingStore, payload_digest


SCHEMA = "NOLANE-REPLAY-PROTOCOL-V1"


@dataclass(slots=True)
class ReplaySplitPolicy:
    train_fraction: float = 0.70
    dev_fraction: float = 0.15
    min_records: int = 6

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0,1)")
        if not 0.0 < self.dev_fraction < 1.0:
            raise ValueError("dev_fraction must be in (0,1)")
        if self.train_fraction + self.dev_fraction >= 1.0:
            raise ValueError("train+dev fractions must leave room for test")
        if self.min_records < 3:
            raise ValueError("min_records must be >= 3")


def _entry(record: dict[str, object]) -> dict[str, Any]:
    before = record["before"]
    after = record["after"]
    event = record["event"]
    return {
        "version": int(record["version"]),
        "event_id": event.event_id,
        "event_kind": event.kind,
        "before_digest": payload_digest(before.to_dict()),
        "after_digest": payload_digest(after.to_dict()),
    }


def _split_counts(n: int, policy: ReplaySplitPolicy) -> tuple[int, int, int]:
    train = max(1, int(n * policy.train_fraction))
    dev = max(1, int(n * policy.dev_fraction))
    if train + dev >= n:
        dev = 1
        train = n - 2
    test = n - train - dev
    if test < 1:
        raise ValueError("not enough records for non-empty train/dev/test")
    return train, dev, test


def build_replay_protocol(store: LivingStore, policy: ReplaySplitPolicy | None = None) -> dict[str, Any]:
    policy = policy or ReplaySplitPolicy()
    policy.validate()
    records = store.replay_records()
    if len(records) < policy.min_records:
        raise ValueError(f"need at least {policy.min_records} replay records")
    train_n, dev_n, _test_n = _split_counts(len(records), policy)
    entries = [_entry(record) for record in records]
    train = entries[:train_n]
    dev = entries[train_n : train_n + dev_n]
    test = entries[train_n + dev_n :]

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "policy": asdict(policy),
        "identity_id": records[0]["before"].identity_id,
        "frozen_max_version": entries[-1]["version"],
        "frozen_tail_digest": entries[-1]["after_digest"],
        "counts": {"total": len(entries), "train": len(train), "dev": len(dev), "test": len(test)},
        "splits": {"train": train, "dev": dev, "test": test},
    }
    protocol["protocol_sha256"] = payload_digest(protocol)
    return protocol


def verify_replay_protocol(store: LivingStore, protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported replay protocol schema")
    supplied_sha = protocol.get("protocol_sha256")
    body = dict(protocol)
    body.pop("protocol_sha256", None)
    expected_sha = payload_digest(body)
    if supplied_sha != expected_sha:
        raise ValueError("replay protocol digest mismatch")

    records = store.replay_records()
    by_version = {int(r["version"]): r for r in records}
    frozen_entries = (
        list(protocol["splits"]["train"])
        + list(protocol["splits"]["dev"])
        + list(protocol["splits"]["test"])
    )
    if not frozen_entries:
        raise ValueError("empty replay protocol")

    first = by_version.get(int(frozen_entries[0]["version"]))
    if first is None or first["before"].identity_id != protocol.get("identity_id"):
        raise ValueError("identity mismatch")

    for frozen in frozen_entries:
        version = int(frozen["version"])
        current = by_version.get(version)
        if current is None:
            raise ValueError(f"missing frozen replay version {version}")
        actual = _entry(current)
        for key in ("event_id", "event_kind", "before_digest", "after_digest"):
            if actual[key] != frozen[key]:
                raise ValueError(f"frozen replay drift at version {version}: {key}")

    tail = by_version[int(protocol["frozen_max_version"])]
    if _entry(tail)["after_digest"] != protocol.get("frozen_tail_digest"):
        raise ValueError("frozen replay tail mismatch")


def records_for_split(store: LivingStore, protocol: dict[str, Any], split: str) -> list[dict[str, object]]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    verify_replay_protocol(store, protocol)
    wanted = {item["event_id"] for item in protocol["splits"][split]}
    return [record for record in store.replay_records() if record["event"].event_id in wanted]
