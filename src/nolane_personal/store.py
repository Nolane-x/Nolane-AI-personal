from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .events import LivingEvent
from .memory import MemoryRecord
from .consolidation import MemoryLink
from .state import LivingState


def canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class LivingStore:
    """Local-first event, state snapshot, and memory store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self.db.close()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                at TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                salience REAL NOT NULL,
                payload_json TEXT NOT NULL,
                after_state_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                version INTEGER PRIMARY KEY,
                at TEXT NOT NULL,
                event_id TEXT,
                state_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            );
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                salience REAL NOT NULL,
                confidence REAL NOT NULL,
                source_event_id TEXT,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
            CREATE TABLE IF NOT EXISTS memory_links (
                parent_memory_id TEXT NOT NULL,
                child_memory_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(parent_memory_id, child_memory_id, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_version ON snapshots(version);
            CREATE INDEX IF NOT EXISTS idx_memory_links_parent ON memory_links(parent_memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_links_child ON memory_links(child_memory_id);
            """
        )
        self.db.commit()

    def load_state(self) -> LivingState | None:
        row = self.db.execute("SELECT state_json FROM snapshots ORDER BY version DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return LivingState.from_dict(json.loads(row["state_json"]))

    def initialize(self, state: LivingState) -> LivingState:
        if self.load_state() is not None:
            raise RuntimeError("living state already initialized")
        state.version = 0
        payload = state.to_dict()
        digest = payload_digest(payload)
        with self.db:
            self.db.execute(
                "INSERT INTO snapshots(version, at, event_id, state_json, digest) VALUES(?,?,?,?,?)",
                (0, state.updated_at, None, canonical_json(payload), digest),
            )
            self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('identity_id',?)", (state.identity_id,))
        return state

    def commit_transition(
        self,
        event: LivingEvent,
        state: LivingState,
        memories: Iterable[MemoryRecord] = (),
        memory_links: Iterable[MemoryLink] = (),
    ) -> LivingState:
        current = self.load_state()
        if current is None:
            raise RuntimeError("store must be initialized before transitions")
        state.version = current.version + 1
        payload = state.to_dict()
        digest = payload_digest(payload)
        with self.db:
            self.db.execute(
                "INSERT INTO events(event_id,at,kind,source,salience,payload_json,after_state_digest) VALUES(?,?,?,?,?,?,?)",
                (event.event_id, event.at, event.kind, event.source, float(event.salience), canonical_json(event.payload), digest),
            )
            self.db.execute(
                "INSERT INTO snapshots(version,at,event_id,state_json,digest) VALUES(?,?,?,?,?)",
                (state.version, state.updated_at, event.event_id, canonical_json(payload), digest),
            )
            for memory in memories:
                self.db.execute(
                    "INSERT OR IGNORE INTO memories(memory_id,created_at,kind,text,salience,confidence,source_event_id,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                    (memory.memory_id, memory.created_at, memory.kind, memory.text, float(memory.salience), float(memory.confidence), memory.source_event_id, canonical_json(memory.metadata)),
                )
            for link in memory_links:
                self.db.execute(
                    "INSERT OR IGNORE INTO memory_links(parent_memory_id,child_memory_id,relation,source_event_id,created_at) VALUES(?,?,?,?,?)",
                    (link.parent_memory_id, link.child_memory_id, link.relation, link.source_event_id, link.created_at),
                )
        return state

    def memories(self, limit: int = 200) -> list[MemoryRecord]:
        rows = self.db.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (max(0, int(limit)),)).fetchall()
        return [
            MemoryRecord(
                memory_id=row["memory_id"],
                created_at=row["created_at"],
                kind=row["kind"],
                text=row["text"],
                salience=row["salience"],
                confidence=row["confidence"],
                source_event_id=row["source_event_id"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def memory_by_ids(self, memory_ids: Iterable[str]) -> dict[str, MemoryRecord]:
        ids = list(dict.fromkeys(str(x) for x in memory_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.execute(
            f"SELECT * FROM memories WHERE memory_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {
            row["memory_id"]: MemoryRecord(
                memory_id=row["memory_id"],
                created_at=row["created_at"],
                kind=row["kind"],
                text=row["text"],
                salience=row["salience"],
                confidence=row["confidence"],
                source_event_id=row["source_event_id"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        }

    def consolidated_parent_ids(self) -> set[str]:
        rows = self.db.execute(
            "SELECT DISTINCT parent_memory_id FROM memory_links WHERE relation='consolidated_into'"
        ).fetchall()
        return {str(row["parent_memory_id"]) for row in rows}

    def memory_links(self, limit: int = 1000) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT parent_memory_id,child_memory_id,relation,source_event_id,created_at FROM memory_links ORDER BY created_at ASC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def replay_records(self, limit: int = 10000, *, after_version: int = 0) -> list[dict[str, object]]:
        rows = self.db.execute(
            """
            SELECT s.version, p.state_json AS before_json, s.state_json AS after_json,
                   e.event_id, e.at, e.kind, e.source, e.salience, e.payload_json
            FROM snapshots AS s
            JOIN events AS e ON e.event_id = s.event_id
            JOIN snapshots AS p ON p.version = s.version - 1
            WHERE s.version > ?
            ORDER BY s.version ASC
            LIMIT ?
            """,
            (max(0, int(after_version)), max(0, int(limit))),
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "version": int(row["version"]),
                    "before": LivingState.from_dict(json.loads(row["before_json"])),
                    "after": LivingState.from_dict(json.loads(row["after_json"])),
                    "event": LivingEvent(
                        event_id=row["event_id"],
                        at=row["at"],
                        kind=row["kind"],
                        source=row["source"],
                        salience=float(row["salience"]),
                        payload=json.loads(row["payload_json"]),
                    ),
                }
            )
        return result

    def snapshot_digest(self, version: int | None = None) -> str | None:
        if version is None:
            row = self.db.execute("SELECT digest FROM snapshots ORDER BY version DESC LIMIT 1").fetchone()
        else:
            row = self.db.execute("SELECT digest FROM snapshots WHERE version=?", (int(version),)).fetchone()
        return None if row is None else str(row["digest"])

    def rollback(self, version: int) -> LivingState:
        row = self.db.execute("SELECT state_json FROM snapshots WHERE version=?", (int(version),)).fetchone()
        if row is None:
            raise ValueError(f"unknown snapshot version: {version}")
        target = LivingState.from_dict(json.loads(row["state_json"]))
        event = LivingEvent(kind="rollback", payload={"target_version": int(version)}, source="operator")
        return self.commit_transition(event, target)
