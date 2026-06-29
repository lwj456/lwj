from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class HistoryRecord:
    id: int
    created_at: str
    operation: str
    input_name: str
    output_name: str | None
    watermark_code: str | None
    detected_code: str | None
    confidence: float | None
    status: str
    artifact_path: str | None
    metadata: dict[str, Any]


class HistoryStore:
    def __init__(self, db_path: Path, storage_root: Path) -> None:
        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_name TEXT NOT NULL,
                    output_name TEXT,
                    watermark_code TEXT,
                    detected_code TEXT,
                    confidence REAL,
                    status TEXT NOT NULL,
                    artifact_path TEXT,
                    metadata TEXT
                )
                """
            )

    def add(self, **payload: Any) -> int:
        created_at = payload.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = payload.get("metadata") or {}
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO history (
                    created_at, operation, input_name, output_name,
                    watermark_code, detected_code, confidence, status,
                    artifact_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    payload["operation"],
                    payload["input_name"],
                    payload.get("output_name"),
                    payload.get("watermark_code"),
                    payload.get("detected_code"),
                    payload.get("confidence"),
                    payload.get("status", "success"),
                    payload.get("artifact_path"),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 100) -> list[HistoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        records: list[HistoryRecord] = []
        for row in rows:
            records.append(
                HistoryRecord(
                    id=int(row["id"]),
                    created_at=row["created_at"],
                    operation=row["operation"],
                    input_name=row["input_name"],
                    output_name=row["output_name"],
                    watermark_code=row["watermark_code"],
                    detected_code=row["detected_code"],
                    confidence=row["confidence"],
                    status=row["status"],
                    artifact_path=row["artifact_path"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
            )
        return records

    def delete(self, record_id: int) -> None:
        record = self.get(record_id)
        if record is None:
            return
        self._remove_artifacts([record.artifact_path, record.metadata.get("input_path")])
        with self._connect() as conn:
            conn.execute("DELETE FROM history WHERE id = ?", (record_id,))

    def clear(self) -> None:
        self._remove_artifacts(
            [row.artifact_path for row in self.list()] + [row.metadata.get("input_path") for row in self.list()]
        )
        with self._connect() as conn:
            conn.execute("DELETE FROM history")

    def get(self, record_id: int) -> HistoryRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        return HistoryRecord(
            id=int(row["id"]),
            created_at=row["created_at"],
            operation=row["operation"],
            input_name=row["input_name"],
            output_name=row["output_name"],
            watermark_code=row["watermark_code"],
            detected_code=row["detected_code"],
            confidence=row["confidence"],
            status=row["status"],
            artifact_path=row["artifact_path"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _remove_artifacts(self, rel_paths: list[str | None]) -> None:
        seen: set[Path] = set()
        for rel_path in rel_paths:
            if not rel_path:
                continue
            abs_path = (self.storage_root / rel_path).resolve()
            if self.storage_root.resolve() not in abs_path.parents and abs_path != self.storage_root.resolve():
                continue
            if abs_path in seen:
                continue
            seen.add(abs_path)
            if abs_path.exists():
                abs_path.unlink()

