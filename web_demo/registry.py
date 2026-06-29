from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class WatermarkRegistration:
    id: int
    name: str
    code: str
    created_at: str
    model_rel_path: str | None


class WatermarkRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watermark_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    code TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    model_rel_path TEXT
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(watermark_registry)").fetchall()}
            if "model_rel_path" not in columns:
                conn.execute("ALTER TABLE watermark_registry ADD COLUMN model_rel_path TEXT")

    def register(self, name: str, code: str, model_rel_path: str | None = None) -> WatermarkRegistration:
        name = (name or "").strip()
        code = (code or "").strip()
        if not name:
            raise ValueError("名称不能为空")
        if not code:
            raise ValueError("水印编码不能为空")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conflict = conn.execute("SELECT name FROM watermark_registry WHERE code = ?", (code,)).fetchone()
            if conflict is not None and conflict["name"] != name:
                raise ValueError(f"该水印编码已被名称 {conflict['name']} 占用")

            conn.execute("DELETE FROM watermark_registry WHERE name = ?", (name,))
            cursor = conn.execute(
                "INSERT INTO watermark_registry (name, code, created_at, model_rel_path) VALUES (?, ?, ?, ?)",
                (name, code, now, model_rel_path),
            )
            record_id = int(cursor.lastrowid)

        return WatermarkRegistration(
            id=record_id,
            name=name,
            code=code,
            created_at=now,
            model_rel_path=model_rel_path,
        )

    def list(self, limit: int = 100) -> list[WatermarkRegistration]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM watermark_registry ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            WatermarkRegistration(
                id=int(row["id"]),
                name=row["name"],
                code=row["code"],
                created_at=row["created_at"],
                model_rel_path=row["model_rel_path"],
            )
            for row in rows
        ]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM watermark_registry")
