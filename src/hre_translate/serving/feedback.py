from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from hre_translate.serving.schemas import FeedbackRequest, FeedbackResponse


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        correction TEXT NOT NULL,
                        model TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        rating INTEGER CHECK (rating IS NULL OR rating BETWEEN 1 AND 5)
                    )"""
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def save(self, feedback: FeedbackRequest) -> FeedbackResponse:
        timestamp = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """INSERT INTO feedback
                       (source, prediction, correction, model, timestamp, rating)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        feedback.source,
                        feedback.prediction,
                        feedback.correction,
                        feedback.model,
                        timestamp,
                        feedback.rating,
                    ),
                )
                row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("Feedback insert did not return an id")
        return FeedbackResponse(id=row_id, timestamp=timestamp)
