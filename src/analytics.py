# src/analytics.py — SQLite analytics logger for MisInfo Guard admin dashboard
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "analytics.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                endpoint         TEXT    NOT NULL,
                response_time_ms REAL,
                bot_probability  REAL,
                image_risk_score REAL,
                risk_level       TEXT,
                is_bot           INTEGER,
                error            INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extension_pings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                extension_id TEXT,
                version      TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_ts  ON api_calls(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pings_ext ON extension_pings(extension_id)")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_api_call(
    endpoint: str,
    response_time_ms: float,
    bot_probability: float | None = None,
    image_risk_score: float | None = None,
    risk_level: str | None = None,
    is_bot: bool | None = None,
    error: bool = False,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO api_calls
               (timestamp, endpoint, response_time_ms, bot_probability,
                image_risk_score, risk_level, is_bot, error)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                ts, endpoint, round(response_time_ms, 2),
                bot_probability, image_risk_score, risk_level,
                int(is_bot) if is_bot is not None else None,
                int(error),
            ),
        )


def log_extension_ping(extension_id: str | None = None, version: str | None = None) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO extension_pings (timestamp, extension_id, version) VALUES (?,?,?)",
            (ts, extension_id, version),
        )


def get_stats() -> dict:
    with _connect() as conn:
        total_calls = conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE error=0"
        ).fetchone()[0]

        today = datetime.now(timezone.utc).date().isoformat()
        calls_today = conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE timestamp LIKE ? AND error=0",
            (f"{today}%",),
        ).fetchone()[0]

        avg_bot = conn.execute(
            """SELECT AVG(bot_probability) FROM
               (SELECT bot_probability FROM api_calls
                WHERE bot_probability IS NOT NULL ORDER BY id DESC LIMIT 1000)"""
        ).fetchone()[0]

        avg_rt = conn.execute(
            """SELECT AVG(response_time_ms) FROM
               (SELECT response_time_ms FROM api_calls
                WHERE error=0 ORDER BY id DESC LIMIT 1000)"""
        ).fetchone()[0]

        unique_installs = conn.execute(
            "SELECT COUNT(DISTINCT extension_id) FROM extension_pings WHERE extension_id IS NOT NULL"
        ).fetchone()[0]

        total_pings = conn.execute(
            "SELECT COUNT(*) FROM extension_pings"
        ).fetchone()[0]

        risk_rows = conn.execute(
            """SELECT risk_level, COUNT(*) AS cnt FROM api_calls
               WHERE risk_level IS NOT NULL GROUP BY risk_level"""
        ).fetchall()

        daily_rows = conn.execute(
            """SELECT strftime('%Y-%m-%d', timestamp) AS day, COUNT(*) AS cnt
               FROM api_calls
               WHERE timestamp >= datetime('now', '-30 days')
               GROUP BY day ORDER BY day"""
        ).fetchall()

        hourly_rows = conn.execute(
            """SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour, COUNT(*) AS cnt
               FROM api_calls
               WHERE timestamp >= datetime('now', '-1 day')
               GROUP BY hour ORDER BY hour"""
        ).fetchall()

        bot_probs = conn.execute(
            """SELECT bot_probability FROM api_calls
               WHERE bot_probability IS NOT NULL ORDER BY id DESC LIMIT 500"""
        ).fetchall()

        img_risks = conn.execute(
            """SELECT image_risk_score FROM api_calls
               WHERE image_risk_score IS NOT NULL ORDER BY id DESC LIMIT 500"""
        ).fetchall()

        recent = conn.execute(
            """SELECT timestamp, endpoint, bot_probability, image_risk_score,
                      risk_level, response_time_ms, error
               FROM api_calls ORDER BY id DESC LIMIT 25"""
        ).fetchall()

        rt_trend = conn.execute(
            """SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour,
                      AVG(response_time_ms) AS avg_ms
               FROM api_calls
               WHERE timestamp >= datetime('now', '-1 day') AND error=0
               GROUP BY hour ORDER BY hour"""
        ).fetchall()

    return {
        "total_calls":          total_calls,
        "calls_today":          calls_today,
        "avg_bot_probability":  round(avg_bot, 4) if avg_bot else 0.0,
        "avg_response_time_ms": round(avg_rt,  1) if avg_rt  else 0.0,
        "unique_installs":      unique_installs,
        "total_pings":          total_pings,
        "risk_breakdown":       {r["risk_level"]: r["cnt"] for r in risk_rows},
        "daily_calls":          [{"day":  r["day"],  "count": r["cnt"]} for r in daily_rows],
        "hourly_calls":         [{"hour": r["hour"], "count": r["cnt"]} for r in hourly_rows],
        "bot_probabilities":    [r["bot_probability"]  for r in bot_probs],
        "image_risk_scores":    [r["image_risk_score"] for r in img_risks],
        "recent_calls":         [dict(r) for r in recent],
        "rt_trend":             [{"hour": r["hour"], "avg_ms": round(r["avg_ms"], 1)} for r in rt_trend],
    }
