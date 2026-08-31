"""SHA-256 hash-chained audit trail for every model decision (prediction,
anomaly flag, scenario run, calibration run, reviewer decision).

Each row's entry_hash = SHA256(prev_hash | timestamp | event_type | loan_id |
references_hash | canonical-JSON payload). Every row commits to the hash of
the row before it, so a single SQLite table becomes tamper-evident: editing
any historical row's payload -- or its stored hash -- changes what verify()
recomputes for that row, which no longer matches the next row's prev_hash,
and every subsequent link breaks. This is a simplified, single-table
version of the Postgres hash-chaining used for agent governance in a prior
project (AegisAgent); SQLite is enough here since there's no
concurrent-writer scenario to design around.

`references_hash` links a decision to a specific earlier entry rather than
just sharing its loan_id -- e.g. a reviewer's override references the exact
entry_hash of the automated flag it's responding to (review/decisions.py).
It's folded into the hash material like every other field, so an override
can't be silently re-pointed at a different flag after the fact without
breaking the chain. Entries with nothing to reference (a plain prediction,
a scenario run) store "" for it.

Run directly: `python -m loan_intelligence.audit.hash_chain` logs a few
example entries from existing model outputs, verifies the chain, then
deliberately tampers with one row and re-verifies to show the break.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_DB_PATH = OUTPUT_DIR / "audit_trail.db"
GENESIS_HASH = "0" * 64


def _compute_hash(prev_hash: str, timestamp: str, event_type: str, loan_id: str, references_hash: str, payload_json: str) -> str:
    material = "|".join([prev_hash, timestamp, event_type, loan_id, references_hash, payload_json]).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class AuditTrail:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                loan_id TEXT,
                references_hash TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def _last_hash(self, conn: sqlite3.Connection) -> str:
        row = conn.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS_HASH

    def log(self, event_type: str, loan_id: str | None, payload: dict, references_hash: str = "") -> str:
        """Appends one hash-chained entry. Returns the new entry's hash.

        references_hash: the entry_hash of a specific prior entry this one
        responds to (e.g. a reviewer_decision referencing the flag it
        overrides). Leave blank for entries that don't respond to anything."""
        conn = self._connect()
        try:
            prev_hash = self._last_hash(conn)
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            entry_hash = _compute_hash(prev_hash, timestamp, event_type, loan_id or "", references_hash, payload_json)
            conn.execute(
                "INSERT INTO audit_log (timestamp, event_type, loan_id, references_hash, payload, prev_hash, entry_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, event_type, loan_id, references_hash, payload_json, prev_hash, entry_hash),
            )
            conn.commit()
            return entry_hash
        finally:
            conn.close()

    def verify(self) -> dict:
        """Walks the full chain from genesis, recomputing each hash. Returns
        {"valid": True, "n_entries": N} or {"valid": False, "broken_at_id": id,
        "reason": ...} at the first link that doesn't reproduce."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, timestamp, event_type, loan_id, references_hash, payload, prev_hash, entry_hash "
            "FROM audit_log ORDER BY id ASC"
        ).fetchall()
        conn.close()

        expected_prev = GENESIS_HASH
        for row_id, timestamp, event_type, loan_id, references_hash, payload, prev_hash, entry_hash in rows:
            if prev_hash != expected_prev:
                return {"valid": False, "broken_at_id": row_id, "reason": "prev_hash does not match preceding entry's hash"}
            recomputed = _compute_hash(prev_hash, timestamp, event_type, loan_id or "", references_hash, payload)
            if recomputed != entry_hash:
                return {"valid": False, "broken_at_id": row_id, "reason": "stored entry_hash does not match recomputed hash (payload or metadata was altered)"}
            expected_prev = entry_hash
        return {"valid": True, "n_entries": len(rows)}

    def history_for_loan(self, loan_id: str) -> pd.DataFrame:
        conn = self._connect()
        df = pd.read_sql_query(
            "SELECT id, timestamp, event_type, loan_id, references_hash, payload, entry_hash FROM audit_log "
            "WHERE loan_id = ? ORDER BY id ASC",
            conn, params=(loan_id,),
        )
        conn.close()
        return df

    def get_entry_by_hash(self, entry_hash: str) -> dict | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT id, timestamp, event_type, loan_id, references_hash, payload, entry_hash "
            "FROM audit_log WHERE entry_hash = ?",
            (entry_hash,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        keys = ["id", "timestamp", "event_type", "loan_id", "references_hash", "payload", "entry_hash"]
        return dict(zip(keys, row))

    def export_all(self) -> pd.DataFrame:
        conn = self._connect()
        df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY id ASC", conn)
        conn.close()
        return df


def _tamper_row(db_path: Path, row_id: int) -> None:
    """Directly rewrites a historical row's payload via raw SQL, bypassing
    AuditTrail.log() entirely -- simulates an operator/DBA editing the table
    out-of-band, which is exactly the threat model hash-chaining defends
    against."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE audit_log SET payload = ? WHERE id = ?",
        (json.dumps({"tampered": True, "note": "probability quietly changed after the fact"}), row_id),
    )
    conn.commit()
    conn.close()


def main() -> None:
    # Uses a separate, throwaway DB for this self-contained demo (including
    # the deliberate tamper below) so it never poisons outputs/audit_trail.db,
    # the trail that real pipeline runs (scenario/simulate.py,
    # review/narrate.py, demo.py) actually append to.
    demo_db_path = OUTPUT_DIR / "audit_trail_demo.db"
    demo_db_path.unlink(missing_ok=True)
    trail = AuditTrail(db_path=demo_db_path)

    predictions = pd.read_csv(OUTPUT_DIR / "predictions_test_calibrated.csv").head(3)
    anomalies = pd.read_csv(OUTPUT_DIR / "anomaly_scores_test.csv").sort_values("anomaly_score", ascending=False).head(2)

    print("logging example prediction entries...")
    for _, r in predictions.iterrows():
        trail.log(
            "prediction",
            r["loan_id"],
            {"xgb_proba_raw": float(r["xgb_proba"]), "xgb_proba_calibrated": float(r["xgb_proba_calibrated"]), "default_flag_actual": int(r["default_flag"])},
        )

    print("logging example anomaly-flag entries...")
    for _, r in anomalies.iterrows():
        trail.log(
            "anomaly_flag",
            r["loan_id"],
            {"anomaly_score": float(r["anomaly_score"]), "flagged_top_1pct": bool(r["flagged_top_1pct"])},
        )

    print("logging an example scenario-run entry...")
    trail.log(
        "scenario_run",
        None,
        {"shock": {"interest_rate": "+2.0"}, "portfolio_default_rate_before": 0.078, "portfolio_default_rate_after": 0.091},
    )

    result = trail.verify()
    print(f"\nchain verification (before tamper): {result}")

    print("\n--- simulating an out-of-band edit directly in SQLite (bypassing AuditTrail.log) ---")
    all_rows = trail.export_all()
    tampered_id = int(all_rows["id"].iloc[0])
    _tamper_row(trail.db_path, tampered_id)

    result_after = trail.verify()
    print(f"chain verification (after tampering row id={tampered_id}): {result_after}")

    print(f"\naudit DB: {trail.db_path}")


if __name__ == "__main__":
    main()
