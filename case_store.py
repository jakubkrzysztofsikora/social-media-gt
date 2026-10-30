"""Encrypted single-owner case store: SQLite rows + AES-GCM at rest.

Plaintext never touches disk. Associated data binds case_id, schema_version,
and created_at so metadata tampering fails authentication.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CASES_SCHEMA_VERSION = 1
AAD_VERSION = b"case-aad-v1"


class CaseStoreError(Exception):
    pass


class CaseNotFound(CaseStoreError):
    pass


class CaseCorrupt(CaseStoreError):
    pass


class CaseKeyError(CaseStoreError):
    pass


def load_key_from_env(environ=None) -> bytes:
    import os
    environ = environ if environ is not None else os.environ
    raw = environ.get("GT_CASE_ENCRYPTION_KEY", "")
    if not raw:
        raise CaseKeyError("GT_CASE_ENCRYPTION_KEY is required")
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise CaseKeyError("GT_CASE_ENCRYPTION_KEY must be valid Base64") from exc
    if len(key) != 32:
        raise CaseKeyError("GT_CASE_ENCRYPTION_KEY must decode to 32 bytes")
    return key


def _aad(case_id: str, schema_version: int, created_at: str) -> bytes:
    payload = json.dumps([case_id, schema_version, created_at],
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    return AAD_VERSION + b"|" + payload


class CaseStore:
    def __init__(self, db_path: str | os.PathLike, key: bytes):
        if len(key) != 32:
            raise CaseKeyError("case-store key must be 32 bytes")
        self._aes = AESGCM(key)
        self._path = Path(db_path)
        parent = self._path.parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA user_version")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL)""")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- save/load

    def save_case(self, title: str, analysis_request: dict, report: dict,
                  confirmed_by_user: bool) -> dict:
        """Store only completed analyses; report must come from a server rerun."""
        if not confirmed_by_user:
            return {"status": "needs_confirmation"}
        disposition = report.get("disposition") or report.get("status")
        if disposition in ("needs_confirmation", "busy", "solver_unavailable",
                           "error", "unsupported", "invalid", "clarification"):
            raise CaseStoreError(f"refusing to save a {disposition} report")
        if report.get("not_covered"):
            raise CaseStoreError("refusing to save a partial not_covered run")
        if "method_results" in report:
            statuses = [result.get("status") for result in report["method_results"]]
            if (not any(status in ("exact_verified", "epsilon_verified", "no_agreement")
                        for status in statuses)
                    or any(status not in ("exact_verified", "epsilon_verified", "no_agreement",
                                          "unsupported") for status in statuses)):
                raise CaseStoreError("refusing to save an incomplete analysis")
        case_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        plaintext = json.dumps({"title": title, "analysis_request": analysis_request,
                                "report": report}, sort_keys=True).encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = self._aes.encrypt(nonce, plaintext,
                                       _aad(case_id, CASES_SCHEMA_VERSION, created_at))
        with self._conn:
            self._conn.execute(
                "INSERT INTO cases (case_id, created_at, schema_version, nonce, ciphertext)"
                " VALUES (?,?,?,?,?)",
                (case_id, created_at, CASES_SCHEMA_VERSION, nonce, ciphertext))
        return {"status": "saved", "case_id": case_id, "created_at": created_at}

    def _decrypt_row(self, row) -> dict:
        case_id, created_at, schema_version, nonce, ciphertext = row
        try:
            plaintext = self._aes.decrypt(nonce, ciphertext,
                                          _aad(case_id, schema_version, created_at))
        except InvalidTag as exc:
            raise CaseCorrupt(f"case {case_id}: authentication failed "
                              "(wrong key or tampered row)") from exc
        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise CaseCorrupt(f"case {case_id}: unreadable plaintext") from exc
        if not isinstance(payload, dict) or "title" not in payload:
            raise CaseCorrupt(f"case {case_id}: unexpected payload shape")
        return payload

    def load_case(self, case_id: str) -> dict:
        row = self._conn.execute(
            "SELECT case_id, created_at, schema_version, nonce, ciphertext"
            " FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFound(case_id)
        return self._decrypt_row(row)

    def list_cases(self) -> list[dict]:
        out = []
        for row in self._conn.execute(
                "SELECT case_id, created_at, schema_version, nonce, ciphertext"
                " FROM cases ORDER BY created_at"):
            try:
                payload = self._decrypt_row(row)
            except CaseCorrupt as exc:
                out.append({"case_id": row[0], "created_at": row[1],
                            "error": str(exc)})
                continue
            out.append({"case_id": row[0], "created_at": row[1],
                        "schema_version": row[2], "title": payload.get("title", "")})
        return out

    def delete_case(self, case_id: str, confirmed_by_user: bool) -> dict:
        if not confirmed_by_user:
            return {"status": "needs_confirmation"}
        with self._conn:
            cursor = self._conn.execute("DELETE FROM cases WHERE case_id=?", (case_id,))
        if cursor.rowcount == 0:
            raise CaseNotFound(case_id)
        return {"status": "deleted", "case_id": case_id}
