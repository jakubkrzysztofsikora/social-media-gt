"""Case-store fixtures: encryption, AAD tamper detection, confirmation gates."""
import base64
import json
import os

import pytest

from case_store import (CaseCorrupt, CaseKeyError, CaseNotFound, CaseStore,
                        load_key_from_env)


@pytest.fixture()
def key():
    return os.urandom(32)


@pytest.fixture()
def store(tmp_path, key):
    return CaseStore(tmp_path / "cases.sqlite3", key)


def _report():
    return {"disposition": "ok", "method_results": [{"method": "pure_nash",
                                                     "status": "exact_verified"}]}


def test_save_requires_confirmation(store):
    out = store.save_case("t", {"game_spec": {}}, _report(), confirmed_by_user=False)
    assert out["status"] == "needs_confirmation"
    assert store.list_cases() == []


def test_save_and_load_roundtrip(store):
    out = store.save_case("title", {"game_spec": {"k": 1}}, _report(),
                          confirmed_by_user=True)
    assert out["status"] == "saved"
    loaded = store.load_case(out["case_id"])
    assert loaded["title"] == "title"
    assert loaded["report"]["method_results"][0]["method"] == "pure_nash"


def test_refuses_dispositions_and_partial(store):
    for disposition in ("needs_confirmation", "busy", "solver_unavailable", "error"):
        with pytest.raises(Exception):
            store.save_case("t", {}, {"disposition": disposition}, confirmed_by_user=True)
    with pytest.raises(Exception):
        store.save_case("t", {}, {"disposition": "ok", "not_covered": [0]},
                        confirmed_by_user=True)


def test_rejects_incomplete_method_results_without_saving(store):
    from case_store import CaseStoreError

    for methods in (
        [{"method": "pure_nash", "status": "error"}],
        [{"method": "pure_nash", "status": "solver_unavailable"}],
        [{"method": "pure_nash", "status": "unsupported"}],
        [],
    ):
        with pytest.raises(CaseStoreError):
            store.save_case("t", {}, {"disposition": "ok", "method_results": methods},
                            confirmed_by_user=True)
    assert store.list_cases() == []


def test_plaintext_absence_on_disk(tmp_path, key):
    path = tmp_path / "cases.sqlite3"
    store = CaseStore(path, key)
    store.save_case("secret-title", {"game_spec": {"alpha": 1}}, _report(),
                    confirmed_by_user=True)
    blob = path.read_bytes()
    assert b"secret-title" not in blob
    assert b"alpha" not in blob
    assert b"exact_verified" not in blob


def test_wrong_key_fails_without_row_change(tmp_path, key):
    path = tmp_path / "cases.sqlite3"
    store = CaseStore(path, key)
    out = store.save_case("t", {}, _report(), confirmed_by_user=True)
    other = CaseStore(path, os.urandom(32))
    with pytest.raises(CaseCorrupt):
        other.load_case(out["case_id"])
    assert len(other.list_cases()) == 1          # row listed as corrupt, not deleted
    assert len(CaseStore(path, key).list_cases()) == 1


def test_metadata_tampering_fails_authentication(tmp_path, key):
    import sqlite3
    path = tmp_path / "cases.sqlite3"
    store = CaseStore(path, key)
    out = store.save_case("t", {}, _report(), confirmed_by_user=True)
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE cases SET created_at='2000-01-01T00:00:00.000000+00:00'"
                 " WHERE case_id=?", (out["case_id"],))
    conn.commit()
    with pytest.raises(CaseCorrupt):
        store.load_case(out["case_id"])


def test_delete_requires_confirmation_and_missing_id(store):
    out = store.save_case("t", {}, _report(), confirmed_by_user=True)
    assert store.delete_case(out["case_id"], confirmed_by_user=False)["status"] == \
        "needs_confirmation"
    assert store.delete_case(out["case_id"], confirmed_by_user=True)["status"] == "deleted"
    with pytest.raises(CaseNotFound):
        store.delete_case(out["case_id"], confirmed_by_user=True)


def test_load_unknown_id(store):
    with pytest.raises(CaseNotFound):
        store.load_case("missing")


def test_key_loader():
    env = {"GT_CASE_ENCRYPTION_KEY": base64.b64encode(os.urandom(32)).decode()}
    assert len(load_key_from_env(env)) == 32
    with pytest.raises(CaseKeyError):
        load_key_from_env({})
    with pytest.raises(CaseKeyError):
        load_key_from_env({"GT_CASE_ENCRYPTION_KEY": "not-base64!!"})
    short = base64.b64encode(os.urandom(16)).decode()
    with pytest.raises(CaseKeyError):
        load_key_from_env({"GT_CASE_ENCRYPTION_KEY": short})
