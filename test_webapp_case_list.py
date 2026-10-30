import webapp


def test_plan_refresh_shows_corrupt_case_id_and_keeps_healthy_choice(monkeypatch):
    monkeypatch.setattr(webapp, "case_list", lambda: [
        {"case_id": "healthy-id", "created_at": "2026-10-28T00:00:00Z",
         "title": "Healthy plan"},
        {"case_id": "bad-key-id", "created_at": "2026-10-29T00:00:00Z",
         "error": "case bad-key-id: authentication failed (wrong key or tampered row)"},
    ])

    status, picker = webapp.plan_refresh()

    assert "bad-key-id" in status
    assert "authentication failed" in status
    assert picker["choices"] == [("Healthy plan (2026-10-28)", "healthy-id")]


def test_plan_refresh_shows_global_encryption_key_error(monkeypatch):
    monkeypatch.setattr(webapp, "case_list", lambda: [
        {"error": "GT_CASE_ENCRYPTION_KEY must decode to 32 bytes"}
    ])

    status, picker = webapp.plan_refresh()

    assert "GT_CASE_ENCRYPTION_KEY" in status
    assert picker["choices"] == []

def test_filesystem_errors_surface_on_saved_plan_actions_without_deleting_records(
        monkeypatch, tmp_path):
    import base64

    monkeypatch.setenv("GT_CASE_ENCRYPTION_KEY", base64.b64encode(b"k" * 32).decode())
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied")
    monkeypatch.setenv("GT_CASE_DB_PATH", str(blocked_parent / "cases.sqlite3"))
    monkeypatch.setattr(webapp, "run_worker_request", lambda request: {"status": "complete"})

    list_error = webapp.case_list()
    assert "File exists" in list_error[0]["error"]
    assert "File exists" in webapp.case_save("Plan", "{}", True)["error"]
    assert "File exists" in webapp.case_load("case-id")["error"]
    assert "File exists" in webapp.case_delete("case-id", True)["error"]
    assert "File exists" in webapp.plan_refresh()[0]
    fingerprint = webapp.spec_fingerprint("{}")
    assert "File exists" in webapp.plan_save("Plan", "{}", fingerprint)[0]
    assert "File exists" in webapp.plan_open("case-id")[0]
    assert "File exists" in webapp.plan_delete("case-id")[0]
    path = tmp_path / "healthy.sqlite3"
    store = webapp.case_store.CaseStore(path, b"k" * 32)
    saved = store.save_case("Keep me", {"game_spec": {}}, {"status": "complete"}, True)
    monkeypatch.setattr(webapp, "_case_store", lambda: store)
    monkeypatch.setattr(
        store, "delete_case",
        lambda case_id, confirmed_by_user: (_ for _ in ()).throw(OSError("disk failure")))

    status, _ = webapp.plan_delete(saved["case_id"])
    assert "disk failure" in status
    assert store.list_cases()[0]["case_id"] == saved["case_id"]
    store.close()
