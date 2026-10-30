import json

import webapp


def test_draft_edit_invalidates_results_and_advanced_confirmation():
    assert webapp.spec_edited() == (
        "", "", None, "", None, False, "", None, None, None,
    )

def test_spec_change_clears_stale_results_and_preserves_loaded_report(monkeypatch):
    spec_json = '{"game":{"kind":"normal_form"}}'
    edited_spec = '{"game":{"kind":"bargaining"}}'
    monkeypatch.setattr(webapp.plain_view, "describe_draft",
                        lambda spec: ("Updated summary", True))
    loaded_result = webapp._tagged(spec_json, None)

    changed = webapp.spec_changed(edited_spec, loaded_result)
    opened = webapp.spec_changed(spec_json, loaded_result)

    assert changed[:2] == ("Updated summary", webapp.gr.update(visible=True))
    assert changed[2:] == webapp._cleared()
    assert opened[:2] == ("Updated summary", webapp.gr.update(visible=True))
    assert opened[2:] == (webapp.gr.skip(),) * 10


def test_saved_plan_restores_matching_report_and_solved_state(monkeypatch):
    spec = {"template_id": "custom", "game": {"kind": "normal_form"}}
    report = {"disposition": "ok", "method_results": [{"method": "pure_nash"}]}
    markdown = "Saved analysis"
    html = "<html>saved analysis</html>"
    monkeypatch.setattr(webapp, "case_load", lambda case_id: {
        "title": "Saved plan",
        "analysis_request": {"game_spec": spec},
        "report": report,
    })
    monkeypatch.setattr(webapp.plain_view, "describe_result",
                        lambda loaded_report, spec_json: markdown)
    monkeypatch.setattr(webapp.share_report, "render_report_html",
                        lambda loaded_markdown, autoprint: html)

    opened = webapp.plan_open("case-id")

    assert opened[1] == json.dumps(spec, indent=2)
    assert opened[2:5] == (markdown, html, report)
    assert opened[6:9] == (None, False, "")
    assert opened[9] == webapp._tagged(opened[1], (report, markdown, html))
    assert opened[10:] == (None, None)


def test_save_rejects_confirmation_for_different_draft(monkeypatch):
    spec_json = '{"game":{"kind":"normal_form"}}'
    saved = []
    monkeypatch.setattr(webapp, "case_save",
                        lambda *args: saved.append(args) or {"status": "saved"})

    message, _ = webapp.plan_save(
        "Plan", spec_json, webapp.spec_fingerprint('{"game":{"kind":"other"}}'))

    assert message == webapp.CONFIRM_BEFORE_SAVE
    assert saved == []


def test_save_reports_case_store_error(monkeypatch):
    spec_json = '{"game":{"kind":"normal_form"}}'

    def fail_save(*args):
        raise webapp.case_store.CaseStoreError("disk unavailable")

    monkeypatch.setattr(webapp, "case_save", fail_save)
    monkeypatch.setattr(webapp, "_plans_update", lambda message: (message, []))

    message, choices = webapp.plan_save("Plan", spec_json,
                                        webapp.spec_fingerprint(spec_json))

    assert message == "Could not save: disk unavailable"
    assert choices == []


def test_case_save_reports_case_store_error(monkeypatch):
    def fail_store():
        raise webapp.case_store.CaseStoreError("key unavailable")

    monkeypatch.setattr(webapp, "_case_store", fail_store)

    out = webapp.case_save("Plan", '{"game": {}}', True)

    assert out == {"status": "error", "error": "key unavailable"}
