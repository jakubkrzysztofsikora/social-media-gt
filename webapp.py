"""Gradio chat app over the social-media game-theory advisor.

Describe a situation in plain words; the advisor infers the toolkit's parameters,
runs all ten scenarios and renders the full mathematical evidence.

The surface is deliberately quiet: one query bar, a few example queries, the
conversation as bordered pods and the evidence below them.
"""
from __future__ import annotations

import argparse

import gradio as gr

import advisor

BRAND = """# social-media-gt

Map a real situation onto the ten game-theoretic scenarios of the paper and read the
exact-arithmetic evidence for every one of them. Describe the situation in your own words,
then press Enter or **Analyze**.
"""

EXAMPLES = [
    ["Two factories in a valley must cut output together. Each gains 1 from running flat out and "
     "cutting costs each 1/2. If both cut, the valley clears and the fine disappears; if only one "
     "cuts, it eats the whole fine alone while the other keeps the full gain. Nobody moves first."],
    ["A cooperative of ten shareholders votes to fund a shared project; it only works if enough of "
     "them contribute, one shareholder is abroad and never contributes, and each contributor pays "
     "3/2 to join while a free-rider pays nothing."],
    ["Twelve villages share an irrigation canal; each village gains 1 from the water, maintenance "
     "costs 1/2 per village, and a village that stops maintaining while the others continue still "
     "receives its share."],
    ["A cartel of twenty oil producers; the regulator caps output for the two thirds of producers "
     "it can observe directly, the rest route barrels through shell companies, and the authority "
     "needs a total output cut of 30% at the lowest enforcement cost."],
]

CSS = """
:root {
  --gt-accent: #2c6fbb;
  --gt-accent-hover: #1f5a99;
  --gt-page: #ffffff;
  --gt-surface: #ffffff;
  --gt-surface-muted: #f5f8fc;
  --gt-border: #e3e3e3;
  --gt-border-strong: #d6d6d6;
  --gt-text: #2b2b2b;
  --gt-muted: #707070;
  --gt-label: #767676;
}

body,
.gradio-container,
body.dark .gradio-container {
  --background-fill-primary: var(--gt-page);
  --background-fill-secondary: #f7f7f7;
  --panel-background-fill: var(--gt-surface);
  --block-background-fill: var(--gt-surface);
  --input-background-fill: var(--gt-surface);
  --body-text-color: var(--gt-text);
  --body-text-color-subdued: var(--gt-muted);
  --block-border-color: var(--gt-border);
  --border-color-primary: var(--gt-border);
  --block-label-text-color: var(--gt-label);
  --button-primary-background-fill: var(--gt-accent);
  --color-accent: var(--gt-accent);
  color: var(--gt-text) !important;
}

body,
.gradio-container,
body.dark .gradio-container {
  background: var(--gt-page) !important;
}

gradio-app,
body.dark gradio-app {
  background: var(--gt-page) !important;
}

.gradio-container label,
body.dark .gradio-container label {
  background: transparent !important;
}

.gradio-container .label-wrap,
body.dark .gradio-container .label-wrap {
  color: var(--gt-text) !important;
}

.gradio-container > .main {
  box-sizing: border-box;
  width: 100% !important;
  max-width: 940px !important;
  margin: 0 auto !important;
  padding: 22px 16px 40px;
}

.gradio-container main.contain {
  box-sizing: border-box;
  width: 100% !important;
  max-width: 100% !important;
  margin-inline: auto !important;
}

#brand { margin: 0 0 14px; }
#brand h1 { font-size: 1.02rem; font-weight: 600; color: var(--gt-text); margin: 0 0 3px; }
#brand p { font-size: .82rem; color: var(--gt-muted); margin: 0; max-width: 68ch; line-height: 1.5; }

#query-bar {
  display: flex;
  flex-wrap: nowrap !important;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--gt-border-strong);
  border-radius: 12px;
  background: var(--gt-surface);
  padding: 6px;
}
#query-bar:focus-within { border-color: var(--gt-accent); box-shadow: 0 0 0 3px rgba(44, 111, 187, .16); }
#query-input, #game-input { flex: 1 1 0% !important; width: auto !important; min-width: 0 !important; }
#query-input textarea,
#game-input textarea {
  box-sizing: border-box;
  width: 100%;
  min-height: 76px;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  color: var(--gt-text) !important;
  font-size: .95rem;
  line-height: 1.5;
  padding: 8px 6px;
  resize: none;
}
#query-input textarea:focus,
#game-input textarea:focus { outline: none !important; }
#analyze-button {
  flex: 0 0 112px !important;
  width: 112px !important;
  min-width: 112px !important;
  height: 44px;
  padding: 0 14px;
  border: none;
  border-radius: 8px;
  background: var(--gt-accent);
  color: #fff;
  font-size: .88rem;
  font-weight: 600;
}
#analyze-button:hover { background: var(--gt-accent-hover); }

#game-query-bar {
  display: flex;
  flex-wrap: nowrap !important;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--gt-border-strong);
  border-radius: 12px;
  background: var(--gt-surface);
  padding: 6px;
}
#game-query-bar:focus-within { border-color: var(--gt-accent); box-shadow: 0 0 0 3px rgba(44, 111, 187, .16); }
#game-draft-button {
  flex: 0 0 140px !important;
  width: 140px !important;
  min-width: 140px !important;
  height: 44px;
  padding: 0 14px;
  border: none;
  border-radius: 8px;
  background: var(--gt-accent);
  color: #fff;
  font-size: .88rem;
  font-weight: 600;
}
#game-draft-button:hover { background: var(--gt-accent-hover); }

@media (max-width: 720px) {
  #query-bar,
  #game-query-bar { flex-wrap: wrap !important; }
  #query-input,
  #game-input { flex: 1 1 100% !important; width: 100% !important; }
  #analyze-button,
  #game-draft-button {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 0 !important;
    height: 48px;
  }
  #query-input textarea,
  #game-input textarea { font-size: 16px; min-height: 104px; }
}

#examples { margin: 10px 0 18px; }
#examples .label { display: none; }
#examples table { border: none !important; }
#examples td { border: none !important; padding: 3px 8px 3px 0 !important; }
#examples .gallery-item {
  height: auto !important;
  padding: 5px 12px !important;
  border: 1px solid var(--gt-border-strong) !important;
  border-radius: 999px !important;
  background: var(--gt-surface) !important;
  color: #4a4a4a !important;
  font-size: .78rem !important;
  text-align: left;
}
#examples .gallery-item:hover { border-color: var(--gt-accent) !important; color: var(--gt-accent) !important; }

#conversation { border: none !important; background: transparent !important; padding: 0 !important; }
#conversation .chatbot { border: none !important; background: transparent !important; }
#conversation .message-wrap { padding: 0 !important; }
#conversation .message-row { border: none !important; background: transparent !important; padding: 2px 0 !important; }
#conversation .message {
  border: 1px solid var(--gt-border);
  border-radius: 10px;
  background: var(--gt-surface);
  color: var(--gt-text) !important;
  padding: 10px 14px;
  font-size: .87rem;
  line-height: 1.55;
}
#conversation .user-row .message { background: var(--gt-surface-muted); }
#conversation .message-content { color: var(--gt-text) !important; font-size: .87rem; }
#conversation .message-content h1 { font-size: .98rem; }
#conversation .message-content h2 { font-size: .9rem; }

#evidence {
  border: 1px solid var(--gt-border);
  border-radius: 10px;
  background: var(--gt-surface);
  color: var(--gt-text) !important;
  padding: 14px 16px;
}
#evidence .prose { color: var(--gt-text) !important; font-size: .85rem; }
#evidence h1 { font-size: .98rem; }
#evidence h2 { font-size: .89rem; border-bottom: 1px solid #eeeeee; padding-bottom: 4px; }
#evidence table { font-size: .8rem; }

#raw-json {
  border: 1px solid var(--gt-border);
  border-radius: 10px;
  background: var(--gt-surface) !important;
  color: var(--gt-text) !important;
  font-size: .78rem;
}
#raw-json .json-node,
#raw-json .json-node.dark-mode {
  background: var(--gt-surface) !important;
  color: var(--gt-text) !important;
  border-color: var(--gt-border) !important;
}

.gradio-container label { color: var(--gt-label); font-size: .72rem; letter-spacing: .04em; }
"""

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]


def respond(history, state, text):
    messages = list(state or history or [])
    if not text or not str(text).strip():
        return messages, messages, "", {}, "", "", ""
    messages.append({"role": "user", "content": str(text)})
    try:
        report = advisor.analyze_situation(messages)
    except Exception as exc:
        failure = f"**Analysis failed:** {exc}"
        messages.append({"role": "assistant", "content": f"Analysis failed: {exc}"})
        return messages, messages, failure, {}, "", "", ""
    raw = {key: value for key, value in report.items() if key != "markdown"}
    if report.get("need_more_information"):
        questions = "\n".join(f"{index}. {question}" for index, question
                              in enumerate(report.get("questions", []), 1))
        messages.append({"role": "assistant",
                         "content": f"Before I run the numbers:\n{questions}"})
        return messages, messages, report["markdown"], raw, "", "", ""
    markdown = report["markdown"]
    html = share_report.render_report_html(markdown, autoprint=True)
    messages.append({"role": "assistant", "content": "Analysis complete. Full evidence is below."})
    return messages, messages, markdown, raw, "", markdown, html


# ---------------------------------------------------------------------------
# Sharing: local PDF (browser print) and public link on share.jakub.team
# ---------------------------------------------------------------------------

import json

import share_report

# gr.State lives server-side and never reaches js inputs; a hidden Textbox does.
PDF_OPEN_JS = """
(html) => {
  if (!html) { alert("Run an analysis first."); return null; }
  const blob = new Blob([html], {type: "text/html"});
  const url = URL.createObjectURL(blob);
  const w = window.open(url, "_blank");
  if (!w) { alert("Allow popups for this site to save the PDF."); }
  setTimeout(() => URL.revokeObjectURL(url), 120000);
  return null;
}
"""

DRAFT_JS = """
() => {
  const fields = ["query-input", "game-input"];
  const buttons = {"analyze-button": "query-input", "game-draft-button": "game-input"};
  const key = (id) => "gt-draft:" + id;

  const wireField = (el) => {
    const ta = el.querySelector("textarea");
    if (!ta) { return; }
    const saved = localStorage.getItem(key(el.id));
    if (saved && !ta.value) {
      ta.value = saved;
      ta.dispatchEvent(new Event("input", {bubbles: true}));
    }
    ta.addEventListener("input", () => localStorage.setItem(key(el.id), ta.value));
    el.dataset.gtDraft = "1";
  };

  const wireButton = (btn, fieldId) => {
    if (btn.dataset.gtDraft) { return; }
    btn.addEventListener("click", () => localStorage.removeItem(key(fieldId)));
    btn.dataset.gtDraft = "1";
  };

  const run = () => {
    fields.forEach((id) => {
      const el = document.getElementById(id);
      if (el && !el.dataset.gtDraft) { wireField(el); }
    });
    Object.entries(buttons).forEach(([id, fieldId]) => {
      const btn = document.getElementById(id);
      if (btn) { wireButton(btn, fieldId); }
    });
  };

  let pending = null;
  const schedule = () => {
    if (pending) { return; }
    pending = setTimeout(() => { pending = null; run(); }, 150);
  };

  run();
  [400, 1200, 3000].forEach((ms) => setTimeout(run, ms));
  new MutationObserver(schedule).observe(document.body, {childList: true, subtree: true});
  return null;
}
"""


def _public_link(markdown: str, title: str) -> str:
    if not markdown or not markdown.strip():
        return "Run an analysis first - there is nothing to share yet."
    out = share_report.public_share_link(markdown, title=title)
    if out["status"] == "ok":
        return out["url"]
    return f"Share failed: {out.get('error', 'unknown error')}"


def paper_public_link(markdown: str) -> str:
    return _public_link(markdown, "social-media-gt scenario analysis")


def game_public_link(markdown) -> str:
    return _public_link(markdown or "", "social-media-gt game analysis")


# ---------------------------------------------------------------------------
# General games tab
# ---------------------------------------------------------------------------

import hashlib
import json as _json
import os
import sqlite3

import case_store
import plain_view
from game_estimation import analyze_sensitivity, apply_estimates, estimate_game_parameters
from game_models import SCHEMA_VERSION
from game_simulation import simulate_game
from solver_worker import run_worker_request


def spec_fingerprint(spec_json) -> str:
    text = "" if spec_json is None else str(spec_json)
    try:
        canonical = _json.dumps(_json.loads(text), sort_keys=True, separators=(",", ":"))
    except (ValueError, RecursionError):
        canonical = "text:" + text
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _blank(spec_json) -> bool:
    return not str(spec_json or "").strip()


def _spec_confirmed(spec_json, checked, confirmed_fp) -> bool:
    return (bool(checked) and bool(confirmed_fp) and not _blank(spec_json)
            and spec_fingerprint(spec_json) == confirmed_fp)


def confirm_spec(checked, spec_json):
    return spec_fingerprint(spec_json) if checked and not _blank(spec_json) else ""


def _cleared():
    return "", "", None, "", None, False, "", None, None, None


def spec_edited(spec_json=None, solved_state=None):
    if (isinstance(solved_state, dict)
            and solved_state.get("fingerprint") == spec_fingerprint(spec_json)):
        return (gr.skip(),) * 10
    return _cleared()


def spec_changed(spec_json, solved_state):
    summary, solve_button = draft_view(spec_json)
    return summary, solve_button, *spec_edited(spec_json, solved_state)


def _tagged(spec_json, value):
    return {"fingerprint": spec_fingerprint(spec_json), "value": value}


def _store_failure(exc) -> dict:
    return {"status": "error", "error": str(exc) or type(exc).__name__}


def _intake_result(messages, message, draft=""):
    return messages, gr.update(value=messages, visible=bool(messages)), message, draft, ""


def _clarification_rounds(history) -> int:
    return sum(1 for message in history or []
               if message.get("role") == "assistant"
               and str(message.get("content", "")).startswith("A couple of questions"))


def game_intake(history, text):
    messages = list(history or [])
    if not text or not str(text).strip():
        return _intake_result(messages, "Describe the situation above.")
    asked_before = _clarification_rounds(history) >= 1
    messages.append({"role": "user", "content": str(text)})
    try:
        parsed = advisor.infer_game_model(messages, asked_already=asked_before)
    except Exception as exc:
        messages.append({"role": "assistant", "content": f"Something went wrong: {exc}"})
        return _intake_result(messages, f"Something went wrong: {exc}")
    disposition = parsed.get("disposition", "")
    if disposition == "clarification" and asked_before:
        messages.append({"role": "assistant",
                         "content": "I still cannot see who is involved and what each side can "
                                    "choose."})
        return _intake_result(messages, "Say it as one concrete example, like: two neighbours "
                                        "decide whether to share a fence, both want the fence but "
                                        "neither wants to pay for it.")
    if disposition == "clarification":
        questions = parsed.get("questions") or []
        if not questions:
            messages.append({"role": "assistant",
                             "content": "I cannot see who decides or what each side can do."})
            return _intake_result(messages, "Say who is involved, what each of them can choose, "
                                            "and which result each one prefers.")
        messages.append({"role": "assistant",
                         "content": "A couple of questions:\n"
                                    + "\n".join(f"{i}. {q}"
                                                for i, q in enumerate(questions, 1))})
        return _intake_result(messages, "Answer the questions in the box above, then press the "
                                        "button again.")
    if disposition == "unsupported":
        messages.append({"role": "assistant", "content": "That is not a decision between people."})
        return _intake_result(messages, "This tool plans decisions where other people's choices "
                                        "matter. Try describing one.")
    if disposition == "invalid":
        messages.append({"role": "assistant", "content": "I could not turn that into numbers."})
        return _intake_result(messages, "I could not turn that into clear scores. Say how much "
                                        "each outcome is worth to each side.")
    draft = _json.dumps({"template_id": parsed.get("template_id", "custom"),
                         "template_options": parsed.get("template_options") or {},
                         "game": parsed.get("game"),
                         "parameters": parsed.get("parameters") or {}},
                        indent=2, default=str)
    messages.append({"role": "assistant",
                     "content": "Draft ready. Check my summary below, then continue."})
    return _intake_result(messages, "", draft)


def game_intake_ui(history, text):
    state, chat, note, spec, _ = game_intake(history, text)
    return (state, chat, note, spec, *_cleared())


def draft_view(spec_json):
    summary, ok = plain_view.describe_draft(spec_json)
    return summary, gr.update(visible=ok)


def game_solve(spec_json, confirm, mediator_confirmed=False):
    if not confirm:
        result = {"schema_version": SCHEMA_VERSION, "disposition": "needs_confirmation"}
    else:
        try:
            spec = _json.loads(spec_json)
        except Exception as exc:
            result = {"schema_version": SCHEMA_VERSION, "disposition": "invalid",
                      "error": f"bad JSON: {exc}"}
        else:
            result = run_worker_request({"op": "solve_game", "game_spec": spec,
                                         "mediator_confirmed": bool(mediator_confirmed)})
    markdown = plain_view.describe_result(result, spec_json)
    html = share_report.render_report_html(markdown, autoprint=True) if markdown else ""
    return result, markdown, html


def game_solve_tagged(spec_json, mediator_confirmed=False):
    return _tagged(spec_json, game_solve(spec_json, True, mediator_confirmed))


def game_solve_show(tagged, spec_json):
    current = spec_fingerprint(spec_json)
    if not tagged or tagged["fingerprint"] != current:
        return (gr.skip(),) * 5
    result, markdown, html = tagged["value"]
    return result, markdown, html, "", ("" if _blank(spec_json) else current)


def game_estimate(spec_json, observations_json, method, confirm):
    if not confirm:
        return {"status": "needs_confirmation"}
    try:
        spec = _json.loads(spec_json)
        observations = _json.loads(observations_json)
    except Exception as exc:
        return {"status": "invalid", "error": f"bad JSON: {exc}"}
    return run_worker_request({"op": "estimate", "game_spec": spec,
                               "observations": observations, "method": method})


def game_sensitivity(spec_json, assignments_json, confirm):
    if not confirm:
        return {"status": "needs_confirmation"}
    try:
        spec = _json.loads(spec_json)
        assignments = _json.loads(assignments_json)
    except Exception as exc:
        return {"status": "invalid", "error": f"bad JSON: {exc}"}
    return run_worker_request({"op": "sensitivity", "game_spec": spec,
                               "assignments": assignments})


def game_simulate(spec_json, sim_json, confirm):
    if not confirm:
        return {"status": "needs_confirmation"}
    try:
        spec = _json.loads(spec_json)
        sim = _json.loads(sim_json)
    except Exception as exc:
        return {"status": "invalid", "error": f"bad JSON: {exc}"}
    return run_worker_request({"op": "simulate", "game_spec": spec, "simulation": sim})


def estimate_tagged(spec_json, observations_json, method, checked, confirmed_fp):
    confirm = _spec_confirmed(spec_json, checked, confirmed_fp)
    return _tagged(spec_json, game_estimate(spec_json, observations_json, method, confirm))


def sensitivity_tagged(spec_json, assignments_json, checked, confirmed_fp):
    confirm = _spec_confirmed(spec_json, checked, confirmed_fp)
    return _tagged(spec_json, game_sensitivity(spec_json, assignments_json, confirm))


def simulate_tagged(spec_json, sim_json, checked, confirmed_fp):
    confirm = _spec_confirmed(spec_json, checked, confirmed_fp)
    return _tagged(spec_json, game_simulate(spec_json, sim_json, confirm))


def approx_show(tagged, spec_json):
    if not tagged or tagged["fingerprint"] != spec_fingerprint(spec_json):
        return gr.skip()
    return tagged["value"]


def link_tagged(markdown):
    return {"markdown": markdown or "", "message": game_public_link(markdown)}


def link_show(tagged, markdown):
    if not tagged or tagged["markdown"] != (markdown or ""):
        return gr.skip()
    return tagged["message"]


def _case_store():
    key = case_store.load_key_from_env()
    path = os.environ.get("GT_CASE_DB_PATH", "game-cases.sqlite3")
    return case_store.CaseStore(path, key)


def case_save(title, spec_json, confirm):
    if not confirm:
        return {"status": "needs_confirmation"}
    try:
        spec = _json.loads(spec_json)
    except Exception as exc:
        return {"status": "invalid", "error": f"bad JSON: {exc}"}
    try:
        store = _case_store()
    except (case_store.CaseStoreError, sqlite3.Error, OSError) as exc:
        return _store_failure(exc)
    analysis_request = {"game_spec": spec}
    report = run_worker_request({"op": "solve_game", "game_spec": spec})
    try:
        return store.save_case(title, analysis_request, report, confirmed_by_user=True)
    except (case_store.CaseStoreError, sqlite3.Error, OSError) as exc:
        return _store_failure(exc)


def case_list():
    try:
        return _case_store().list_cases()
    except (case_store.CaseStoreError, sqlite3.Error, OSError) as exc:
        return [{"error": str(exc) or type(exc).__name__}]


def case_load(case_id):
    try:
        return _case_store().load_case(case_id)
    except case_store.CaseNotFound:
        return {"status": "not_found"}
    except (case_store.CaseStoreError, sqlite3.Error, OSError) as exc:
        return _store_failure(exc)


def case_delete(case_id, confirm):
    if not confirm:
        return {"status": "needs_confirmation"}
    try:
        return _case_store().delete_case(case_id, confirmed_by_user=True)
    except case_store.CaseNotFound:
        return {"status": "not_found"}
    except (case_store.CaseStoreError, sqlite3.Error, OSError) as exc:
        return _store_failure(exc)


def _case_choices(cases):
    return [(f"{c['title'] or 'Untitled plan'} ({c['created_at'][:10]})", c["case_id"])
            for c in cases if "title" in c]


def _plans_update(message):
    cases = case_list()
    choices = _case_choices(cases)
    errors = []
    for case in cases:
        if "error" in case:
            case_id = case.get("case_id")
            error = case["error"]
            errors.append(f"- {case_id}: {error}" if case_id else f"- {error}")
    if errors:
        warnings = "Saved plans with errors:\n" + "\n".join(errors)
        message = f"{message}\n\n{warnings}" if message else warnings
    return message, gr.update(choices=choices, value=None)


CONFIRM_BEFORE_SAVE = ("Confirm the details first: press “Looks right” (or tick the box under "
                       "What-if tools) while they are shown, then save.")


def plan_save(title, spec_json, confirmed_fp):
    if not _spec_confirmed(spec_json, True, confirmed_fp):
        return CONFIRM_BEFORE_SAVE, gr.skip()
    try:
        out = case_save((title or "").strip() or "Untitled plan", spec_json, True)
    except (case_store.CaseStoreError, sqlite3.Error) as exc:
        out = _store_failure(exc)
    if out.get("status") == "saved":
        return _plans_update("Saved.")
    return _plans_update(f"Could not save: {out.get('error') or out.get('status')}")


def plan_refresh():
    return _plans_update("")


def plan_delete(case_id):
    if not case_id:
        return _plans_update("Pick a saved plan first.")
    out = case_delete(case_id, True)
    return _plans_update("Deleted." if out.get("status") == "deleted"
                         else f"Could not delete: {out.get('error') or out.get('status')}")


def plan_open(case_id):
    keep = (gr.skip(),) * 11
    if not case_id:
        return ("Pick a saved plan first.", *keep)
    case = case_load(case_id)
    request = case.get("analysis_request")
    report = case.get("report")
    if (not isinstance(request, dict) or "game_spec" not in request
            or not isinstance(report, dict)):
        return (f"Could not open: {case.get('error') or case.get('status') or 'incomplete plan'}",
                *keep)
    spec_json = _json.dumps(request["game_spec"], indent=2)
    markdown = plain_view.describe_result(report, spec_json)
    html = share_report.render_report_html(markdown, autoprint=True) if markdown else ""
    solved = _tagged(spec_json, (report, markdown, html))
    return (f"Opened “{case['title']}”.", spec_json, markdown, html, report, "", None,
            False, "", solved, None, None)


GAME_EXAMPLES = [
    ["Two roommates each choose to do the dishes or leave them. If both do them, each scores 3. "
     "If only one does them, that one scores 1 and the other scores 4. If neither does them, "
     "both score 0."],
    ["Two coffee shops on one street each set a low or high price. Both high: each scores 5. "
     "Both low: each scores 2. One low and one high: the low one scores 7 and the high one 1."],
    ["Two friends each pick a restaurant, Thai or pizza. If they pick the same place, the one "
     "who loves that cuisine scores 3 and the other 2 (Thai: friend A scores 3, friend B 2; "
     "pizza: friend A 2, friend B 3). If they pick different places, both score 0."],
]


def build_general_tab():
    with gr.Tab("Plan a decision"):
        gr.Markdown("## Plan your move\nDescribe a situation where other people's choices "
                    "affect your result. Say what each side can do and how much each outcome "
                    "is worth to them (any simple scores work).")
        gr.Markdown("**Type the situation in the box, then press “Plan my move”.**",
                    elem_id="game-query-hint")
        with gr.Row(elem_id="game-query-bar"):
            game_box = gr.Textbox(label="What is going on?", show_label=False, lines=3,
                                  placeholder="What is going on?", container=False,
                                  scale=1, min_width=0, elem_id="game-input")
            intake_button = gr.Button("Plan my move", variant="primary", scale=0,
                                      min_width=140, elem_id="game-draft-button")
        gr.Examples(examples=GAME_EXAMPLES, inputs=[game_box],
                    example_labels=["Dishes", "Coffee prices", "Where to eat"],
                    cache_examples=False, elem_id="game-examples")
        game_history = gr.State([])
        solved_state = gr.State(None)
        approx_state = gr.State(None)
        link_state = gr.State(None)
        confirmed_state = gr.State("")
        chat = gr.Chatbot(label="Conversation", height=220, visible=False,
                          elem_id="game-conversation")
        note_md = gr.Markdown(elem_id="game-note")
        summary_md = gr.Markdown(elem_id="game-summary")
        mediator_box = gr.Checkbox(label="Someone can enforce a joint plan (mediator)",
                                   value=False, elem_id="game-mediator")
        solve_button = gr.Button("Looks right: show the best moves", variant="primary",
                                 visible=False, elem_id="game-solve-button")
        answer_md = gr.Markdown(elem_id="game-answer")
        answer_html = gr.Textbox(visible="hidden", show_label=False, container=False,
                                 elem_id="game-report-html")
        with gr.Row(visible=False, elem_id="game-share-bar") as share_row:
            game_pdf_button = gr.Button("Download PDF", elem_id="game-pdf-button")
            game_link_button = gr.Button("Get a link to share", elem_id="game-link-button")
            game_share_url = gr.Textbox(show_label=False, placeholder="Link appears here",
                                        elem_id="game-share-url")

        with gr.Accordion("Edit the details (advanced)", open=False):
            spec_box = gr.Textbox(label="Game details (JSON)", lines=10, elem_id="game-spec")
        with gr.Accordion("What-if tools (advanced)", open=False):
            confirm_box = gr.Checkbox(label="I have checked the details above",
                                      elem_id="game-confirm")
            obs_box = gr.Textbox(label="Observed choices (JSON)", lines=3, elem_id="game-obs")
            method_box = gr.Radio(["conditional_logit", "qre_marginals"],
                                  label="Estimation method", value="qre_marginals",
                                  elem_id="game-method")
            estimate_button = gr.Button("Estimate", elem_id="game-estimate-button")
            sensitivity_box = gr.Textbox(label="Try other scores (JSON)", lines=3,
                                         elem_id="game-sensitivity")
            sensitivity_button = gr.Button("Run sensitivity",
                                           elem_id="game-sensitivity-button")
            sim_box = gr.Textbox(label="Simulation settings (JSON)", lines=3,
                                 value='{"algorithm":"simulation","episodes":20,"seed":1,'
                                       '"policies":{}}', elem_id="game-sim")
            simulate_button = gr.Button("Simulate", elem_id="game-simulate-button")
            est_result = gr.JSON(label="Approximate result", elem_id="game-approx-result")
        with gr.Accordion("My saved plans", open=False):
            plan_title = gr.Textbox(label="Name this plan", elem_id="case-title")
            save_button = gr.Button("Save this plan", elem_id="case-save-button")
            plan_pick = gr.Dropdown(label="Saved plans", choices=[], elem_id="case-pick")
            with gr.Row():
                list_button = gr.Button("Show my saved plans", elem_id="case-list-button")
                load_button = gr.Button("Open", elem_id="case-load-button")
                delete_button = gr.Button("Delete", elem_id="case-delete-button")
            plan_status = gr.Markdown(elem_id="case-status")
        with gr.Accordion("Raw result (JSON)", open=False):
            evidence_box = gr.JSON(label="Analysis result", elem_id="game-result")

        stale_outputs = [answer_md, answer_html, evidence_box, game_share_url, est_result,
                         confirm_box, confirmed_state, solved_state, approx_state, link_state]
        intake_io = dict(fn=game_intake_ui, inputs=[game_history, game_box],
                         outputs=[game_history, chat, note_md, spec_box, *stale_outputs])
        intake_button.click(**intake_io)
        game_box.submit(**intake_io)
        spec_box.change(fn=spec_changed, inputs=[spec_box, solved_state],
                         outputs=[summary_md, solve_button, *stale_outputs])
        confirm_box.input(fn=confirm_spec, inputs=[confirm_box, spec_box],
                          outputs=[confirmed_state], show_progress="hidden")
        solve_button.click(fn=game_solve_tagged, inputs=[spec_box, mediator_box],
                           outputs=[solved_state]
                           ).success(fn=game_solve_show, inputs=[solved_state, spec_box],
                                     outputs=[evidence_box, answer_md, answer_html,
                                              game_share_url, confirmed_state])
        answer_md.change(fn=lambda md: gr.update(visible=bool(md)), inputs=[answer_md],
                         outputs=[share_row])
        game_pdf_button.click(fn=None, inputs=[answer_html], js=PDF_OPEN_JS)
        game_link_button.click(fn=link_tagged, inputs=[answer_md], outputs=[link_state]
                               ).success(fn=link_show, inputs=[link_state, answer_md],
                                         outputs=[game_share_url])
        estimate_button.click(fn=estimate_tagged,
                              inputs=[spec_box, obs_box, method_box, confirm_box,
                                      confirmed_state],
                              outputs=[approx_state]
                              ).success(fn=approx_show, inputs=[approx_state, spec_box],
                                        outputs=[est_result])
        sensitivity_button.click(fn=sensitivity_tagged,
                                 inputs=[spec_box, sensitivity_box, confirm_box,
                                         confirmed_state],
                                 outputs=[approx_state]
                                 ).success(fn=approx_show, inputs=[approx_state, spec_box],
                                           outputs=[est_result])
        simulate_button.click(fn=simulate_tagged,
                              inputs=[spec_box, sim_box, confirm_box, confirmed_state],
                              outputs=[approx_state]
                              ).success(fn=approx_show, inputs=[approx_state, spec_box],
                                        outputs=[est_result])
        save_button.click(fn=plan_save, inputs=[plan_title, spec_box, confirmed_state],
                          outputs=[plan_status, plan_pick])
        list_button.click(fn=plan_refresh, inputs=[], outputs=[plan_status, plan_pick])
        delete_button.click(fn=plan_delete, inputs=[plan_pick], outputs=[plan_status, plan_pick])
        load_button.click(fn=plan_open, inputs=[plan_pick],
                          outputs=[plan_status, spec_box, answer_md, answer_html, evidence_box,
                                   game_share_url, est_result, confirm_box, confirmed_state,
                                   solved_state, approx_state, link_state])


def _require_auth():
    import os
    username = os.environ.get("GT_APP_USERNAME", "")
    password = os.environ.get("GT_APP_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("GT_APP_USERNAME and GT_APP_PASSWORD are required: "
                           "refusing to start without credentials")
    return (username, password)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="social-media-gt advisor") as demo:
        with gr.Tabs():
            with gr.Tab("Paper scenarios"):
                gr.Markdown(BRAND, elem_id="brand")

                gr.Markdown("**Type the situation in the box, then press “Analyze”.**",
                            elem_id="query-hint")
                with gr.Row(elem_id="query-bar"):
                    message_box = gr.Textbox(
                        label="Describe the situation",
                        lines=3,
                        placeholder="Describe the situation",
                        show_label=False,
                        elem_id="query-input",
                        container=False,
                        scale=1,
                        min_width=0,
                    )
                    analyze_button = gr.Button("Analyze", variant="primary",
                                               elem_id="analyze-button",
                                               scale=0, min_width=112)

                state = gr.State([])
                gr.Examples(examples=EXAMPLES, inputs=[message_box],
                            example_labels=["Two factories", "Ten shareholders",
                                            "Twelve villages", "An oil cartel"],
                            elem_id="examples", cache_examples=False)

                history_box = gr.Chatbot(label="Conversation", height=220,
                                         elem_id="conversation")
                report_box = gr.Markdown(label="Full mathematical evidence",
                                         elem_id="evidence",
                                         latex_delimiters=LATEX_DELIMITERS)
                raw_box = gr.JSON(label="Raw evidence JSON", elem_id="raw-json")

                report_md_state = gr.State("")
                report_html_box = gr.Textbox(visible="hidden", show_label=False,
                                             container=False, elem_id="paper-report-html")
                with gr.Row(elem_id="share-bar"):
                    pdf_button = gr.Button("Download PDF", elem_id="paper-pdf-button")
                    link_button = gr.Button("Public link", elem_id="paper-link-button")
                    share_url_box = gr.Textbox(label="Public link",
                                               elem_id="paper-share-url")

                outputs = [history_box, state, report_box, raw_box, message_box,
                           report_md_state, report_html_box]
                analyze_button.click(fn=respond, inputs=[history_box, state, message_box],
                                     outputs=outputs)
                message_box.submit(fn=respond, inputs=[history_box, state, message_box],
                                   outputs=outputs)
                pdf_button.click(fn=None, inputs=[report_html_box], js=PDF_OPEN_JS)
                link_button.click(fn=paper_public_link, inputs=[report_md_state],
                                  outputs=[share_url_box])
            build_general_tab()
        demo.load(fn=None, js=DRAFT_JS)
    return demo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    build_demo().queue(default_concurrency_limit=8).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        auth=_require_auth(), css=CSS, footer_links=[])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
