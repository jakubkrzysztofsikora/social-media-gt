"""Share-module fixtures: HTML rendering, public-link upload, error paths."""
import json
import shutil
import subprocess

from fractions import Fraction as F

import pytest

import share_report


SAMPLE_MD = """# Situation analysis

Player **A** faces $\\tau = 1/2$.

| strategy | payoff |
| --- | --- |
| restrict | 3/2 |

<script>alert("x")</script>
"""


def test_render_contains_math_table_and_cdn():
    html = share_report.render_report_html(SAMPLE_MD, title="t")
    assert "katex" in html
    assert "<table>" in html
    assert "$\\tau$" in html or "\\tau" in html
    assert "<strong>A</strong>" in html


def test_render_escapes_title():
    html = share_report.render_report_html("body", title='<img src=x onerror=alert(1)>')
    assert "<img src=x onerror" not in html.split("<body>")[0]
    assert "&lt;img" in html.split("<body>")[0]


def test_render_escapes_raw_html_in_markdown():
    """Regression (advisory B-1): LLM-generated markdown must never pass raw
    HTML through to the public share page."""
    md = 'hello <script>alert("xss")</script> <img src=x onerror=alert(1)>'
    html = share_report.render_report_html(md)
    assert '<script>alert("xss")' not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html


def test_render_autoprint_flag():
    plain = share_report.render_report_html("x")
    printable = share_report.render_report_html("x", autoprint=True)
    assert "window.print" not in plain
    assert "window.print" in printable


def test_public_link_requires_token(monkeypatch):
    monkeypatch.delenv("GT_SHARE_TOKEN", raising=False)
    out = share_report.public_share_link("md", token="")
    assert out["status"] == "error" and "GT_SHARE_TOKEN" in out["error"]


def test_public_link_uploads_and_returns_url(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"url": "https://share.jakub.team/abc123", "expires_in_days": 90}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeResponse()

    monkeypatch.setattr(share_report.urllib.request, "urlopen", fake_urlopen)
    out = share_report.public_share_link("# hi", token="tok")
    assert out == {"status": "ok", "url": "https://share.jakub.team/abc123",
                   "expires_in_days": 90}
    request = captured["request"]
    assert request.get_header("X-share-token") == "tok"
    assert b"<h1>hi</h1>" in request.data


def test_public_link_maps_http_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError("url", 401, "unauthorized", {}, None)

    monkeypatch.setattr(share_report.urllib.request, "urlopen", fake_urlopen)
    out = share_report.public_share_link("md", token="wrong")
    assert out["status"] == "error" and "401" in out["error"]


def test_public_link_maps_network_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(share_report.urllib.request, "urlopen", fake_urlopen)
    out = share_report.public_share_link("md", token="tok")
    assert out["status"] == "error" and "unreachable" in out["error"]

def test_autoprint_handles_cdn_load_error_and_timeout_once():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the embedded browser script")

    html = share_report.render_report_html("x", autoprint=True)
    harness = r"""
const fs = require("fs");
const vm = require("vm");
const html = JSON.parse(fs.readFileSync(0, "utf8"));
const inline = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)[1];
function run(event) {
  let timeout;
  let prints = 0;
  let renders = 0;
  const context = {
    window: {print() { prints++; }, renderMathInElement() { renders++; }},
    document: {body: {}},
    setTimeout(fn, delay) { timeout = {fn, delay}; return 1;},
    clearTimeout() {},
  };
  vm.runInNewContext(inline, context);
  if (event === "load") context.renderMath();
  if (event === "error") context.renderMath();
  if (event === "timeout") timeout.fn();
  if (event === "late") { timeout.fn(); context.renderMath(); }
  context.renderMath();
  return {prints, renders, delay: timeout.delay};
}
console.log(JSON.stringify(["load", "error", "timeout", "late"].map(run)));
"""
    result = subprocess.run(
        [node, "-e", harness], input=json.dumps(html), capture_output=True,
        text=True, timeout=5, check=True,
    )
    outcomes = json.loads(result.stdout)
    assert [outcome["prints"] for outcome in outcomes] == [1, 1, 1, 1]
    assert [outcome["renders"] for outcome in outcomes] == [1, 1, 1, 1]
    assert 0 < outcomes[2]["delay"] <= 10_000
