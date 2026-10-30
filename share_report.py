"""Share support: self-contained HTML rendering, PDF print, public links.

PDF: the browser opens the rendered HTML and shows its print dialog
(save as PDF) - no extra server dependency. Public links: the rendered HTML
is uploaded to the share worker (share.jakub.team) under a random id; links
expire after 90 days.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from markdown_it import MarkdownIt

SHARE_WORKER_URL = "https://share.jakub.team"

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
        onerror="renderMath()"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMath()" onerror="renderMath()"></script>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 860px;
         margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; line-height: 1.55;
         background: #ffffff; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid #2c6fbb; padding-bottom: .3rem; }}
  h2 {{ font-size: 1.25rem; color: #2c6fbb; margin-top: 1.8rem; }}
  code, pre {{ font-family: 'IBM Plex Mono', Menlo, monospace; font-size: .88em;
              background: #f5f6f8; border-radius: 4px; }}
  pre {{ padding: .8rem; overflow-x: auto; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #d6d6d6; padding: .35rem .6rem; text-align: left; }}
  th {{ background: #eef2f7; }}
  footer {{ margin-top: 3rem; font-size: .8rem; color: #707070;
           border-top: 1px solid #e3e3e3; padding-top: .6rem; }}
  .katex {{ font-size: 1.02em; }}
  @media print {{
    body {{ margin: 0; }}
    h2 {{ break-after: avoid; }}
    table, pre {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
{body}
<footer>Generated {generated} · social-media-gt · exact-arithmetic evidence report</footer>
<script>
  {autoprint_timeout}
  var mathRendered = false;
  function renderMath() {{
    if (!mathRendered && window.renderMathInElement) {{
      try {{
        window.renderMathInElement(document.body, {{
          delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}},
          ],
          throwOnError: false,
        }});
        mathRendered = true;
      }} catch (error) {{}}
    }}
    {autoprint}
  }}
</script>
</body>
</html>
"""

_AUTOPRINT = """if (!printed) {{
    printed = true;
    clearTimeout(printTimeout);
    window.print();
  }}"""
_AUTOPRINT_TIMEOUT = """var printed = false;
  var printTimeout = setTimeout(function () {{ renderMath(); }}, 3000);"""


def render_report_html(markdown: str, title: str = "social-media-gt analysis",
                       autoprint: bool = False) -> str:
    """Render a report markdown into self-contained shareable HTML.

    Raw HTML in the markdown is escaped (html: False) - report text comes from
    LLM output over user conversations and is published on a public origin.
    """
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(markdown or "")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _HTML_TEMPLATE.format(
        title=_escape(title),
        body=body,
        generated=_escape(generated),
        autoprint=_AUTOPRINT if autoprint else "",
        autoprint_timeout=_AUTOPRINT_TIMEOUT if autoprint else "",
    )


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def public_share_link(markdown: str, title: str = "social-media-gt analysis",
                      endpoint: str | None = None, token: str | None = None,
                      timeout: float = 20.0) -> dict:
    """Upload the rendered HTML to the share worker; returns the public URL."""
    endpoint = endpoint or os.environ.get("GT_SHARE_ENDPOINT",
                                          f"{SHARE_WORKER_URL}/api/share")
    token = token or os.environ.get("GT_SHARE_TOKEN", "")
    if not token:
        return {"status": "error", "error": "GT_SHARE_TOKEN is not configured"}
    html = render_report_html(markdown, title=title)
    payload = json.dumps({"html": html}).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"content-type": "application/json", "x-share-token": token})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        return {"status": "error", "error": f"share worker returned {exc.code}: {detail}"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"status": "error", "error": f"share worker unreachable: {exc}"}
    return {"status": "ok", "url": data.get("url", ""),
            "expires_in_days": data.get("expires_in_days")}
