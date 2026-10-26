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
#query-input { flex: 1 1 0% !important; width: auto !important; min-width: 0 !important; }
#query-input textarea {
  box-sizing: border-box;
  width: 100%;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  color: var(--gt-text) !important;
  font-size: .95rem;
  line-height: 1.5;
  padding: 8px 6px;
  resize: none;
}
#query-input textarea:focus { outline: none !important; }
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
        return messages, messages, "", {}, ""
    messages.append({"role": "user", "content": str(text)})
    try:
        report = advisor.analyze_situation(messages)
    except Exception as exc:
        failure = f"**Analysis failed:** {exc}"
        messages.append({"role": "assistant", "content": f"Analysis failed: {exc}"})
        return messages, messages, failure, {}, ""
    markdown = report["markdown"]
    messages.append({"role": "assistant", "content": "Analysis complete. Full evidence is below."})
    raw = {key: value for key, value in report.items() if key != "markdown"}
    return messages, messages, markdown, raw, ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="social-media-gt advisor") as demo:
        gr.Markdown(BRAND, elem_id="brand")

        with gr.Row(elem_id="query-bar"):
            message_box = gr.Textbox(
                label="Describe the situation",
                lines=2,
                placeholder="Describe the situation",
                show_label=False,
                elem_id="query-input",
                container=False,
                scale=1,
                min_width=0,
            )
            analyze_button = gr.Button("Analyze", variant="primary", elem_id="analyze-button",
                                       scale=0, min_width=112)

        state = gr.State([])
        gr.Examples(examples=EXAMPLES, inputs=[message_box],
                    example_labels=["Two factories", "Ten shareholders", "Twelve villages",
                                    "An oil cartel"],
                    elem_id="examples", cache_examples=False)

        history_box = gr.Chatbot(label="Conversation", height=220, elem_id="conversation")
        report_box = gr.Markdown(label="Full mathematical evidence", elem_id="evidence",
                                 latex_delimiters=LATEX_DELIMITERS)
        raw_box = gr.JSON(label="Raw evidence JSON", elem_id="raw-json")

        outputs = [history_box, state, report_box, raw_box, message_box]
        analyze_button.click(fn=respond, inputs=[history_box, state, message_box], outputs=outputs)
        message_box.submit(fn=respond, inputs=[history_box, state, message_box], outputs=outputs)
    return demo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    build_demo().launch(server_name=args.host, server_port=args.port, share=args.share,
                        css=CSS, footer_links=[])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
