import html
import time
import streamlit as st

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import build_graph, seed_from_memory, save_results
from agent.state import ResearchState
import config

st.set_page_config(page_title="Competitive Intelligence Agent", layout="wide")

FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

STYLE = """
<style>
:root {
  --paper: #EEECE6;
  --ink: #1B2430;
  --teal: #2F6F63;
  --amber: #B8792B;
  --hairline: rgba(27,36,48,0.14);
}
.stApp { background-color: var(--paper); font-family: 'Public Sans', sans-serif; color: var(--ink); }
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink) !important; }

.block-container { max-width: 1080px; margin: 0 auto; padding-top: 3.5rem; }

.cia-eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--amber); margin-bottom: 6px; line-height: 1.8;
}
.cia-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.4rem; margin: 0; color: var(--ink); }
.cia-tagline { font-family: 'Public Sans', sans-serif; color: rgba(27,36,48,0.65); margin-top: 6px; font-size: 0.98rem; }

.stamp {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 3px;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.07em;
  text-transform: uppercase; font-weight: 600;
}
.stamp-confirmed { background: rgba(47,111,99,0.12); color: var(--teal); border: 1px solid var(--teal); }
.stamp-pending { background: rgba(27,36,48,0.05); color: rgba(27,36,48,0.45); border: 1px solid rgba(27,36,48,0.22); }
.stamp-insufficient { background: rgba(184,121,43,0.12); color: var(--amber); border: 1px solid var(--amber); }
.stamp-alert { background: rgba(178,58,52,0.10); color: #B23A34; border: 1px solid #B23A34; }

.cia-log {
  background: var(--ink); color: #E8E4D9; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem; line-height: 1.65; padding: 16px 18px; border-radius: 4px;
  height: 380px; overflow-y: auto; white-space: pre-wrap;
}
.cia-log .tag-plan { color: #8FB8D9; }
.cia-log .tag-found { color: #7FBFA8; }
.cia-log .tag-critic { color: #E0B15A; }
.cia-log .tag-stop { color: #D97D74; }
.cia-log .dim { color: rgba(232,228,217,0.4); }

.cia-status-card {
  background: #F8F6F1; border: 1px solid var(--hairline); border-radius: 4px;
  height: 380px; overflow-y: auto; padding: 6px 18px;
}
.cia-status-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 0; border-bottom: 1px solid var(--hairline);
}
.cia-status-row:last-child { border-bottom: none; }
.cia-status-label { font-size: 0.88rem; color: var(--ink); }

.cia-meta {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: rgba(27,36,48,0.55);
  border-top: 1px solid var(--hairline); margin-top: 24px; padding-top: 12px;
}

.cia-brief-eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--amber); margin-bottom: 4px;
}
.cia-brief-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2rem; margin: 0; color: var(--ink); }
.cia-brief-subline {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: rgba(27,36,48,0.5);
  margin-top: 8px; margin-bottom: 20px;
}
.cia-brief-divider { border: none; border-top: 1px solid var(--hairline); margin: 0 0 22px 0; }

.cia-flow-row {
  display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
  align-items: stretch; gap: 8px; margin: 4px 0;
}
.cia-flow-box {
  border: 1px solid var(--hairline); border-radius: 4px; padding: 10px 12px;
  background: #F8F6F1; text-align: center; min-width: 0;
}
.cia-flow-role {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.06em;
  text-transform: uppercase; font-weight: 600; color: var(--ink); overflow-wrap: break-word;
}
.cia-flow-model { font-size: 0.72rem; color: rgba(27,36,48,0.55); margin-top: 2px; }
.cia-flow-arrow {
  font-family: 'IBM Plex Mono', monospace; color: var(--amber); font-size: 1.1rem;
  display: flex; align-items: center; justify-content: center;
}
.cia-flow-note {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: rgba(27,36,48,0.5);
  margin-top: 10px;
}

div[data-testid="stExpander"] summary {
  font-family: 'Public Sans', sans-serif; font-weight: 600; font-size: 1rem; color: var(--ink);
}
div[data-testid="stExpander"] summary p { font-family: 'Public Sans', sans-serif; font-weight: 600; color: var(--ink); }

div[data-testid="stTextInput"] input {
  background: #F8F6F1; border: 1px solid var(--hairline); border-radius: 3px;
  color: var(--ink); font-family: 'Public Sans', sans-serif;
}
div[data-testid="stTextInput"] input:focus { border-color: var(--amber); box-shadow: none; }

div[data-testid="stButton"] button {
  font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.06em; text-transform: uppercase;
  font-size: 0.78rem; background: var(--ink); color: var(--paper); border: none; border-radius: 3px;
}
div[data-testid="stButton"] button:hover { background: #2A3648; color: var(--paper); }
div[data-testid="stDownloadButton"] button {
  font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.05em; text-transform: uppercase;
  font-size: 0.74rem; background: transparent; color: var(--ink); border: 1px solid var(--ink); border-radius: 3px;
}
div[data-testid="stDownloadButton"] button:hover { background: var(--ink); color: var(--paper); }
</style>
"""

st.markdown(FONTS + STYLE, unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center;">
    <div class="cia-eyebrow">Autonomous Research Pipeline</div>
    <h1 class="cia-title">Competitive Intelligence Agent</h1>
    <p class="cia-tagline">Enter a company or product. The agent plans its research, searches and
    verifies findings, and files a sourced brief covering what it does, funding &amp; ownership,
    recent news, competitors, and risks.</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

with st.expander("Architecture"):
    st.markdown(
        """
        <div class="cia-flow-row">
          <div class="cia-flow-box"><div class="cia-flow-role">Planner</div><div class="cia-flow-model">Llama &middot; NIM</div></div>
          <div class="cia-flow-arrow">&rarr;</div>
          <div class="cia-flow-box"><div class="cia-flow-role">Executor</div><div class="cia-flow-model">gpt-oss-120b &middot; Groq</div></div>
          <div class="cia-flow-arrow">&rarr;</div>
          <div class="cia-flow-box"><div class="cia-flow-role">Critic</div><div class="cia-flow-model">Gemini</div></div>
          <div class="cia-flow-arrow">&rarr;</div>
          <div class="cia-flow-box"><div class="cia-flow-role">Synthesizer</div><div class="cia-flow-model">Gemini</div></div>
        </div>
        <div class="cia-flow-note">Critic can send gaps back to Planner &mdash; up to 3 replan cycles. Synthesizer writes only from sourced scratchpad findings, never guesses.</div>
        """,
        unsafe_allow_html=True,
    )

with st.container(border=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        entity = st.text_input("Company or product", placeholder="e.g. Anthropic", label_visibility="collapsed")
    with col2:
        run_clicked = st.button("Run inquiry", use_container_width=True)

FIELD_LABELS = {
    "what_it_does": "What it does",
    "funding_ownership": "Funding & ownership",
    "recent_news": "Recent news",
    "competitors": "Competitors",
    "risks": "Risks",
}


def render_stamps(field_status: dict) -> str:
    chips = []
    for field, label in FIELD_LABELS.items():
        status = field_status.get(field)
        if status == "confirmed":
            cls, text = "stamp-confirmed", "confirmed"
        elif status == "insufficient information":
            cls, text = "stamp-insufficient", "insufficient"
        else:
            cls, text = "stamp-pending", "pending"
        chips.append(f'<span class="stamp {cls}">{html.escape(label)} &middot; {text}</span>')
    return "".join(chips)


def render_status_card(field_status: dict) -> str:
    rows = []
    for field, label in FIELD_LABELS.items():
        status = field_status.get(field)
        if status == "confirmed":
            cls, text = "stamp-confirmed", "confirmed"
        elif status == "insufficient information":
            cls, text = "stamp-insufficient", "insufficient"
        else:
            cls, text = "stamp-pending", "pending"
        rows.append(
            f'<div class="cia-status-row"><span class="cia-status-label">{html.escape(label)}</span>'
            f'<span class="stamp {cls}">{text}</span></div>'
        )
    return f'<div class="cia-status-card">{"".join(rows)}</div>'


def append_log(lines: list, entry: str, tag: str = "") -> None:
    cls = f"tag-{tag}" if tag else ""
    lines.append(f'<span class="{cls}">{html.escape(entry)}</span>')


if run_clicked and entity:
    app = build_graph()
    initial_state: ResearchState = {
        "entity": entity,
        "plan": [],
        "scratchpad": [],
        "critique": {"approved": False, "gaps": []},
        "replan_count": 0,
        "tool_call_count": 0,
        "tool_call_log": [],
        "start_time": time.time(),
        "report": "",
        "field_status": {},
        "stop_reason": "",
        "memory_note": "",
    }
    initial_state, memory_note = seed_from_memory(initial_state, entity)
    initial_state["memory_note"] = memory_note

    log_col, status_col = st.columns([3, 2])
    with log_col:
        st.markdown('<p style="text-align:center; font-weight:600;">Research log</p>', unsafe_allow_html=True)
        log_box = st.empty()
    with status_col:
        st.markdown('<p style="text-align:center; font-weight:600;">Dossier status</p>', unsafe_allow_html=True)
        status_box = st.empty()

    log_lines: list = []
    seen_plan_len = 0
    seen_scratchpad_len = 0
    seen_replan_count = 0
    final_state = None

    for step_state in app.stream(initial_state, stream_mode="values"):
        final_state = step_state

        if len(step_state["plan"]) != seen_plan_len and step_state["plan"]:
            append_log(log_lines, f"PLAN   {len(step_state['plan'])} sub-questions drafted", "plan")
            seen_plan_len = len(step_state["plan"])

        for entry in step_state["scratchpad"][seen_scratchpad_len:]:
            label = FIELD_LABELS.get(entry["field"], entry["field"])
            append_log(log_lines, f"FOUND  [{label}] via {entry['tool']}: {entry['source'][:70]}", "found")
        seen_scratchpad_len = len(step_state["scratchpad"])

        if step_state["replan_count"] != seen_replan_count:
            gaps = ", ".join(step_state["critique"]["gaps"])
            append_log(log_lines, f"CRITIC gaps in [{gaps}] -> replanning (cycle {step_state['replan_count']}/{config.MAX_REPLAN_CYCLES})", "critic")
            seen_replan_count = step_state["replan_count"]
        elif step_state["critique"]["approved"]:
            append_log(log_lines, "CRITIC all fields confirmed, approved", "critic")

        if step_state["stop_reason"]:
            append_log(log_lines, f"STOP   {step_state['stop_reason']}", "stop")

        log_box.markdown(f'<div class="cia-log">{"<br>".join(log_lines)}</div>', unsafe_allow_html=True)

        live_status = {e["field"]: "confirmed" for e in step_state["scratchpad"]}
        status_box.markdown(render_status_card(live_status), unsafe_allow_html=True)

    if final_state:
        save_results(entity, final_state)
        status_box.markdown(render_status_card(final_state["field_status"]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; font-weight:600;">Filed brief</p>', unsafe_allow_html=True)

        filed_date = time.strftime("%d %b %Y")
        source_count = len({e["source"] for e in final_state["scratchpad"]})
        report_safe = final_state["report"].replace("$", "\\$")
        full_markdown = (
            f"# Competitive Intelligence Brief: {final_state['entity']}\n\n"
            f"Filed {filed_date} &middot; {source_count} sources reviewed\n\n---\n\n"
            f"{final_state['report']}"
        )

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="cia-brief-eyebrow">Intelligence Brief</div>
                <h2 class="cia-brief-title">{html.escape(final_state['entity'])}</h2>
                <div class="cia-brief-subline">Filed {filed_date} &middot; {source_count} sources reviewed</div>
                <hr class="cia-brief-divider">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(report_safe)

            meta_bits = [
                f"entity: {html.escape(final_state['entity'])}",
                f"tool_calls: {final_state['tool_call_count']}",
                f"replans: {final_state['replan_count']}",
            ]
            if final_state["stop_reason"]:
                meta_bits.append(f"stop_reason: {html.escape(final_state['stop_reason'])}")
            st.markdown(f'<div class="cia-meta">{" &nbsp;&middot;&nbsp; ".join(meta_bits)}</div>', unsafe_allow_html=True)

            if final_state["memory_note"]:
                st.markdown(
                    f'<div style="margin-top:10px"><span class="stamp stamp-alert">memory</span> '
                    f'{html.escape(final_state["memory_note"])}</div>',
                    unsafe_allow_html=True,
                )

        st.download_button(
            "Download brief (.md)",
            data=full_markdown,
            file_name=f"{entity.strip().lower().replace(' ', '_')}_brief.md",
            mime="text/markdown",
        )
