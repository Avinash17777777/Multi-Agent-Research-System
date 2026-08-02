"""
Streamlit UI for the multi-agent research pipeline defined in pipeline.py.

Run with:
    streamlit run app.py

Expects to live in the same folder as pipeline.py, agents.py and tools.py.
"""

import io
import re
import time
import contextlib
import traceback

import streamlit as st

from pipeline import run_research_pipeline, extract_urls


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_STEPS = [
    ("1", "Search", "Finds recent, reliable sources on the topic"),
    ("2", "Read", "Scrapes the most promising source in depth"),
    ("3", "Write", "Drafts a report from everything gathered"),
    ("4", "Critique", "Reviews the draft and flags weak spots"),
]


# ---------------------------------------------------------------------------
# Light custom styling — kept minimal, works with Streamlit's own
# light/dark theme rather than fighting it.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

    h1 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 700 !important;
    }
    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        margin-bottom: 0.55rem;
    }
    .step-badge {
        flex-shrink: 0;
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        background: var(--primary-color, #C9932E);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .step-text b { font-size: 0.92rem; }
    .step-text span { font-size: 0.8rem; opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def text_of(value) -> str:
    """
    Normalize an agent/chain result into plain text.

    Depending on how a LangChain runnable ends (e.g. with or without a
    StrOutputParser), .invoke() can return a plain string or a message
    object with a .content attribute. Handling both keeps the UI from
    breaking either way.
    """
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    return str(value)


def slugify(text: str, max_len: int = 50) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:max_len] or "research"


def init_state():
    defaults = {
        "result": None,
        "logs": "",
        "error": None,
        "error_trace": None,
        "topic": "",
        "elapsed": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Research Agent")
    st.caption("Search → Read → Write → Critique")
    st.divider()

    for num, name, desc in PIPELINE_STEPS:
        st.markdown(
            f"""<div class="step-row">
                    <div class="step-badge">{num}</div>
                    <div class="step-text"><b>{name}</b><br><span>{desc}</span></div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    show_logs = st.checkbox(
        "Show run logs", value=False,
        help="Console output captured from the pipeline (same as the terminal version prints).",
    )

    if st.session_state.result is not None or st.session_state.error is not None:
        st.divider()
        if st.button("🗑️ Clear results", use_container_width=True):
            for key in ("result", "logs", "error", "error_trace", "elapsed"):
                st.session_state[key] = None if key != "logs" else ""
            st.rerun()


# ---------------------------------------------------------------------------
# Header + input
# ---------------------------------------------------------------------------
st.title("Multi-Agent Research Assistant")
st.write(
    "Enter a topic below. A search agent, reader agent, writer and critic will "
    "research it end-to-end and hand back a reviewed report."
)

with st.form("research_form", clear_on_submit=False):
    topic_input = st.text_input(
        "Research topic",
        placeholder="e.g. Recent advances in solid-state batteries",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🚀 Run Research", type="primary", use_container_width=True)

if submitted:
    if not topic_input or not topic_input.strip():
        st.warning("Enter a topic first.")
    else:
        topic = topic_input.strip()
        st.session_state.topic = topic
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.error_trace = None

        log_buffer = io.StringIO()
        start_time = time.time()

        with st.status(
            "Running search → read → write → critique... this can take a minute or two.",
            expanded=True,
        ) as status:
            try:
                with contextlib.redirect_stdout(log_buffer):
                    result = run_research_pipeline(topic)
                st.session_state.result = result
                st.session_state.elapsed = time.time() - start_time
                status.update(
                    label=f"Done in {st.session_state.elapsed:.0f}s",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.error_trace = traceback.format_exc()
                status.update(label="Pipeline failed", state="error", expanded=True)
            finally:
                st.session_state.logs = log_buffer.getvalue()

        # Rerun once so the sidebar (rendered earlier in this same pass, before
        # the pipeline finished) picks up the freshly-set session state too.
        st.rerun()


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------
if st.session_state.error:
    st.error(f"The pipeline hit an error: {st.session_state.error}")
    with st.expander("Traceback"):
        st.code(st.session_state.error_trace or "", language="text")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    if st.session_state.elapsed:
        st.caption(
            f"✅ Research on **{st.session_state.topic}** completed in "
            f"{st.session_state.elapsed:.1f}s"
        )

    sources = extract_urls(result.get("search_results", "") or "")
    if sources:
        with st.expander(f"🔗 {len(sources)} source(s) discovered during search"):
            for url in sources:
                st.markdown(f"- [{url}]({url})")

    tab_report, tab_critic, tab_search, tab_scraped = st.tabs(
        ["📄 Report", "🧐 Critic Feedback", "🔎 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        report_text = text_of(result.get("report", ""))
        st.markdown(report_text)
        st.download_button(
            "⬇️ Download report (.md)",
            data=report_text,
            file_name=f"report_{slugify(st.session_state.topic)}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tab_critic:
        st.markdown(text_of(result.get("feedback", "")))

    with tab_search:
        st.text_area(
            "Raw search results",
            value=result.get("search_results", "") or "No search results captured.",
            height=350,
            disabled=True,
            label_visibility="collapsed",
            key="search_results_box",
        )

    with tab_scraped:
        st.text_area(
            "Raw scraped content",
            value=result.get("scraped_content", "") or "No scraped content captured.",
            height=350,
            disabled=True,
            label_visibility="collapsed",
            key="scraped_content_box",
        )

if show_logs and st.session_state.logs:
    with st.expander("🖥️ Console logs from this run", expanded=True):
        st.code(st.session_state.logs, language="text")

if result is None and not st.session_state.error:
    st.info("Enter a topic above and click **Run Research** to get started.")
