"""
Detecting Irrelevant Comments in Indonesian Social Media Using IndoBERT-Relevancy
Streamlit Web Application
"""

from __future__ import annotations

import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predictor import Predictor
from utils import (
    extract_video_id,
    format_date,
    format_number,
    get_moderation_status,
    truncate_text,
)
from youtube_scraper import YouTubeScraper

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IndoBERT Relevancy Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* ── Base & typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Header banner ── */
    .app-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 60%, #60a5fa 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
        color: white !important;
    }
    .app-header p {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
        color: white !important;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.10); }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        color: #1e40af;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }
    .metric-card.relevant .metric-value { color: #16a34a; }
    .metric-card.irrelevant .metric-value { color: #dc2626; }
    .metric-card.pct-rel .metric-value { color: #0891b2; }
    .metric-card.pct-irr .metric-value { color: #9333ea; }

    /* ── Video info card ── */
    .video-info-card {
        background: #f8faff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .video-info-card h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e40af;
        margin: 0 0 0.6rem 0;
    }
    .video-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        font-size: 0.85rem;
        color: #475569;
    }
    .video-meta span { display: flex; align-items: center; gap: 0.35rem; }

    /* ── Section heading ── */
    .section-heading {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Table styling ── */
    .result-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    .result-table th {
        background: #f1f5f9;
        color: #374151;
        font-weight: 600;
        padding: 0.7rem 0.9rem;
        text-align: left;
        border-bottom: 2px solid #e2e8f0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .result-table td {
        padding: 0.65rem 0.9rem;
        border-bottom: 1px solid #f1f5f9;
        vertical-align: top;
    }
    .result-table tr:hover td { background: #f8faff; }
    .badge-relevant {
        display: inline-block;
        background: #dcfce7;
        color: #15803d;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-irrelevant {
        display: inline-block;
        background: #fee2e2;
        color: #b91c1c;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .conf-bar-wrap {
        background: #f1f5f9;
        border-radius: 20px;
        height: 6px;
        width: 80px;
        display: inline-block;
        margin-left: 6px;
        vertical-align: middle;
    }
    .conf-bar {
        height: 6px;
        border-radius: 20px;
    }

    /* ── Moderation summary ── */
    .mod-summary {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.8;
    }
    .mod-summary pre {
        margin: 0;
        white-space: pre-wrap;
        font-family: inherit;
        font-size: inherit;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #f0f4ff;
        border-right: 1px solid #dbeafe;
    }
    .sidebar-logo {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #dbeafe;
        margin-bottom: 1rem;
    }
    .sidebar-logo h2 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e40af;
        margin: 0.3rem 0 0.1rem 0;
    }
    .sidebar-logo p {
        font-size: 0.72rem;
        color: #64748b;
        margin: 0;
    }
    .status-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ok { background:#dcfce7; color:#15803d; }
    .status-err { background:#fee2e2; color:#b91c1c; }
    .status-warn { background:#fef9c3; color:#a16207; }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for key in ["results_df", "video_meta", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    predictor = Predictor()
    predictor.load()
    return predictor


predictor = load_predictor()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div style="font-size:2.5rem;">🔍</div>
            <h2>IndoBERT Relevancy</h2>
            <p>Comment Detection System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🤖 Model Information")
    model_status_html = (
        '<span class="status-badge status-ok">● Loaded</span>'
        if predictor.is_loaded
        else '<span class="status-badge status-err">● Error</span>'
    )
    st.markdown(
        f"""
        | | |
        |---|---|
        | **Model** | IndoBERT-Relevancy |
        | **Base** | apriandito/indobert-relevancy-classifier |
        | **Task** | Binary Classification |
        | **Status** | {model_status_html} |
        """,
        unsafe_allow_html=True,
    )

    if not predictor.is_loaded and predictor.error:
        st.error(predictor.error)

    st.divider()

    st.markdown("#### ⚙️ Settings")

    max_comments = st.selectbox(
        "Number of comments to fetch",
        options=[50, 100, 200, 500],
        index=1,
        help="More comments = longer analysis time",
    )

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help="Predictions below this threshold will be flagged as uncertain",
    )

    st.divider()

    st.markdown("#### 🔑 API Configuration")
    api_key = st.text_input(
        "YouTube Data API v3 Key",
        type="password",
        placeholder="AIza...",
        help="Get your key at console.cloud.google.com",
    )

    if api_key:
        st.markdown(
            '<span class="status-badge status-ok">● API Key Set</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-warn">● API Key Required</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<p style='font-size:0.72rem;color:#94a3b8;text-align:center;'>"
        "Penelitian — UPN Veteran Jawa Timur<br>"
        "IndoBERT-Relevancy Classifier</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>🔍 Detecting Irrelevant Comments in Indonesian Social Media</h1>
        <p>Using IndoBERT-Relevancy · Fine-tuned Binary Classifier · YouTube Comment Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────

with st.container():
    col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")

    with col_url:
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=xxxxxxxx",
            label_visibility="visible",
        )

    with col_btn:
        analyze_btn = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True,
            disabled=not predictor.is_loaded,
        )

    if not predictor.is_loaded:
        st.warning(
            "⚠️ Model not loaded. Please ensure all model files are present "
            "in the `model/` directory before running analysis.",
            icon="⚠️",
        )

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    # ── Validate inputs ──────────────────────────────────────────────────────
    if not api_key:
        st.error("❌ Please enter your YouTube Data API v3 key in the sidebar.")
        st.stop()

    if not youtube_url.strip():
        st.error("❌ Please enter a YouTube video URL.")
        st.stop()

    video_id = extract_video_id(youtube_url.strip())
    if not video_id:
        st.error(
            "❌ Invalid YouTube URL. Supported formats:\n"
            "- `https://www.youtube.com/watch?v=VIDEO_ID`\n"
            "- `https://youtu.be/VIDEO_ID`\n"
            "- `https://www.youtube.com/shorts/VIDEO_ID`"
        )
        st.stop()

    scraper = YouTubeScraper(api_key)

    # ── Fetch metadata ────────────────────────────────────────────────────────
    with st.status("Fetching video metadata…", expanded=True) as status:
        try:
            meta = scraper.get_video_metadata(video_id)
            st.write(f"✅ Video found: **{meta['title']}**")
        except ValueError as e:
            status.update(label="Failed", state="error")
            st.error(f"❌ {e}")
            st.stop()

        if meta.get("comments_disabled"):
            status.update(label="Failed", state="error")
            st.error("❌ Comments are disabled for this video.")
            st.stop()

        # ── Fetch comments ────────────────────────────────────────────────────
        st.write(f"Fetching up to **{max_comments}** comments…")
        try:
            raw_comments = scraper.get_comments(video_id, max_comments)
            st.write(f"✅ Retrieved **{len(raw_comments)}** comments.")
        except ValueError as e:
            status.update(label="Failed", state="error")
            st.error(f"❌ {e}")
            st.stop()

        if not raw_comments:
            status.update(label="Failed", state="error")
            st.error("❌ No comments found for this video.")
            st.stop()

        # ── Model prediction ──────────────────────────────────────────────────
        st.write("Running IndoBERT-Relevancy classifier…")
        progress_bar = st.progress(0)
        comment_texts = [c["text"] for c in raw_comments]
        total = len(comment_texts)

        def update_progress(current, total):
            progress_bar.progress(current / total)

        try:
            predictions = predictor.predict_batch(
                meta["title"], comment_texts, batch_size=16,
                progress_callback=update_progress,
            )
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"❌ Model inference failed: {e}")
            st.stop()

        progress_bar.progress(1.0)
        st.write("✅ Classification complete.")
        status.update(label="Analysis complete!", state="complete", expanded=False)

    # ── Build result DataFrame ────────────────────────────────────────────────
    rows = []
    for i, (comment, pred) in enumerate(zip(raw_comments, predictions), 1):
        rows.append(
            {
                "No": i,
                "Comment": comment["text"],
                "Author": comment["author"],
                "Prediction": pred["label"],
                "Confidence": pred["confidence"],
                "Confidence_pct": f"{pred['confidence']*100:.1f}%",
                "Likes": comment["like_count"],
                "Published": format_date(comment["published_at"]),
                "Low_Confidence": pred["confidence"] < confidence_threshold,
            }
        )

    df = pd.DataFrame(rows)
    st.session_state["results_df"] = df
    st.session_state["video_meta"] = meta
    st.session_state["analysis_done"] = True

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["analysis_done"] and st.session_state["results_df"] is not None:
    df: pd.DataFrame = st.session_state["results_df"]
    meta: dict = st.session_state["video_meta"]

    n_total = len(df)
    n_relevant = (df["Prediction"] == "Relevant").sum()
    n_irrelevant = (df["Prediction"] == "Irrelevant").sum()
    pct_rel = n_relevant / n_total if n_total else 0
    pct_irr = n_irrelevant / n_total if n_total else 0

    # ── Video information ─────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">📹 Video Information</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="video-info-card">
            <h3>{meta['title']}</h3>
            <div class="video-meta">
                <span>📺 {meta['channel_name']}</span>
                <span>📅 {format_date(meta['published_at'])}</span>
                <span>💬 {format_number(meta['comment_count'])} comments</span>
                <span>👀 {format_number(meta['view_count'])} views</span>
                <span>👍 {format_number(meta['like_count'])} likes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Statistics cards ──────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">📊 Analysis Statistics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "", str(n_total), "Total Comments"),
        (c2, "relevant", str(n_relevant), "Relevant"),
        (c3, "irrelevant", str(n_irrelevant), "Irrelevant"),
        (c4, "pct-rel", f"{pct_rel*100:.1f}%", "Relevant Rate"),
        (c5, "pct-irr", f"{pct_irr*100:.1f}%", "Irrelevant Rate"),
    ]
    for col, cls, val, label in cards:
        with col:
            st.markdown(
                f'<div class="metric-card {cls}">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visualizations ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">📈 Visualizations</div>', unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        pie_fig = px.pie(
            names=["Relevant", "Irrelevant"],
            values=[n_relevant, n_irrelevant],
            color=["Relevant", "Irrelevant"],
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#ef4444"},
            hole=0.45,
            title="Comment Distribution",
        )
        pie_fig.update_traces(textposition="inside", textinfo="percent+label")
        pie_fig.update_layout(
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            title_font_size=14,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with viz_col2:
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Relevant", "Irrelevant"],
                    y=[n_relevant, n_irrelevant],
                    marker_color=["#22c55e", "#ef4444"],
                    text=[n_relevant, n_irrelevant],
                    textposition="outside",
                    width=0.5,
                )
            ]
        )
        bar_fig.update_layout(
            title="Comments by Category",
            yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            title_font_size=14,
            margin=dict(t=50, b=30, l=30, r=20),
            yaxis=dict(gridcolor="#f1f5f9"),
            bargap=0.4,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ── Confidence distribution ───────────────────────────────────────────────
    with st.expander("📉 Confidence Score Distribution", expanded=False):
        conf_fig = px.histogram(
            df,
            x="Confidence",
            color="Prediction",
            nbins=20,
            barmode="overlay",
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#ef4444"},
            labels={"Confidence": "Confidence Score", "count": "Number of Comments"},
            title="Confidence Score Distribution by Prediction",
            opacity=0.75,
        )
        conf_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(conf_fig, use_container_width=True)

    # ── Filter & search ───────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">💬 Comment Results</div>', unsafe_allow_html=True)

    filter_col, search_col, export_col = st.columns([2, 4, 2])

    with filter_col:
        pred_filter = st.selectbox(
            "Filter by prediction",
            ["All", "Relevant", "Irrelevant"],
            label_visibility="collapsed",
        )

    with search_col:
        search_query = st.text_input(
            "Search",
            placeholder="🔎 Search comment…",
            label_visibility="collapsed",
        )

    with export_col:
        export_df = df[["No", "Comment", "Author", "Prediction", "Confidence_pct", "Likes", "Published"]].copy()
        csv_buffer = io.StringIO()
        export_df.rename(columns={"Confidence_pct": "Confidence"}).to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇ Download CSV",
            data=csv_buffer.getvalue(),
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered_df = df.copy()
    if pred_filter != "All":
        filtered_df = filtered_df[filtered_df["Prediction"] == pred_filter]
    if search_query.strip():
        mask = filtered_df["Comment"].str.contains(search_query.strip(), case=False, na=False)
        filtered_df = filtered_df[mask]

    st.caption(f"Showing **{len(filtered_df)}** of **{n_total}** comments")

    # ── Render table ──────────────────────────────────────────────────────────
    if filtered_df.empty:
        st.info("No comments match your filter/search criteria.")
    else:
        table_html = """
        <table class="result-table">
            <thead>
                <tr>
                    <th style="width:50px;">#</th>
                    <th>Comment</th>
                    <th style="width:120px;">Prediction</th>
                    <th style="width:130px;">Confidence</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in filtered_df.iterrows():
            pred = row["Prediction"]
            conf = row["Confidence"]
            badge_cls = "badge-relevant" if pred == "Relevant" else "badge-irrelevant"
            dot = "🟢" if pred == "Relevant" else "🔴"
            bar_color = "#22c55e" if pred == "Relevant" else "#ef4444"
            bar_width = int(conf * 80)
            uncertain_note = (
                ' <span style="color:#f59e0b;font-size:0.7rem;" title="Low confidence">⚠</span>'
                if row["Low_Confidence"]
                else ""
            )
            comment_text = truncate_text(row["Comment"], 140)

            table_html += f"""
            <tr>
                <td style="color:#94a3b8;font-size:0.78rem;">{row['No']}</td>
                <td style="max-width:480px;">
                    <div style="color:#1e293b;">{comment_text}</div>
                    <div style="color:#94a3b8;font-size:0.72rem;margin-top:2px;">👤 {row['Author']} · 📅 {row['Published']}</div>
                </td>
                <td>{dot} <span class="{badge_cls}">{pred}</span>{uncertain_note}</td>
                <td>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:500;">
                        {conf*100:.1f}%
                    </span>
                    <span class="conf-bar-wrap">
                        <span class="conf-bar" style="width:{bar_width}px;background:{bar_color};display:block;"></span>
                    </span>
                </td>
            </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    # ── Moderation summary ────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">📋 Comment Moderation Summary</div>', unsafe_allow_html=True)

    mod = get_moderation_status(pct_irr)
    avg_conf = df["Confidence"].mean()
    low_conf_count = df["Low_Confidence"].sum()

    summary_text = f"""  Total comments analyzed : {n_total}
  Relevant comments       : {n_relevant}
  Irrelevant comments     : {n_irrelevant}

  Irrelevant Ratio        : {pct_irr*100:.1f}%
  Avg Confidence Score    : {avg_conf*100:.1f}%
  Low Confidence Flags    : {low_conf_count}

  Moderation Status:
  {mod['emoji']} {mod['status']}

  Recommendation:
  {mod['recommendation']}"""

    st.markdown(
        f"""
        <div class="mod-summary" style="background:{mod['bg']};border:1.5px solid {mod['color']}40;">
            <pre style="color:{mod['color'][:7]};">{summary_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

elif not st.session_state["analysis_done"]:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;color:#94a3b8;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">📊</div>
                <h3 style="color:#475569;font-weight:600;">Ready to Analyze</h3>
                <p style="max-width:480px;margin:0 auto;font-size:0.9rem;">
                    Enter a YouTube video URL above and click <strong>Analyze</strong> to detect 
                    relevant and irrelevant comments using IndoBERT-Relevancy.
                </p>
                <div style="margin-top:1.5rem;display:flex;justify-content:center;gap:2rem;font-size:0.82rem;color:#64748b;">
                    <span>✅ Binary classification</span>
                    <span>📊 Interactive dashboard</span>
                    <span>💾 CSV export</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
