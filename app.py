"""
YouTube Comment Moderation Dashboard
Detecting Irrelevant Comments in Indonesian Social Media Using IndoBERT-Relevancy

"""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from predictor import Predictor, download_model_if_missing
from utils import (
    extract_video_id,
    format_date,
    format_number,
    get_health_score,
    get_moderation_status,
    get_youtube_thumbnail,
    truncate_text,
)
from youtube_scraper import YouTubeScraper

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Comment Moderation Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%);
        border-radius: 18px;
        padding: 1.9rem 2.4rem;
        margin-bottom: 1.6rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .app-header-left { display: flex; align-items: center; gap: 1rem; }
    .app-header-icon {
        font-size: 2.2rem;
        background: rgba(255,255,255,0.15);
        border-radius: 14px;
        width: 56px; height: 56px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .app-header h1 {
        font-size: 1.5rem; font-weight: 700; margin: 0;
        letter-spacing: -0.02em; color: white !important;
    }
    .app-header p {
        font-size: 0.88rem; opacity: 0.88; margin: 0.15rem 0 0 0;
        color: white !important;
    }
    .app-header-badge {
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 0.78rem;
        font-weight: 500;
        color: white;
        white-space: nowrap;
    }

    /* ── Generic card ── */
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Section heading ── */
    .section-heading {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.8rem 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-heading .line {
        flex: 1;
        height: 1px;
        background: #e2e8f0;
    }

    /* ── Video info card ── */
    .video-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        display: flex;
        gap: 1.3rem;
        align-items: stretch;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .video-thumb {
        width: 200px;
        min-width: 200px;
        border-radius: 10px;
        object-fit: cover;
        background: #f1f5f9;
    }
    .video-meta-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 0.5rem 0;
        line-height: 1.35;
    }
    .video-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.9rem 1.4rem;
        font-size: 0.83rem;
        color: #64748b;
        margin-top: 0.6rem;
    }
    .video-meta-row span { display: flex; align-items: center; gap: 0.35rem; }
    .video-meta-row b { color: #334155; font-weight: 600; }

    /* ── Health score card ── */
    .health-card {
        border-radius: 16px;
        padding: 1.8rem 2rem;
        display: flex;
        align-items: center;
        gap: 2.2rem;
        flex-wrap: wrap;
    }
    .health-score-circle {
        width: 120px; height: 120px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        background: white;
        box-shadow: inset 0 0 0 8px rgba(0,0,0,0.04);
    }
    .health-score-num { font-size: 2.1rem; font-weight: 800; line-height: 1; }
    .health-score-den { font-size: 0.72rem; color: #94a3b8; font-weight: 600; margin-top: 2px; }
    .health-info h3 {
        font-size: 1.15rem; font-weight: 700; margin: 0 0 0.25rem 0; color: #1e293b;
    }
    .health-info p { font-size: 0.85rem; color: #64748b; margin: 0; max-width: 480px; }
    .health-stats {
        display: flex;
        gap: 1.8rem;
        margin-left: auto;
        flex-wrap: wrap;
    }
    .health-stat { text-align: center; }
    .health-stat .val { font-size: 1.3rem; font-weight: 700; color: #1e293b; }
    .health-stat .lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }

    /* ── Status banner ── */
    .status-banner {
        border-radius: 14px;
        padding: 1.1rem 1.5rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    .status-banner .icon { font-size: 1.6rem; line-height: 1; margin-top: 1px; }
    .status-banner h4 { margin: 0 0 0.25rem 0; font-size: 0.98rem; font-weight: 700; }
    .status-banner p { margin: 0; font-size: 0.85rem; opacity: 0.9; }

    /* ── Metric mini cards ── */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card .metric-value { font-size: 1.7rem; font-weight: 700; line-height: 1.1; color: #1e40af; }
    .metric-card .metric-label {
        font-size: 0.72rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem;
    }
    .metric-card.relevant .metric-value { color: #16a34a; }
    .metric-card.irrelevant .metric-value { color: #dc2626; }

    /* ── Flagged / relevant comment list ── */
    .comment-item {
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.55rem;
        border-left: 4px solid transparent;
    }
    .comment-item.flagged { background: #fef2f2; border-left-color: #ef4444; }
    .comment-item.relevant { background: #f0fdf4; border-left-color: #22c55e; }
    .comment-item-text { font-size: 0.87rem; color: #1e293b; line-height: 1.45; }
    .comment-item-meta {
        font-size: 0.72rem; color: #94a3b8; margin-top: 0.35rem;
        display: flex; gap: 0.9rem; align-items: center;
    }
    .conf-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.1rem 0.5rem;
        border-radius: 8px;
    }
    .conf-chip.high-irr { background: #fee2e2; color: #b91c1c; }
    .conf-chip.high-rel { background: #dcfce7; color: #15803d; }

    /* ── Badges ── */
    .badge-relevant {
        display: inline-block; background: #dcfce7; color: #15803d;
        border-radius: 20px; padding: 0.15rem 0.7rem; font-weight: 600; font-size: 0.76rem;
    }
    .badge-irrelevant {
        display: inline-block; background: #fee2e2; color: #b91c1c;
        border-radius: 20px; padding: 0.15rem 0.7rem; font-weight: 600; font-size: 0.76rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #f5f8ff; border-right: 1px solid #dbeafe; }
    .sidebar-logo { text-align: center; padding: 0.4rem 0 1rem 0; border-bottom: 1px solid #dbeafe; margin-bottom: 1rem; }
    .sidebar-logo h2 { font-size: 1.02rem; font-weight: 700; color: #1e40af; margin: 0.3rem 0 0.1rem 0; }
    .sidebar-logo p { font-size: 0.7rem; color: #64748b; margin: 0; }
    .status-badge {
        display: inline-block; border-radius: 20px; padding: 0.15rem 0.6rem;
        font-size: 0.74rem; font-weight: 600;
    }
    .status-ok { background:#dcfce7; color:#15803d; }
    .status-err { background:#fee2e2; color:#b91c1c; }
    .status-warn { background:#fef9c3; color:#a16207; }

    /* ── Moderation report card ── */
    .report-card {
        border-radius: 14px;
        padding: 1.6rem 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.83rem;
        line-height: 1.85;
    }

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

for key in ["results_df", "video_meta", "video_id"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached) — pipeline untouched
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_URL = "https://drive.google.com/file/d/1at-RgOpN7LgBgwWPNOvkCqP7dIy19_CF/view?usp=sharing"
download_model_if_missing(DRIVE_URL)


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    predictor = Predictor()
    predictor.load()
    return predictor


predictor = load_predictor()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (simplified)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div style="font-size:2.3rem;">🛡️</div>
            <h2>Comment Moderation</h2>
            <p>Powered by IndoBERT-Relevancy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### System Status")

    model_badge = (
        '<span class="status-badge status-ok">● Model Ready</span>'
        if predictor.is_loaded
        else '<span class="status-badge status-err">● Model Error</span>'
    )
    st.markdown(model_badge, unsafe_allow_html=True)

    if not predictor.is_loaded and predictor.error:
        st.error(predictor.error)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### ⚙️ Analysis Settings")

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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🔑 YouTube API")
    api_key = st.text_input(
        "YouTube Data API v3 Key",
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed",
        help="Get your key at console.cloud.google.com",
    )

    if api_key:
        st.markdown('<span class="status-badge status-ok">● API Key Set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warn">● API Key Required</span>', unsafe_allow_html=True)

    with st.expander("ℹ️ Technical details"):
        st.markdown(
            """
            **Model:** IndoBERT-Relevancy
            **Base:** apriandito/indobert-relevancy-classifier
            **Task:** Binary text-pair classification
            **Research:** UPN Veteran Jawa Timur
            """
        )

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-header-icon">🛡️</div>
            <div>
                <h1>YouTube Comment Moderation Dashboard</h1>
                <p>Analyze comment relevance using IndoBERT-Relevancy</p>
            </div>
        </div>
        <div class="app-header-badge">🤖 AI-Powered Relevance Detection</div>
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
            placeholder="Paste a YouTube video link to analyze its comments…",
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
# ANALYSIS PIPELINE — pipeline logic untouched, only wrapped in same structure
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn:
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
    st.session_state["video_id"] = video_id
    st.session_state["analysis_done"] = True

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["analysis_done"] and st.session_state["results_df"] is not None:
    df: pd.DataFrame = st.session_state["results_df"]
    meta: dict = st.session_state["video_meta"]
    vid: str = st.session_state["video_id"]

    n_total = len(df)
    n_relevant = int((df["Prediction"] == "Relevant").sum())
    n_irrelevant = int((df["Prediction"] == "Irrelevant").sum())
    pct_rel = n_relevant / n_total if n_total else 0
    pct_irr = n_irrelevant / n_total if n_total else 0

    health = get_health_score(pct_irr)
    mod = get_moderation_status(pct_irr)

    # ── 1. Video information card ─────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">📹 Video Information<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    thumb_url = get_youtube_thumbnail(vid)
    st.markdown(
        f"""
        <div class="video-card">
            <img class="video-thumb" src="{thumb_url}" />
            <div>
                <div class="video-meta-title">{meta['title']}</div>
                <div class="video-meta-row">
                    <span>📺 <b>{meta['channel_name']}</b></span>
                    <span>📅 {format_date(meta['published_at'])}</span>
                </div>
                <div class="video-meta-row">
                    <span>👀 <b>{format_number(meta['view_count'])}</b> views</span>
                    <span>👍 <b>{format_number(meta['like_count'])}</b> likes</span>
                    <span>💬 <b>{format_number(meta['comment_count'])}</b> total comments</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Video Health Score ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">🩺 Video Health Score<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="health-card" style="background:{health['bg']};border:1px solid {health['color']}33;">
            <div class="health-score-circle" style="box-shadow: inset 0 0 0 8px {health['color']}26;">
                <div class="health-score-num" style="color:{health['color']};">{health['score']}</div>
                <div class="health-score-den">/ 100</div>
            </div>
            <div class="health-info">
                <h3>{health['tier_emoji']} {health['tier']}</h3>
                <p>{health['tier_msg']}</p>
            </div>
            <div class="health-stats">
                <div class="health-stat">
                    <div class="val">{n_total}</div>
                    <div class="lbl">Total</div>
                </div>
                <div class="health-stat">
                    <div class="val" style="color:#16a34a;">{n_relevant}</div>
                    <div class="lbl">Relevant</div>
                </div>
                <div class="health-stat">
                    <div class="val" style="color:#dc2626;">{n_irrelevant}</div>
                    <div class="lbl">Irrelevant</div>
                </div>
                <div class="health-stat">
                    <div class="val">{pct_rel*100:.0f}%</div>
                    <div class="lbl">Relevant Rate</div>
                </div>
                <div class="health-stat">
                    <div class="val">{pct_irr*100:.0f}%</div>
                    <div class="lbl">Irrelevant Rate</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3. Moderation status banner ───────────────────────────────────────
    st.markdown(
        '<div class="section-heading">🚦 Moderation Status<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    status_label = {"Low": "SAFE", "Moderate": "MODERATE", "High": "WARNING"}[mod["label"]]
    st.markdown(
        f"""
        <div class="status-banner" style="background:{mod['bg']};border:1px solid {mod['color']}33;">
            <div class="icon">{mod['emoji']}</div>
            <div>
                <h4 style="color:{mod['color']};">{status_label}</h4>
                <p style="color:#475569;">{mod['status']} <strong>Recommendation:</strong> {mod['recommendation']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 4 & 5. Flagged + Relevant comments ──────────────────────────────────
    st.markdown(
        '<div class="section-heading">🚩 Flagged & Relevant Comments<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    flag_col, rel_col = st.columns(2)

    with flag_col:
        st.markdown("**🔴 Top Flagged (Irrelevant)**")
        flagged = (
            df[df["Prediction"] == "Irrelevant"]
            .sort_values("Confidence", ascending=False)
            .head(20)
        )
        if flagged.empty:
            st.info("No irrelevant comments detected.")
        else:
            items_html = ""
            for _, row in flagged.iterrows():
                items_html += f"""
                <div class="comment-item flagged">
                    <div class="comment-item-text">{truncate_text(row['Comment'], 140)}</div>
                    <div class="comment-item-meta">
                        <span>👤 {row['Author']}</span>
                        <span class="conf-chip high-irr">{row['Confidence']*100:.1f}% confidence</span>
                    </div>
                </div>
                """
            st.markdown(
                f'<div style="max-height:480px;overflow-y:auto;padding-right:4px;">{items_html}</div>',
                unsafe_allow_html=True,
            )

    with rel_col:
        st.markdown("**🟢 Top Relevant**")
        relevant = (
            df[df["Prediction"] == "Relevant"]
            .sort_values("Confidence", ascending=False)
            .head(20)
        )
        if relevant.empty:
            st.info("No relevant comments detected.")
        else:
            items_html = ""
            for _, row in relevant.iterrows():
                items_html += f"""
                <div class="comment-item relevant">
                    <div class="comment-item-text">{truncate_text(row['Comment'], 140)}</div>
                    <div class="comment-item-meta">
                        <span>👤 {row['Author']}</span>
                        <span class="conf-chip high-rel">{row['Confidence']*100:.1f}% confidence</span>
                    </div>
                </div>
                """
            st.markdown(
                f'<div style="max-height:480px;overflow-y:auto;padding-right:4px;">{items_html}</div>',
                unsafe_allow_html=True,
            )

    # ── 6. Comment analytics ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">📈 Comment Analytics<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        pie_fig = px.pie(
            names=["Relevant", "Irrelevant"],
            values=[n_relevant, n_irrelevant],
            color=["Relevant", "Irrelevant"],
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#ef4444"},
            hole=0.5,
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
        conf_fig = px.histogram(
            df,
            x="Confidence",
            color="Prediction",
            nbins=20,
            barmode="overlay",
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#ef4444"},
            labels={"Confidence": "Confidence Score", "count": "Number of Comments"},
            title="Confidence Score Distribution",
            opacity=0.75,
        )
        conf_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            title_font_size=14,
            yaxis=dict(gridcolor="#f1f5f9"),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(conf_fig, use_container_width=True)

    with st.expander("👍 Like Count vs Prediction"):
        like_fig = px.box(
            df,
            x="Prediction",
            y="Likes",
            color="Prediction",
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#ef4444"},
            points="outliers",
            title="Like Count Distribution by Prediction",
        )
        like_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            yaxis=dict(gridcolor="#f1f5f9"),
            showlegend=False,
        )
        st.plotly_chart(like_fig, use_container_width=True)

    # ── 7. Comment Explorer ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">🔎 Comment Explorer<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    filter_col, search_col, sort_col = st.columns([2, 4, 2])

    with filter_col:
        pred_filter = st.selectbox(
            "Filter",
            ["All", "Relevant", "Irrelevant"],
            label_visibility="collapsed",
        )

    with search_col:
        search_query = st.text_input(
            "Search",
            placeholder="🔎 Search comment…",
            label_visibility="collapsed",
        )

    with sort_col:
        sort_option = st.selectbox(
            "Sort by",
            ["Newest first", "Confidence (high → low)", "Confidence (low → high)", "Most liked"],
            label_visibility="collapsed",
        )

    filtered_df = df.copy()
    if pred_filter != "All":
        filtered_df = filtered_df[filtered_df["Prediction"] == pred_filter]
    if search_query.strip():
        mask = filtered_df["Comment"].str.contains(search_query.strip(), case=False, na=False)
        filtered_df = filtered_df[mask]

    if sort_option == "Confidence (high → low)":
        filtered_df = filtered_df.sort_values("Confidence", ascending=False)
    elif sort_option == "Confidence (low → high)":
        filtered_df = filtered_df.sort_values("Confidence", ascending=True)
    elif sort_option == "Most liked":
        filtered_df = filtered_df.sort_values("Likes", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("No", ascending=True)

    st.caption(f"Showing **{len(filtered_df)}** of **{n_total}** comments")

    PAGE_SIZE = 25
    total_pages = max(1, (len(filtered_df) - 1) // PAGE_SIZE + 1)

    if "explorer_page" not in st.session_state:
        st.session_state["explorer_page"] = 1
    st.session_state["explorer_page"] = min(st.session_state["explorer_page"], total_pages)

    if filtered_df.empty:
        st.info("No comments match your filter/search criteria.")
    else:
        start = (st.session_state["explorer_page"] - 1) * PAGE_SIZE
        page_df = filtered_df.iloc[start : start + PAGE_SIZE]

        display_df = page_df[["No", "Comment", "Prediction", "Confidence_pct", "Likes"]].copy()
        display_df.columns = ["No", "Comment", "Prediction", "Confidence", "Likes"]

        def highlight_row(row):
            color = "#f0fdf4" if row["Prediction"] == "Relevant" else "#fef2f2"
            return [f"background-color: {color}"] * len(row)

        styled = display_df.style.apply(highlight_row, axis=1)

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "No": st.column_config.NumberColumn(width="small"),
                "Comment": st.column_config.TextColumn(width="large"),
                "Prediction": st.column_config.TextColumn(width="medium"),
                "Confidence": st.column_config.TextColumn(width="small"),
                "Likes": st.column_config.NumberColumn(width="small"),
            },
        )

        pg_col1, pg_col2, pg_col3 = st.columns([1, 2, 1])
        with pg_col1:
            if st.button("⬅ Previous", disabled=st.session_state["explorer_page"] <= 1):
                st.session_state["explorer_page"] -= 1
                st.rerun()
        with pg_col2:
            st.markdown(
                f"<p style='text-align:center;color:#64748b;font-size:0.85rem;padding-top:6px;'>"
                f"Page {st.session_state['explorer_page']} of {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with pg_col3:
            if st.button("Next ➡", disabled=st.session_state["explorer_page"] >= total_pages):
                st.session_state["explorer_page"] += 1
                st.rerun()

    # ── 9. Moderation report ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">📋 Moderation Report<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    avg_conf = df["Confidence"].mean()
    low_conf_count = int(df["Low_Confidence"].sum())
    moderation_required = "Yes" if pct_irr >= 0.10 else "No"

    report_text = f"""  Video Health Score      : {health['score']}/100  ({health['tier']})

  Total comments analyzed : {n_total}
  Relevant comments       : {n_relevant}
  Irrelevant comments     : {n_irrelevant}

  Relevant Rate            : {pct_rel*100:.1f}%
  Irrelevant Rate          : {pct_irr*100:.1f}%
  Avg Confidence Score     : {avg_conf*100:.1f}%
  Low Confidence Flags     : {low_conf_count}

  Recommendation:
  {mod['recommendation']}

  Moderation required      : {moderation_required}"""

    st.markdown(
        f"""
        <div class="report-card" style="background:{health['bg']};border:1.5px solid {health['color']}40;color:{health['color']};">
            <pre style="margin:0;white-space:pre-wrap;font-family:inherit;font-size:inherit;">{report_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 10. Export ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">⬇️ Export<div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    full_export = df[["No", "Comment", "Author", "Prediction", "Confidence_pct", "Likes", "Published"]].copy()
    full_export = full_export.rename(columns={"Confidence_pct": "Confidence"})

    flagged_export = full_export[full_export["Prediction"] == "Irrelevant"].copy()

    with exp_col1:
        buf = io.StringIO()
        full_export.to_csv(buf, index=False)
        st.download_button(
            "📄 Download All Comments (CSV)",
            data=buf.getvalue(),
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        buf2 = io.StringIO()
        flagged_export.to_csv(buf2, index=False)
        st.download_button(
            "🚩 Download Flagged Comments (CSV)",
            data=buf2.getvalue(),
            file_name="flagged_comments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col3:
        report_buf = io.StringIO()
        report_buf.write("YOUTUBE COMMENT MODERATION REPORT\n")
        report_buf.write("=" * 40 + "\n\n")
        report_buf.write(f"Video: {meta['title']}\n")
        report_buf.write(f"Channel: {meta['channel_name']}\n\n")
        report_buf.write(report_text.replace("  ", ""))
        st.download_button(
            "📊 Download Analysis Report (TXT)",
            data=report_buf.getvalue(),
            file_name="moderation_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

elif not st.session_state["analysis_done"]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8;">
            <div style="font-size:3.2rem;margin-bottom:0.8rem;">🛡️</div>
            <h3 style="color:#475569;font-weight:700;">Ready to Moderate</h3>
            <p style="max-width:480px;margin:0 auto;font-size:0.9rem;">
                Paste a YouTube video URL above and click <strong>Analyze</strong> to get a
                full comment relevance breakdown, health score, and moderation recommendation.
            </p>
            <div style="margin-top:1.4rem;display:flex;justify-content:center;gap:1.8rem;font-size:0.82rem;color:#64748b;flex-wrap:wrap;">
                <span>🩺 Health score</span>
                <span>🚩 Flagged comments</span>
                <span>📈 Analytics</span>
                <span>📋 Moderation report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
