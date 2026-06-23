"""
RelevancyTube
Context-Aware YouTube Comment Relevance Detection System

"""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    page_title="RelevancyTube · Context-Aware YouTube Comment Relevance Detection System",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (YouTube palette: #FF0000 red · #0F0F0F dark · #FFFFFF white · #F2F2F2 light)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }

.main .block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1300px; }

/* ── Header ── */
.cg-header {
    background: #0F0F0F;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
}
.cg-header-left { display: flex; align-items: center; gap: 1rem; }
.cg-logo {
    background: #FF0000; border-radius: 12px; width: 52px; height: 52px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; flex-shrink: 0;
}
.cg-header h1 { font-size: 1.45rem; font-weight: 700; margin: 0; color: #FFFFFF !important; letter-spacing: -.01em; }
.cg-header p  { font-size: 0.82rem; margin: 0.2rem 0 0 0; color: #AAAAAA !important; }
.cg-pill {
    background: #FF0000; color: white; border-radius: 20px;
    padding: 0.35rem 1rem; font-size: 0.75rem; font-weight: 500; white-space: nowrap;
}

/* ── Section heading ── */
.cg-section {
    font-size: 0.92rem; font-weight: 700; color: #0F0F0F;
    margin: 1.6rem 0 0.8rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid #FF0000; display: flex; align-items: center; gap: 0.5rem;
}

/* ── Video card ── */
.cg-video-card {
    background: #FFFFFF; border: 1px solid #E5E5E5; border-radius: 14px;
    display: flex; gap: 1.2rem; padding: 1.1rem; align-items: flex-start;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.cg-thumb { width: 200px; min-width: 200px; border-radius: 10px; object-fit: cover; background: #F2F2F2; }
.cg-video-title { font-size: 1.05rem; font-weight: 700; color: #0F0F0F; margin: 0 0 0.55rem 0; line-height: 1.35; }
.cg-video-meta { display: flex; flex-wrap: wrap; gap: 0.7rem 1.4rem; font-size: 0.82rem; color: #606060; margin-top: 0.4rem; }
.cg-video-meta span { display: flex; align-items: center; gap: 0.3rem; }

/* ── Health score ── */
.cg-health {
    border-radius: 14px; padding: 1.5rem 1.8rem;
    display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
}
.cg-score-ring {
    width: 110px; height: 110px; border-radius: 50%; flex-shrink: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: white;
}
.cg-score-num { font-size: 2rem; font-weight: 800; line-height: 1; }
.cg-score-den { font-size: 0.68rem; color: #909090; font-weight: 600; }
.cg-health-info h3 { font-size: 1.1rem; font-weight: 700; margin: 0 0 0.2rem 0; color: #0F0F0F; }
.cg-health-info p  { font-size: 0.83rem; color: #606060; margin: 0; }
.cg-health-stats { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-left: auto; }
.cg-hstat .val { font-size: 1.35rem; font-weight: 700; color: #0F0F0F; }
.cg-hstat .lbl { font-size: 0.67rem; color: #909090; text-transform: uppercase; letter-spacing: .04em; }

/* ── Status banner ── */
.cg-banner {
    border-radius: 12px; padding: 1rem 1.4rem;
    display: flex; align-items: flex-start; gap: 0.9rem;
}
.cg-banner h4 { margin: 0 0 0.2rem 0; font-size: 0.95rem; font-weight: 700; }
.cg-banner p  { margin: 0; font-size: 0.83rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #F9F9F9; border-right: 1px solid #E5E5E5; }
.cg-sb-logo { text-align: center; padding: 0.3rem 0 0.9rem 0; border-bottom: 1px solid #E5E5E5; margin-bottom: 0.9rem; }
.cg-sb-logo h2 { font-size: 1rem; font-weight: 700; color: #FF0000; margin: 0.3rem 0 0.05rem 0; }
.cg-sb-logo p  { font-size: 0.68rem; color: #909090; margin: 0; }
.cg-badge { display: inline-block; border-radius: 20px; padding: .12rem .55rem; font-size: .72rem; font-weight: 600; }
.cg-ok   { background:#dcfce7; color:#15803d; }
.cg-err  { background:#fee2e2; color:#b91c1c; }
.cg-warn { background:#fef9c3; color:#a16207; }

/* ── Report card ── */
.cg-report {
    border-radius: 14px; padding: 1.5rem 1.8rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.9;
}

#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for _k in ["results_df", "video_meta", "video_id", "analysis_done", "explorer_page"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
if st.session_state["analysis_done"] is None:
    st.session_state["analysis_done"] = False
if st.session_state["explorer_page"] is None:
    st.session_state["explorer_page"] = 1

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — pipeline untouched
# ─────────────────────────────────────────────────────────────────────────────

DRIVE_URL = "https://drive.google.com/file/d/1at-RgOpN7LgBgwWPNOvkCqP7dIy19_CF/view?usp=sharing"
download_model_if_missing(DRIVE_URL)


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    p = Predictor()
    p.load()
    return p


predictor = load_predictor()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="cg-sb-logo">
        <div style="font-size:2rem;">▶</div>
        <h2>RelevancyTube</h2>
        <p>Context-Aware YouTube Comment Relevance Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    model_badge = (
        '<span class="cg-badge cg-ok">● Model Ready</span>'
        if predictor.is_loaded
        else '<span class="cg-badge cg-err">● Model Error</span>'
    )
    st.markdown("**System Status**")
    st.markdown(model_badge, unsafe_allow_html=True)
    if not predictor.is_loaded and predictor.error:
        st.error(predictor.error)

    st.divider()
    st.markdown("**⚙️ Settings**")

    max_comments = st.selectbox(
        "Comments to fetch",
        [50, 100, 200, 500],
        index=1,
        help="More comments = longer analysis time",
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99, value=0.70, step=0.01,
        help="Predictions below this are flagged as uncertain",
    )

    st.divider()
    st.markdown("**🔑 YouTube API**")
    api_key = st.text_input(
        "API Key", type="password", placeholder="AIza…",
        label_visibility="collapsed",
    )
    if api_key:
        st.markdown('<span class="cg-badge cg-ok">● API Key Set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="cg-badge cg-warn">● API Key Required</span>', unsafe_allow_html=True)

    with st.expander("ℹ️ About"):
        st.markdown("""
**RelevancyTube** detects irrelevant comments in Indonesian YouTube discussions using a fine-tuned IndoBERT binary classifier.

**Research:** UPN Veteran Jawa Timur
**Model:** IndoBERT-Relevancy
**Task:** Relevant / Irrelevant classification
        """)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="cg-header">
    <div class="cg-header-left">
        <div class="cg-logo">▶</div>
        <div>
            <h1>RelevancyTube</h1>
            <p>Context-Aware YouTube Comment Relevance Detection System · Powered by IndoBERT-Relevancy</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────

col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_url:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="Paste a YouTube video link to analyze its comments…",
        label_visibility="collapsed",
    )
with col_btn:
    analyze_btn = st.button(
        "▶ Analyze", type="primary",
        use_container_width=True,
        disabled=not predictor.is_loaded,
    )

if not predictor.is_loaded:
    st.warning("⚠️ Model not loaded. Check the `model/` directory.", icon="⚠️")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE  — logic identical to original
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
            "❌ Invalid YouTube URL. Supported:\n"
            "- `https://www.youtube.com/watch?v=ID`\n"
            "- `https://youtu.be/ID`\n"
            "- `https://www.youtube.com/shorts/ID`"
        )
        st.stop()

    scraper = YouTubeScraper(api_key)

    with st.status("Fetching video metadata…", expanded=True) as _status:
        try:
            meta = scraper.get_video_metadata(video_id)
            st.write(f"✅ Video found: **{meta['title']}**")
        except ValueError as e:
            _status.update(label="Failed", state="error")
            st.error(f"❌ {e}")
            st.stop()

        if meta.get("comments_disabled"):
            _status.update(label="Failed", state="error")
            st.error("❌ Comments are disabled for this video.")
            st.stop()

        st.write(f"Fetching up to **{max_comments}** comments…")
        try:
            raw_comments = scraper.get_comments(video_id, max_comments)
            st.write(f"✅ Retrieved **{len(raw_comments)}** comments.")
        except ValueError as e:
            _status.update(label="Failed", state="error")
            st.error(f"❌ {e}")
            st.stop()

        if not raw_comments:
            _status.update(label="Failed", state="error")
            st.error("❌ No comments found for this video.")
            st.stop()

        st.write("Running IndoBERT-Relevancy classifier…")
        pb = st.progress(0)
        comment_texts = [c["text"] for c in raw_comments]

        def _upd(cur, tot): pb.progress(cur / tot)

        try:
            predictions = predictor.predict_batch(
                meta["title"], comment_texts, batch_size=16,
                progress_callback=_upd,
            )
        except Exception as e:
            _status.update(label="Failed", state="error")
            st.error(f"❌ Model inference failed: {e}")
            st.stop()

        pb.progress(1.0)
        st.write("✅ Classification complete.")
        _status.update(label="Analysis complete!", state="complete", expanded=False)

    rows = []
    for i, (c, p) in enumerate(zip(raw_comments, predictions), 1):
        rows.append({
            "No": i,
            "Comment": c["text"],
            "Author": c["author"],
            "Prediction": p["label"],
            "Confidence": p["confidence"],
            "Confidence_pct": f"{p['confidence']*100:.1f}%",
            "Likes": c["like_count"],
            "Published": format_date(c["published_at"]),
            "Low_Confidence": p["confidence"] < confidence_threshold,
        })

    df = pd.DataFrame(rows)
    st.session_state["results_df"] = df
    st.session_state["video_meta"] = meta
    st.session_state["video_id"] = video_id
    st.session_state["analysis_done"] = True
    st.session_state["explorer_page"] = 1

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["analysis_done"] and st.session_state["results_df"] is not None:
    df: pd.DataFrame = st.session_state["results_df"]
    meta: dict       = st.session_state["video_meta"]
    vid: str         = st.session_state["video_id"]

    n_total     = len(df)
    n_relevant  = int((df["Prediction"] == "Relevant").sum())
    n_irrelevant= int((df["Prediction"] == "Irrelevant").sum())
    pct_rel     = n_relevant  / n_total if n_total else 0
    pct_irr     = n_irrelevant/ n_total if n_total else 0
    avg_conf    = float(df["Confidence"].mean())
    low_conf    = int(df["Low_Confidence"].sum())

    health = get_health_score(pct_irr)
    mod    = get_moderation_status(pct_irr)

    # ── 1. Video Card ─────────────────────────────────────────────────────
    st.markdown('<div class="cg-section">📹 Video Information</div>', unsafe_allow_html=True)

    thumb = get_youtube_thumbnail(vid)
    left, right = st.columns([1, 3])
    with left:
        st.image(thumb, use_container_width=True)
    with right:
        st.markdown(f"### {meta['title']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("📺 Channel",  meta["channel_name"])
        c2.metric("📅 Published", format_date(meta["published_at"]))
        c3.metric("💬 Comments",  format_number(meta["comment_count"]))
        c4, c5, _ = st.columns(3)
        c4.metric("👀 Views", format_number(meta["view_count"]))
        c5.metric("👍 Likes", format_number(meta["like_count"]))

    st.divider()

    # ── 2. Health Score ───────────────────────────────────────────────────
    st.markdown('<div class="cg-section">🩺 Video Health Score</div>', unsafe_allow_html=True)

    # Score ring via HTML (safe — no user content injected)
    st.markdown(f"""
    <div class="cg-health" style="background:{health['bg']};border:1px solid {health['color']}30;">
        <div class="cg-score-ring" style="box-shadow:inset 0 0 0 9px {health['color']}25;">
            <div class="cg-score-num" style="color:{health['color']};">{health['score']}</div>
            <div class="cg-score-den">/ 100</div>
        </div>
        <div class="cg-health-info">
            <h3>{health['tier_emoji']} {health['tier']}</h3>
            <p>{health['tier_msg']}</p>
        </div>
        <div class="cg-health-stats">
            <div class="cg-hstat"><div class="val">{n_total}</div><div class="lbl">Total</div></div>
            <div class="cg-hstat"><div class="val" style="color:#16a34a;">{n_relevant}</div><div class="lbl">Relevant</div></div>
            <div class="cg-hstat"><div class="val" style="color:#FF0000;">{n_irrelevant}</div><div class="lbl">Irrelevant</div></div>
            <div class="cg-hstat"><div class="val">{pct_rel*100:.0f}%</div><div class="lbl">Rel Rate</div></div>
            <div class="cg-hstat"><div class="val">{pct_irr*100:.0f}%</div><div class="lbl">Irr Rate</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 3. Moderation Status ──────────────────────────────────────────────
    st.markdown('<div class="cg-section">🚦 Moderation Status</div>', unsafe_allow_html=True)

    status_label = {"Low": "✅ SAFE", "Moderate": "⚠️ MODERATE", "High": "🚨 WARNING"}[mod["label"]]
    st.markdown(f"""
    <div class="cg-banner" style="background:{mod['bg']};border:1px solid {mod['color']}30;">
        <div>
            <h4 style="color:{mod['color']};">{status_label}</h4>
            <p style="color:#444;">{mod['status']} &nbsp;·&nbsp; <em>{mod['recommendation']}</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 4&5. Flagged + Relevant (native Streamlit — no raw HTML loops) ────
    st.markdown('<div class="cg-section">🚩 Flagged & Relevant Comments</div>', unsafe_allow_html=True)

    flagged_df = (
        df[df["Prediction"] == "Irrelevant"]
        .sort_values("Confidence", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    relevant_df = (
        df[df["Prediction"] == "Relevant"]
        .sort_values("Confidence", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    flag_col, rel_col = st.columns(2)

    with flag_col:
        st.markdown("**🔴 Top Flagged (Irrelevant)**")
        if flagged_df.empty:
            st.success("No irrelevant comments detected.")
        else:
            for _, row in flagged_df.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"<span style='font-size:.87rem;color:#0F0F0F;'>{truncate_text(row['Comment'], 150)}</span>",
                        unsafe_allow_html=True,
                    )
                    badge_col, conf_col = st.columns([2, 1])
                    badge_col.caption(f"👤 {row['Author']}")
                    conf_col.markdown(
                        f"<span style='background:#fee2e2;color:#b91c1c;border-radius:8px;"
                        f"padding:.1rem .5rem;font-size:.73rem;font-weight:700;font-family:monospace;'>"
                        f"{row['Confidence']*100:.1f}% conf</span>",
                        unsafe_allow_html=True,
                    )

    with rel_col:
        st.markdown("**🟢 Top Relevant**")
        if relevant_df.empty:
            st.warning("No relevant comments detected.")
        else:
            for _, row in relevant_df.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"<span style='font-size:.87rem;color:#0F0F0F;'>{truncate_text(row['Comment'], 150)}</span>",
                        unsafe_allow_html=True,
                    )
                    badge_col, conf_col = st.columns([2, 1])
                    badge_col.caption(f"👤 {row['Author']}")
                    conf_col.markdown(
                        f"<span style='background:#dcfce7;color:#15803d;border-radius:8px;"
                        f"padding:.1rem .5rem;font-size:.73rem;font-weight:700;font-family:monospace;'>"
                        f"{row['Confidence']*100:.1f}% conf</span>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── 6. Analytics ─────────────────────────────────────────────────────
    st.markdown('<div class="cg-section">📊 Comment Analytics</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        pie = px.pie(
            names=["Relevant", "Irrelevant"],
            values=[n_relevant, n_irrelevant],
            color=["Relevant", "Irrelevant"],
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#FF0000"},
            hole=0.50,
            title="Comment Distribution",
        )
        pie.update_traces(textposition="inside", textinfo="percent+label")
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Roboto", title_font_size=14,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(pie, use_container_width=True)

    with v2:
        hist = px.histogram(
            df, x="Confidence", color="Prediction", nbins=20, barmode="overlay",
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#FF0000"},
            labels={"Confidence": "Confidence Score"},
            title="Confidence Score Distribution",
            opacity=0.78,
        )
        hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Roboto", title_font_size=14,
            yaxis=dict(gridcolor="#F2F2F2"),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(hist, use_container_width=True)

    with st.expander("👍 Like Count vs Prediction"):
        box = px.box(
            df, x="Prediction", y="Likes", color="Prediction",
            color_discrete_map={"Relevant": "#22c55e", "Irrelevant": "#FF0000"},
            points="outliers", title="Like Count by Prediction",
        )
        box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Roboto", yaxis=dict(gridcolor="#F2F2F2"), showlegend=False,
        )
        st.plotly_chart(box, use_container_width=True)

    st.divider()

    # ── 7. Comment Explorer ───────────────────────────────────────────────
    st.markdown('<div class="cg-section">🔎 Comment Explorer</div>', unsafe_allow_html=True)

    f_col, s_col, sort_col = st.columns([2, 4, 2])
    with f_col:
        pred_filter = st.selectbox("Filter", ["All", "Relevant", "Irrelevant"], label_visibility="collapsed")
    with s_col:
        search_q = st.text_input("Search", placeholder="🔎 Search comment…", label_visibility="collapsed")
    with sort_col:
        sort_opt = st.selectbox(
            "Sort", ["Default", "Confidence ↓", "Confidence ↑", "Most Liked"],
            label_visibility="collapsed",
        )

    fdf = df.copy()
    if pred_filter != "All":
        fdf = fdf[fdf["Prediction"] == pred_filter]
    if search_q.strip():
        fdf = fdf[fdf["Comment"].str.contains(search_q.strip(), case=False, na=False)]
    if sort_opt == "Confidence ↓":
        fdf = fdf.sort_values("Confidence", ascending=False)
    elif sort_opt == "Confidence ↑":
        fdf = fdf.sort_values("Confidence", ascending=True)
    elif sort_opt == "Most Liked":
        fdf = fdf.sort_values("Likes", ascending=False)
    else:
        fdf = fdf.sort_values("No")

    PAGE_SIZE  = 25
    total_pages = max(1, (len(fdf) - 1) // PAGE_SIZE + 1)
    pg = max(1, min(int(st.session_state["explorer_page"]), total_pages))
    st.session_state["explorer_page"] = pg

    st.caption(f"Showing **{len(fdf)}** of **{n_total}** comments · Page {pg}/{total_pages}")

    if fdf.empty:
        st.info("No comments match your criteria.")
    else:
        page_df = fdf.iloc[(pg-1)*PAGE_SIZE : pg*PAGE_SIZE]
        disp = page_df[["No", "Comment", "Prediction", "Confidence_pct", "Likes", "Author"]].copy()
        disp.columns = ["No", "Comment", "Prediction", "Confidence", "Likes", "Author"]

        def _hl(row):
            bg = "#f0fdf4" if row["Prediction"] == "Relevant" else "#fff1f2"
            return [f"background-color:{bg}"] * len(row)

        st.dataframe(
            disp.style.apply(_hl, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "No":         st.column_config.NumberColumn(width="small"),
                "Comment":    st.column_config.TextColumn(width="large"),
                "Prediction": st.column_config.TextColumn(width="medium"),
                "Confidence": st.column_config.TextColumn(width="small"),
                "Likes":      st.column_config.NumberColumn(width="small"),
                "Author":     st.column_config.TextColumn(width="medium"),
            },
        )

        pc1, pc2, pc3 = st.columns([1, 2, 1])
        with pc1:
            if st.button("⬅ Prev", disabled=pg <= 1):
                st.session_state["explorer_page"] = pg - 1
                st.rerun()
        with pc2:
            st.markdown(
                f"<p style='text-align:center;color:#606060;font-size:.83rem;padding-top:5px;'>"
                f"Page {pg} of {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with pc3:
            if st.button("Next ➡", disabled=pg >= total_pages):
                st.session_state["explorer_page"] = pg + 1
                st.rerun()

    st.divider()

    # ── 8. Moderation Report ──────────────────────────────────────────────
    st.markdown('<div class="cg-section">📋 Moderation Report</div>', unsafe_allow_html=True)

    mod_req = "Yes — manual moderation recommended." if pct_irr >= 0.10 else "No — discussion looks healthy."

    report_txt = f"""  Video Health Score       : {health['score']}/100  ({health['tier']})
  ─────────────────────────────────────────────
  Total comments analyzed  : {n_total}
  Relevant comments        : {n_relevant}
  Irrelevant comments      : {n_irrelevant}

  Relevant Rate            : {pct_rel*100:.1f}%
  Irrelevant Rate          : {pct_irr*100:.1f}%
  Avg Confidence Score     : {avg_conf*100:.1f}%
  Low Confidence Flags     : {low_conf}
  ─────────────────────────────────────────────
  Recommendation:
  {mod['recommendation']}

  Moderation Required      : {mod_req}"""

    st.markdown(
        f'<div class="cg-report" style="background:{health["bg"]};border:1.5px solid {health["color"]}35;color:{health["color"]};">'
        f'<pre style="margin:0;white-space:pre-wrap;font-family:inherit;font-size:inherit;">{report_txt}</pre>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 9. Export ─────────────────────────────────────────────────────────
    st.markdown('<div class="cg-section">⬇️ Export</div>', unsafe_allow_html=True)

    full_exp     = df[["No","Comment","Author","Prediction","Confidence_pct","Likes","Published"]].rename(columns={"Confidence_pct":"Confidence"})
    flagged_exp  = full_exp[full_exp["Prediction"] == "Irrelevant"]

    e1, e2, e3 = st.columns(3)

    with e1:
        buf = io.StringIO()
        full_exp.to_csv(buf, index=False)
        st.download_button(
            "📄 All Comments (CSV)", buf.getvalue(),
            file_name="prediction_results.csv", mime="text/csv",
            use_container_width=True,
        )

    with e2:
        buf2 = io.StringIO()
        flagged_exp.to_csv(buf2, index=False)
        st.download_button(
            "🚩 Flagged Comments (CSV)", buf2.getvalue(),
            file_name="flagged_comments.csv", mime="text/csv",
            use_container_width=True,
        )

    with e3:
        rbuf = io.StringIO()
        rbuf.write("COMMENTGUARD — MODERATION REPORT\n")
        rbuf.write("=" * 42 + "\n\n")
        rbuf.write(f"Video   : {meta['title']}\n")
        rbuf.write(f"Channel : {meta['channel_name']}\n\n")
        rbuf.write(report_txt.strip())
        st.download_button(
            "📊 Analysis Report (TXT)", rbuf.getvalue(),
            file_name="moderation_report.txt", mime="text/plain",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

elif not st.session_state["analysis_done"]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#909090;">
        <div style="font-size:3rem;margin-bottom:.8rem;">🛡️</div>
        <h3 style="color:#0F0F0F;font-weight:700;">Ready to Analyze</h3>
        <p style="max-width:460px;margin:.5rem auto 0;font-size:.9rem;">
            Paste a YouTube video URL and click <strong style="color:#FF0000;">▶ Analyze</strong>
            to get a full comment relevance breakdown, health score, and moderation report.
        </p>
        <div style="margin-top:1.3rem;display:flex;justify-content:center;gap:1.6rem;font-size:.82rem;flex-wrap:wrap;">
            <span>🩺 Health score</span>
            <span>🚩 Flagged comments</span>
            <span>📊 Analytics</span>
            <span>📋 Moderation report</span>
            <span>⬇️ Export CSV</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
