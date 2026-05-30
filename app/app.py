"""
================================================================
Author  : @quantum hustel
Module  : app/app.py
Purpose : Streamlit teacher dashboard — Switch4Schools
          6 views: Alert Inbox, Class Overview, At-Risk Students,
          Student Deep-Dive, Risk Predictions, Cluster View
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Switch4Schools — Wellbeing Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
}
.stApp { background-color: #F4F6FA; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1F36 0%, #2D3561 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #E8ECF8 !important; font-family: 'Poppins', sans-serif !important; }
[data-testid="stSidebar"] .stRadio label { font-size:0.88rem; font-weight:500; padding:6px 0; letter-spacing:0.3px; }

h1 { font-weight:700; color:#1A1F36; letter-spacing:-0.5px; }
h2 { font-weight:600; color:#1A1F36; font-size:1.3rem; }
h3 { font-weight:600; color:#2D3561; font-size:1.05rem; }

[data-testid="metric-container"] {
    background:white; border-radius:14px; padding:18px 20px;
    box-shadow:0 2px 12px rgba(0,0,0,0.07); border:1px solid #EEF0F6;
}
[data-testid="metric-container"] label { font-size:0.78rem !important; font-weight:600 !important; text-transform:uppercase; letter-spacing:0.8px; color:#7B8AB8 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:2.1rem !important; font-weight:700 !important; color:#1A1F36 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:0.75rem !important; font-weight:500 !important; }

[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,0.06); }

.card { background:white; border-radius:14px; padding:24px 28px; box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:20px; border:1px solid #EEF0F6; }
.badge-high   { background:#FDECEA; color:#C0392B; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#FEF9E7; color:#D4AC0D; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#EAFAF1; color:#1E8449; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }

hr { border:none; border-top:1.5px solid #EEF0F6; margin:18px 0; }
.js-plotly-plot { border-radius:12px; }
.stWarning, .stInfo { border-radius:10px; font-size:0.88rem; }

.section-header {
    background:linear-gradient(90deg, #2D3561, #4A5899);
    color:white; padding:10px 20px; border-radius:10px;
    font-weight:600; font-size:0.95rem; margin-bottom:18px; letter-spacing:0.3px;
}
.alert-card {
    background:#FFF5F5; border-left:4px solid #E74C3C;
    border-radius:8px; padding:12px 16px; margin-bottom:10px;
}
.alert-card-medium {
    background:#FFFBF0; border-left:4px solid #F39C12;
    border-radius:8px; padding:12px 16px; margin-bottom:10px;
}
.flag-chip {
    display:inline-block; background:#FDECEA; color:#C0392B;
    border-radius:12px; padding:2px 10px; font-size:0.75rem;
    font-weight:600; margin-right:4px; margin-bottom:4px;
}
.flag-chip-orange {
    display:inline-block; background:#FEF9E7; color:#D4AC0D;
    border-radius:12px; padding:2px 10px; font-size:0.75rem;
    font-weight:600; margin-right:4px; margin-bottom:4px;
}
.sidebar-brand { font-size:1.15rem; font-weight:700; color:white; letter-spacing:0.5px; margin-bottom:4px; }
.sidebar-sub   { font-size:0.72rem; color:#9EA8CC; letter-spacing:0.5px; margin-bottom:24px; }
</style>
""", unsafe_allow_html=True)

# ── File paths ────────────────────────────────────────────────────────────────
OUTPUTS_DIR       = Path(__file__).resolve().parent.parent / "outputs"
PROFILES_PATH     = OUTPUTS_DIR / "class_emotional_profiles.json"
RECS_PATH         = OUTPUTS_DIR / "class_recommendations.json"
PREDICTIONS_PATH  = OUTPUTS_DIR / "risk_predictions.json"
STU_PROFILES_PATH  = OUTPUTS_DIR / "student_profiles.json"
CLUSTERS_PATH      = OUTPUTS_DIR / "student_clusters.json"
CLASS_MAP_PATH     = OUTPUTS_DIR / "class_student_map.json"

PRIORITY_COLORS = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
EMOTION_COLORS  = {
    "HAPPY":"#F4D03F","EXCITED":"#E67E22","CALM":"#3498DB",
    "SAD":"#8E44AD","ANXIOUS":"#E74C3C","ANGRY":"#C0392B",
    "SCARED":"#922B21","BORED":"#95A5A6","CONFUSED":"#1ABC9C","UNKNOWN":"#BDC3C7",
}
CHART_THEME = dict(font_family="Poppins", plot_bgcolor="white", paper_bgcolor="white", font_color="#1A1F36")


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_profiles():
    if not PROFILES_PATH.exists(): return {}
    with open(PROFILES_PATH) as f: data = json.load(f)
    return data.get("class_profiles", {})

@st.cache_data
def load_recommendations():
    if not RECS_PATH.exists(): return {}
    with open(RECS_PATH) as f: data = json.load(f)
    return data.get("class_recommendations", {})

@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists(): return []
    with open(PREDICTIONS_PATH) as f: data = json.load(f)
    return data.get("predictions", [])

@st.cache_data
def load_student_profiles():
    if not STU_PROFILES_PATH.exists(): return {}
    with open(STU_PROFILES_PATH) as f: data = json.load(f)
    return data.get("student_profiles", {})

@st.cache_data
def load_clusters():
    if not CLUSTERS_PATH.exists(): return {}
    with open(CLUSTERS_PATH) as f: return json.load(f)

@st.cache_data
def load_class_map():
    if not CLASS_MAP_PATH.exists(): return {}
    with open(CLASS_MAP_PATH) as f: data = json.load(f)
    return data.get("class_map", {})


# ── Load all data ─────────────────────────────────────────────────────────────
profiles     = load_profiles()
recs         = load_recommendations()
preds        = load_predictions()
stu_profiles = load_student_profiles()
clusters     = load_clusters()
class_map    = load_class_map()

high_count   = sum(1 for r in recs.values() if r.get("priority_level") == "HIGH")
medium_count = sum(1 for r in recs.values() if r.get("priority_level") == "MEDIUM")
low_count    = sum(1 for r in recs.values() if r.get("priority_level") == "LOW")

# Build merged predictions lookup (student_id → pred dict)
pred_lookup = {p["student_id"]: p for p in preds}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Switch4Schools</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">WELLBEING INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Global class selector ──────────────────────────────────
    # Teacher picks their class first; every student-level view
    # (Alert Inbox, At-Risk, Deep-Dive) filters to that class only.
    # Initialise class-change flag before the if/else so it's always defined
    _class_just_changed = False

    if class_map:
        class_options = {"All Classes": None}
        for cid, cdata in sorted(class_map.items(), key=lambda x: x[1]["class_name"]):
            class_options[cdata["class_name"]] = cid
        selected_class_name = st.selectbox("📚 Select Class", list(class_options.keys()))
        selected_class_id   = class_options[selected_class_name]

        # Detect class change → show loading spinner in sidebar
        if "prev_class" not in st.session_state:
            st.session_state.prev_class = selected_class_name
        _class_just_changed = st.session_state.prev_class != selected_class_name
        if _class_just_changed:
            st.session_state.prev_class = selected_class_name
            with st.spinner(f"Loading {selected_class_name}..."):
                time.sleep(0.45)   # brief pause so the spinner is visible

        # Set of student IDs in selected class (None = no filter)
        active_sids = (
            set(class_map[selected_class_id]["student_ids"])
            if selected_class_id else None
        )
    else:
        selected_class_name = "All Classes"
        selected_class_id   = None
        active_sids = None
        st.info("Run pipeline to enable class filtering.")

    st.markdown("---")

    view_mode = st.radio(
        "Navigation",
        ["🚨 Alert Inbox", "📊 Class Overview", "⚠️ At-Risk Students",
         "🔍 Student Deep-Dive", "📈 Risk Predictions", "🔵 Cluster View"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Apply class filter to derive scoped datasets used across all views
    scoped_preds        = [p for p in preds if active_sids is None or p["student_id"] in active_sids]
    scoped_stu_profiles = {sid: sp for sid, sp in stu_profiles.items() if active_sids is None or sid in active_sids}

    st.markdown("**Quick Stats**")
    if selected_class_name != "All Classes":
        st.markdown(f"Class: **{selected_class_name}**")
    st.markdown(f"Classes monitored: **{len(profiles)}**")
    st.markdown(f"Students in view: **{len(scoped_preds):,}**")
    escalating = sum(1 for p in scoped_preds if p.get("trajectory") == "escalating")
    st.markdown(f"Escalating now: **{escalating:,}**")
    if scoped_stu_profiles:
        compound_flags = sum(
            1 for sp in scoped_stu_profiles.values()
            if sp.get("dysregulation_flags", {}).get("compound_flag_rate", 0) >= 0.05
        )
        st.markdown(f"Multi-flag alerts: **{compound_flags:,}**")

    st.markdown("---")
    st.markdown('<span style="font-size:0.72rem; color:#6B7A9F;">IFN735 Team 29 · Quantum Hustle</span>', unsafe_allow_html=True)


# Toast notification when class selection changes
if _class_just_changed and selected_class_name != "All Classes":
    st.toast(f"📚 Switched to **{selected_class_name}**", icon="✅")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## Wellbeing Intelligence Dashboard")
scope_label = f" — {selected_class_name}" if selected_class_name != "All Classes" else ""
st.markdown(f'<p style="color:#7B8AB8; font-size:0.88rem; margin-top:-12px; margin-bottom:20px;">Real-time student emotional wellbeing monitoring{scope_label}</p>', unsafe_allow_html=True)

# ── KPI row (scoped to selected class) ────────────────────────────────────────
scoped_high   = sum(1 for p in scoped_preds if p.get("risk_level") == "HIGH")
scoped_medium = sum(1 for p in scoped_preds if p.get("risk_level") == "MEDIUM")
scoped_low    = sum(1 for p in scoped_preds if p.get("risk_level") == "LOW")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Classes Monitored",  len(profiles))
k2.metric("Students in View",   f"{len(scoped_preds):,}")
k3.metric("High Risk",          scoped_high,   delta="Needs attention", delta_color="inverse")
k4.metric("Medium Risk",        scoped_medium, delta_color="off")
k5.metric("Low Risk",           scoped_low,    delta_color="off")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — ALERT INBOX
# ═══════════════════════════════════════════════════════════════════════════════
if view_mode == "🚨 Alert Inbox":
    st.markdown(f'<div class="section-header">🚨 Alert Inbox — {selected_class_name}</div>', unsafe_allow_html=True)

    if not scoped_stu_profiles:
        st.warning("No student profiles found. Run `python run_pipeline.py` first.")
    else:
        # Build alert list from new metrics
        alert_rows = []
        for sid, sp in scoped_stu_profiles.items():
            em   = sp.get("emotions", {})
            wl   = sp.get("wellness_indicators", {})
            tr   = sp.get("trend_analysis", {})
            res  = sp.get("resilience", {})
            flags = sp.get("dysregulation_flags", {})
            pred  = pred_lookup.get(sid, {})

            compound_rate  = flags.get("compound_flag_rate", 0)
            bounce_back    = res.get("bounce_back_avg_checkins")
            longitudinal   = tr.get("longitudinal_direction", "insufficient_data")
            risk_level     = pred.get("risk_level", "LOW")
            trajectory     = pred.get("trajectory", "stable")
            risk_score     = pred.get("risk_score", 0)
            priority_score = pred.get("priority_score", 0)

            # Determine alert flags
            alert_flags = []
            if risk_level == "HIGH":
                alert_flags.append("HIGH RISK")
            if compound_rate >= 0.10:
                alert_flags.append(f"Compound Flags {compound_rate:.0%}")
            elif compound_rate >= 0.05:
                alert_flags.append(f"Multi-Flag {compound_rate:.0%}")
            if bounce_back is not None and bounce_back >= 5:
                alert_flags.append(f"Slow Recovery ({bounce_back:.1f} check-ins)")
            if longitudinal == "worsening":
                alert_flags.append("7d Trend ↑ Worsening")
            if trajectory == "escalating":
                alert_flags.append("Escalating")
            if wl.get("chat_requests", 0) > 0:
                alert_flags.append(f"Chat Request ×{wl['chat_requests']}")

            if alert_flags:
                alert_rows.append({
                    "student_id":        sid,
                    "risk_level":        risk_level,
                    "risk_score":        risk_score,
                    "priority_score":    priority_score,
                    "dominant_emotion":  em.get("dominant_emotion", "N/A"),
                    "avg_intensity":     em.get("avg_intensity", 0),
                    "compound_flag_rate": compound_rate,
                    "bounce_back":       bounce_back,
                    "longitudinal":      longitudinal,
                    "trajectory":        trajectory,
                    "alert_flags":       ", ".join(alert_flags),
                    "dysreg_events":     res.get("dysregulation_events", 0),
                    "suggested_action":  pred.get("suggested_action", "Review student profile"),
                })

        alert_rows.sort(key=lambda x: x["priority_score"], reverse=True)

        # Summary counts
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Alerts",      len(alert_rows))
        a2.metric("HIGH Risk",         sum(1 for r in alert_rows if r["risk_level"] == "HIGH"))
        a3.metric("Multi-Flag",        sum(1 for r in alert_rows if r["compound_flag_rate"] >= 0.05))
        a4.metric("7d Worsening",      sum(1 for r in alert_rows if r["longitudinal"] == "worsening"))

        st.markdown("---")

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            alert_risk_filter = st.multiselect("Filter Risk Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
        with col2:
            max_alerts = st.slider("Max students shown", 10, 200, 50)

        alert_df = pd.DataFrame([r for r in alert_rows if r["risk_level"] in alert_risk_filter]).head(max_alerts)

        if not alert_df.empty:
            display_cols = ["student_id", "risk_level", "risk_score", "priority_score",
                            "dominant_emotion", "avg_intensity", "compound_flag_rate",
                            "bounce_back", "longitudinal", "trajectory", "alert_flags", "suggested_action"]

            def colour_risk(val):
                return "background-color: " + {"HIGH":"#FDECEA","MEDIUM":"#FEF9E7","LOW":"#EAFAF1"}.get(val,"white")

            st.dataframe(
                alert_df[display_cols].style.applymap(colour_risk, subset=["risk_level"]),
                use_container_width=True,
                height=420,
            )

            st.markdown("---")

            # Alert breakdown charts
            c1, c2, c3 = st.columns(3)
            with c1:
                fig = px.pie(alert_df, names="risk_level", color="risk_level",
                             color_discrete_map=PRIORITY_COLORS, title="Alert Risk Levels", hole=0.45)
                fig.update_layout(**CHART_THEME, title_font_size=13, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(alert_df, x="compound_flag_rate", nbins=20,
                                   color_discrete_sequence=["#E74C3C"],
                                   title="Compound Flag Rate Distribution",
                                   labels={"compound_flag_rate": "Compound Flag Rate"})
                fig.update_layout(**CHART_THEME, title_font_size=13, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)
            with c3:
                bb_data = alert_df[alert_df["bounce_back"].notna()]
                if not bb_data.empty:
                    fig = px.histogram(bb_data, x="bounce_back", nbins=20,
                                       color_discrete_sequence=["#8E44AD"],
                                       title="Bounce-Back Rate Distribution",
                                       labels={"bounce_back": "Check-ins to Recover"})
                    fig.update_layout(**CHART_THEME, title_font_size=13, margin=dict(t=40,b=10,l=10,r=10))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No students match the selected alert filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — CLASS OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "📊 Class Overview":

    if not profiles:
        st.warning("No class profiles found. Run `python run_pipeline.py` first.")

    # ── SINGLE CLASS DETAILED VIEW ─────────────────────────────────────────
    elif selected_class_id and selected_class_id in profiles:
        p   = profiles[selected_class_id]
        rec = recs.get(selected_class_id, {})
        em  = p.get("emotions", {})
        wl  = p.get("wellness_indicators", {})
        tr  = p.get("trend_analysis", {})
        class_name = p.get("class_name", selected_class_id)

        st.markdown(f'<div class="section-header">📊 Class Overview — {class_name}</div>', unsafe_allow_html=True)

        # ── Key metrics row ────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Sessions",  p.get("total_sessions", 0))
        m2.metric("Total Students",  p.get("total_students", 0))
        m3.metric("Dominant Emotion", em.get("dominant_emotion", "N/A"))
        m4.metric("Avg Intensity",   f"{em.get('avg_intensity', 0):.3f}")
        m5.metric("Avg Tiredness",   f"{wl.get('avg_tiredness', 0):.2f} / 5")

        st.markdown("---")

        # ── Two-column detail cards ────────────────────────────────
        col_info, col_act = st.columns(2)

        with col_info:
            priority    = rec.get("priority_level", "N/A")
            pcolor      = {"HIGH":"#E74C3C","MEDIUM":"#F39C12","LOW":"#27AE60"}.get(priority,"#7B8AB8")
            trend       = tr.get("trend", "N/A")
            slope       = tr.get("slope")
            absence     = wl.get("absence_rate", 0)
            trend_color = {"worsening":"#E74C3C","improving":"#27AE60","stable":"#3498DB"}.get(trend,"#95A5A6")
            slope_str   = f" &nbsp;<span style='font-size:0.8rem;color:#9EA8CC;'>(slope: {slope})</span>" if slope is not None else ""
            st.markdown(f"""
            <div class="card">
              <p style="margin:0 0 12px 0;font-size:1.05rem;font-weight:600;">Class Health Summary</p>
              <p style="margin:0 0 8px 0;"><b>Priority Level:</b>
                <span style="color:{pcolor};font-weight:700;font-size:1.05rem;">&nbsp;{priority}</span></p>
              <p style="margin:0 0 8px 0;"><b>Emotional Trend:</b>
                <span style="color:{trend_color};font-weight:600;">&nbsp;{trend.upper()}</span>{slope_str}</p>
              <p style="margin:0 0 8px 0;"><b>Absence Rate:</b>&nbsp; {absence*100:.1f}%</p>
              <p style="margin:0 0 8px 0;"><b>Avg Intensity:</b>&nbsp; {em.get('avg_intensity',0):.3f}
                &nbsp;<span style='font-size:0.8rem;color:#9EA8CC;'>(HIGH = dysregulation)</span></p>
              <p style="margin:0;"><b>Last Session:</b>&nbsp; {p.get('last_session','N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_act:
            activities = rec.get("recommended_activities", [])
            duration   = rec.get("recommended_duration", "N/A")
            rationale  = rec.get("rationale", "")
            act_html   = "".join(f'<p style="margin:3px 0;">&#8226; {a}</p>' for a in activities) \
                         if activities else "<p style='color:#7B8AB8;'>No specific activities recommended.</p>"
            rat_html   = f'<p style="margin:8px 0 0 0;color:#7B8AB8;font-size:0.82rem;">{rationale}</p>' if rationale else ""
            st.markdown(f"""
            <div class="card">
              <p style="margin:0 0 12px 0;font-size:1.05rem;font-weight:600;">Recommended Activities</p>
              {act_html}
              <p style="margin:10px 0 0 0;"><b>Suggested Duration:</b>&nbsp; {duration}</p>
              {rat_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Emotion distribution chart for this class ──────────────
        emo_dist = em.get("emotion_distribution", em.get("emotion_counts", {}))
        if emo_dist:
            emo_df = pd.DataFrame(list(emo_dist.items()), columns=["Emotion", "Count"])
            emo_df = emo_df.sort_values("Count", ascending=False)
            fig = px.bar(emo_df, x="Emotion", y="Count", color="Emotion",
                         color_discrete_map=EMOTION_COLORS,
                         title=f"Emotion Distribution — {class_name}")
            fig.update_layout(**CHART_THEME, title_font_size=14, showlegend=False,
                              height=300, margin=dict(t=40,b=10,l=10,r=10),
                              xaxis=dict(tickangle=-20))
            st.plotly_chart(fig, use_container_width=True)

        # ── Single-row summary table ───────────────────────────────
        st.markdown("#### Full Profile Summary")
        summary_row = pd.DataFrame([{
            "Class":          class_name,
            "Emotion":        em.get("dominant_emotion","N/A"),
            "Intensity":      round(em.get("avg_intensity",0),3),
            "Tiredness":      round(wl.get("avg_tiredness",0),2),
            "Absence %":      f"{wl.get('absence_rate',0)*100:.1f}%",
            "Trend":          tr.get("trend","N/A"),
            "Sessions":       p.get("total_sessions",0),
            "Students":       p.get("total_students",0),
            "Priority":       rec.get("priority_level","N/A"),
            "Activities":     "; ".join((rec.get("recommended_activities",[]))[:3]),
        }])
        def colour_priority_ov(val):
            return "background-color: " + {"HIGH":"#FDECEA","MEDIUM":"#FEF9E7","LOW":"#EAFAF1"}.get(val,"white")
        st.dataframe(summary_row.style.applymap(colour_priority_ov, subset=["Priority"]),
                     use_container_width=True)

    # ── ALL CLASSES VIEW ───────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-header">📊 All Classes — Emotional Profiles</div>', unsafe_allow_html=True)

        rows = []
        for cid, p in profiles.items():
            rec = recs.get(cid, {})
            rows.append({
                "Class":          p.get("class_name", cid),
                "Emotion":        p["emotions"].get("dominant_emotion", "N/A"),
                "Intensity":      round(p["emotions"].get("avg_intensity", 0), 3),
                "Tiredness":      round(p["wellness_indicators"].get("avg_tiredness", 0), 2),
                "Absence %":      f"{p['wellness_indicators'].get('absence_rate', 0)*100:.1f}%",
                "Trend":          p["trend_analysis"].get("trend", "N/A"),
                "Sessions":       p.get("total_sessions", 0),
                "Students":       p.get("total_students", 0),
                "Priority":       rec.get("priority_level", "N/A"),
                "Top Activities": "; ".join(rec.get("recommended_activities", [])[:2]),
            })
        df = pd.DataFrame(rows)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            priority_filter = st.multiselect("Priority", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
        with col_f2:
            trend_filter = st.multiselect("Trend", df["Trend"].unique().tolist(), default=df["Trend"].unique().tolist())
        with col_f3:
            emotion_filter = st.multiselect("Dominant Emotion", df["Emotion"].unique().tolist(), default=df["Emotion"].unique().tolist())

        df_f = df[
            df["Priority"].isin(priority_filter) &
            df["Trend"].isin(trend_filter) &
            df["Emotion"].isin(emotion_filter)
        ]

        def colour_priority(val):
            return "background-color: " + {"HIGH":"#FDECEA","MEDIUM":"#FEF9E7","LOW":"#EAFAF1"}.get(val,"white")

        st.dataframe(df_f.style.applymap(colour_priority, subset=["Priority"]), use_container_width=True, height=360)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            fig = px.pie(df_f, names="Priority", color="Priority", color_discrete_map=PRIORITY_COLORS,
                         title="Priority Distribution", hole=0.45)
            fig.update_layout(**CHART_THEME, title_font_size=14, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            emo_counts = df_f["Emotion"].value_counts().reset_index()
            emo_counts.columns = ["Emotion", "Count"]
            fig = px.bar(emo_counts, x="Emotion", y="Count", color="Emotion",
                         color_discrete_map=EMOTION_COLORS, title="Dominant Emotions")
            fig.update_layout(**CHART_THEME, title_font_size=14, showlegend=False,
                              margin=dict(t=40,b=10,l=10,r=10), xaxis=dict(tickangle=-30))
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            trend_counts = df_f["Trend"].value_counts().reset_index()
            trend_counts.columns = ["Trend", "Count"]
            fig = px.bar(trend_counts, x="Trend", y="Count", color="Trend",
                         color_discrete_map={"improving":"#27AE60","stable":"#3498DB","declining":"#E74C3C","insufficient_data":"#95A5A6"},
                         title="Emotional Trends")
            fig.update_layout(**CHART_THEME, title_font_size=14, showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Intensity vs Tiredness by Class")
        df_scatter = pd.DataFrame([{
            "Class":     p.get("class_name", cid),
            "Intensity": p["emotions"].get("avg_intensity", 0),
            "Tiredness": p["wellness_indicators"].get("avg_tiredness", 0),
            "Priority":  recs.get(cid, {}).get("priority_level", "N/A"),
            "Trend":     p["trend_analysis"].get("trend", "N/A"),
        } for cid, p in profiles.items()])
        fig = px.scatter(df_scatter, x="Intensity", y="Tiredness", color="Priority", symbol="Trend",
                         color_discrete_map=PRIORITY_COLORS, hover_data=["Class"],
                         title="Class Wellbeing Landscape",
                         labels={"Intensity":"Avg Emotion Intensity","Tiredness":"Avg Tiredness (1–5)"})
        fig.update_layout(**CHART_THEME, title_font_size=14, height=380, margin=dict(t=40,b=20,l=20,r=20))
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3 — AT-RISK STUDENTS  (enhanced with new metrics)
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "⚠️ At-Risk Students":
    st.markdown(f'<div class="section-header">⚠️ At-Risk Students — {selected_class_name}</div>', unsafe_allow_html=True)

    if not scoped_preds:
        st.warning("No risk prediction data found. Run `python run_pipeline.py` first.")
    else:
        high_risk = [p for p in scoped_preds if p.get("risk_level") == "HIGH"]
        med_risk  = [p for p in scoped_preds if p.get("risk_level") == "MEDIUM"]

        s1, s2, s3 = st.columns(3)
        s1.metric("HIGH Risk Students",   len(high_risk))
        s2.metric("MEDIUM Risk Students", len(med_risk))
        s3.metric("LOW Risk Students",    len(scoped_preds) - len(high_risk) - len(med_risk))

        st.markdown("---")

        # Build enriched table including new profile metrics
        enriched = []
        for p in scoped_preds:
            sid = p["student_id"]
            sp  = scoped_stu_profiles.get(sid, {})
            res = sp.get("resilience", {})
            fl  = sp.get("dysregulation_flags", {})
            tr  = sp.get("trend_analysis", {})
            enriched.append({
                "student_id":         sid,
                "risk_level":         p.get("risk_level"),
                "risk_score":         p.get("risk_score"),
                "priority_score":     p.get("priority_score", 0),
                "trajectory":         p.get("trajectory"),
                "escalation_prob":    p.get("escalation_probability"),
                "compound_flag_%":    f"{fl.get('compound_flag_rate',0)*100:.1f}%",
                "bounce_back":        res.get("bounce_back_avg_checkins"),
                "dysreg_events":      res.get("dysregulation_events", 0),
                "longitudinal":       tr.get("longitudinal_direction","N/A"),
                "rolling_7d_avg":     tr.get("rolling_7d_avg"),
                "suggested_action":   p.get("suggested_action",""),
            })

        risk_filter = st.multiselect("Filter by Risk Level", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM"])
        filtered = [r for r in enriched if r.get("risk_level") in risk_filter]
        df_risk = pd.DataFrame(filtered)

        if not df_risk.empty:
            def colour_risk(val):
                return "background-color: " + {"HIGH":"#FDECEA","MEDIUM":"#FEF9E7","LOW":"#EAFAF1"}.get(val,"white")

            st.dataframe(
                df_risk.style.applymap(colour_risk, subset=["risk_level"]),
                use_container_width=True,
                height=420,
            )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            risk_counts = pd.DataFrame(scoped_preds)["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig = px.pie(risk_counts, names="Risk Level", values="Count", color="Risk Level",
                         color_discrete_map={"HIGH":"#E74C3C","MEDIUM":"#F39C12","LOW":"#27AE60"},
                         title=f"Risk Distribution — {selected_class_name}", hole=0.45)
            fig.update_layout(**CHART_THEME, title_font_size=14, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "trajectory" in pd.DataFrame(scoped_preds).columns:
                traj_counts = pd.DataFrame(scoped_preds)["trajectory"].value_counts().reset_index()
                traj_counts.columns = ["Trajectory", "Count"]
                fig = px.bar(traj_counts, x="Trajectory", y="Count", color="Trajectory",
                             color_discrete_map={"escalating":"#E74C3C","stable":"#3498DB","recovering":"#27AE60"},
                             title="Risk Trajectory Breakdown")
                fig.update_layout(**CHART_THEME, title_font_size=14, showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

        # Risk factors breakdown for top HIGH risk students
        if high_risk:
            st.markdown("---")
            st.markdown("#### Risk Factors — Top HIGH Risk Students")
            top_high = sorted(high_risk, key=lambda x: x.get("risk_score", 0), reverse=True)[:5]
            for p in top_high:
                with st.expander(f"Student {p['student_id'][:12]}…  |  Score: {p.get('risk_score')}  |  {p.get('trajectory','N/A').upper()}"):
                    factors = p.get("risk_factors", [])
                    if factors:
                        for f in factors:
                            st.markdown(f"- {f}")
                    else:
                        st.markdown("_No risk factors recorded._")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4 — STUDENT DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "🔍 Student Deep-Dive":
    st.markdown(f'<div class="section-header">🔍 Student Deep-Dive — {selected_class_name}</div>', unsafe_allow_html=True)

    if not scoped_stu_profiles:
        st.warning("No student profiles found. Run `python run_pipeline.py` first.")
    else:
        # Sort students by priority_score — highest risk first in dropdown
        sorted_sids = sorted(
            scoped_stu_profiles.keys(),
            key=lambda s: pred_lookup.get(s, {}).get("priority_score", 0),
            reverse=True,
        )

        selected_sid = st.selectbox(
            "Select a student (sorted by priority — highest risk first)",
            sorted_sids,
            format_func=lambda s: f"{s[:16]}…  [{pred_lookup.get(s,{}).get('risk_level','?')} | Score {pred_lookup.get(s,{}).get('risk_score',0)}]"
        )

        sp   = scoped_stu_profiles[selected_sid]
        pred = pred_lookup.get(selected_sid, {})

        em    = sp.get("emotions", {})
        wl    = sp.get("wellness_indicators", {})
        tr    = sp.get("trend_analysis", {})
        res   = sp.get("resilience", {})
        flags = sp.get("dysregulation_flags", {})

        # ── Top KPI row ────────────────────────────────────────────────
        st.markdown("---")
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Risk Level",     pred.get("risk_level", "N/A"))
        d2.metric("Risk Score",     pred.get("risk_score", 0))
        d3.metric("Priority Score", pred.get("priority_score", 0))
        d4.metric("Trajectory",     pred.get("trajectory", "N/A"))
        d5.metric("Total Check-ins", sp.get("total_checkins", 0))
        d6.metric("Dysreg. Events",  res.get("dysregulation_events", 0))

        st.markdown("---")

        # ── Row 1: Emotions + Wellness ─────────────────────────────────
        col_em, col_wl = st.columns(2)

        with col_em:
            st.markdown("#### Emotion Profile")
            st.markdown(f"**Dominant emotion:** `{em.get('dominant_emotion','N/A')}`")
            st.markdown(f"**Avg intensity:** `{em.get('avg_intensity', 0):.3f}`  *(HIGH = dysregulation)*")
            st.markdown(f"**Unpleasant ratio:** `{em.get('unpleasant_ratio', 0):.1%}`")
            st.markdown(f"**Recent emotions:** `{' → '.join(em.get('recent_emotions', [])[-5:])}`")

            emo_counts = em.get("emotion_counts", {})
            if emo_counts:
                emo_df = pd.DataFrame(list(emo_counts.items()), columns=["Emotion","Count"])
                fig = px.bar(emo_df.sort_values("Count", ascending=False),
                             x="Emotion", y="Count", color="Emotion",
                             color_discrete_map=EMOTION_COLORS,
                             title="Emotion Frequency")
                fig.update_layout(**CHART_THEME, title_font_size=13, showlegend=False,
                                  height=260, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

        with col_wl:
            st.markdown("#### Wellness Indicators")
            avg_tired = wl.get("avg_tiredness", 0)
            tired_pct = (avg_tired / 5) * 100
            st.markdown(f"**Avg tiredness:** `{avg_tired:.2f} / 5`")
            st.progress(int(tired_pct), text=f"Tiredness {tired_pct:.0f}%")

            absence = wl.get("absence_rate", 0)
            st.markdown(f"**Absence rate:** `{absence:.1%}`")
            st.progress(int(absence * 100), text=f"Absence {absence:.1%}")

            chat = wl.get("chat_requests", 0)
            st.markdown(f"**Chat support requests:** `{chat}`")
            if chat > 0:
                st.warning(f"This student has requested wellbeing chat {chat} time(s).")

        st.markdown("---")

        # ── Row 2: Trend + Resilience ──────────────────────────────────
        col_tr, col_res = st.columns(2)

        with col_tr:
            st.markdown("#### Trend Analysis")
            trend_dir  = tr.get("trend", "N/A")
            slope      = tr.get("slope")
            rolling7d  = tr.get("rolling_7d_avg")
            longit_dir = tr.get("longitudinal_direction", "N/A")

            trend_color = {"worsening":"#E74C3C","improving":"#27AE60","stable":"#3498DB"}.get(trend_dir,"#95A5A6")
            st.markdown(f"**Linear trend:** <span style='color:{trend_color};font-weight:600'>{trend_dir.upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"**Slope:** `{slope}`  *(positive slope = intensity increasing = worse)*")
            st.markdown(f"**7-day rolling avg intensity:** `{rolling7d}`")

            long_color = {"worsening":"#E74C3C","improving":"#27AE60","stable":"#3498DB"}.get(longit_dir,"#95A5A6")
            st.markdown(f"**Longitudinal direction (14-day):** <span style='color:{long_color};font-weight:600'>{longit_dir.upper()}</span>", unsafe_allow_html=True)

        with col_res:
            st.markdown("#### Resilience — Bounce-Back Rate")
            bb = res.get("bounce_back_avg_checkins")
            dysreg = res.get("dysregulation_events", 0)

            st.markdown(f"**Dysregulation events recorded:** `{dysreg}`")
            if bb is None:
                st.info("No dysregulation events found — student has not shown high-intensity unpleasant emotions.")
            else:
                bb_color = "#E74C3C" if bb >= 5 else ("#F39C12" if bb >= 3 else "#27AE60")
                st.markdown(f"**Avg check-ins to recover:** <span style='color:{bb_color};font-weight:700;font-size:1.4rem'>{bb:.1f}</span>", unsafe_allow_html=True)
                resilience_label = "Low resilience — slow recovery" if bb >= 5 else ("Moderate resilience" if bb >= 3 else "Good resilience — fast recovery")
                st.markdown(f"*{resilience_label}*")

        st.markdown("---")

        # ── Row 3: Compound Flags ──────────────────────────────────────
        st.markdown("#### Compound Dysregulation Flags")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Dysreg. Sessions",  flags.get("flag_dysregulation_sessions", 0))
        f2.metric("High Tiredness",    flags.get("flag_high_tiredness_sessions", 0))
        f3.metric("Chat Sessions",     flags.get("flag_chat_sessions", 0))
        f4.metric("Compound Sessions", flags.get("compound_flag_sessions", 0))
        f5.metric("Compound Rate",     f"{flags.get('compound_flag_rate', 0):.1%}")

        compound_rate = flags.get("compound_flag_rate", 0)
        if compound_rate >= 0.10:
            st.error(f"HIGH compound dysregulation rate ({compound_rate:.1%}) — this student regularly experiences multiple stressors simultaneously.")
        elif compound_rate >= 0.05:
            st.warning(f"Moderate compound dysregulation rate ({compound_rate:.1%}) — monitor closely.")
        else:
            st.success(f"Low compound flag rate ({compound_rate:.1%}) — no major co-occurring stressors detected.")

        # ── Row 4: Risk factors from prediction ────────────────────────
        st.markdown("---")
        st.markdown("#### Risk Factors Identified by System")
        risk_factors = pred.get("risk_factors", [])
        if risk_factors:
            for rf in risk_factors:
                st.markdown(f"- {rf}")
        else:
            st.info("No specific risk factors flagged for this student.")

        # ── Suggested action ───────────────────────────────────────────
        action = pred.get("suggested_action", "")
        if action:
            st.markdown("---")
            st.markdown(f"**Suggested Teacher Action:** `{action}`")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 5 — RISK PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "📈 Risk Predictions":
    st.markdown(f'<div class="section-header">📈 Risk Predictions — {selected_class_name}</div>', unsafe_allow_html=True)

    if not scoped_preds:
        st.warning("No prediction data found. Run `python run_pipeline.py` first.")
    else:
        df_pred = pd.DataFrame(scoped_preds)

        st.markdown("#### Top 10 Highest Priority Students")
        top10_cols = ["student_id","risk_level","risk_score","priority_score","escalation_probability","trajectory"]
        if "suggested_action" in df_pred.columns:
            top10_cols.append("suggested_action")
        top10 = df_pred.sort_values("priority_score", ascending=False).head(10)[top10_cols]

        def colour_risk_level(val):
            return "background-color: " + {"HIGH":"#FDECEA","MEDIUM":"#FEF9E7","LOW":"#EAFAF1"}.get(val,"white")

        st.dataframe(top10.style.applymap(colour_risk_level, subset=["risk_level"]),
                     use_container_width=True, height=280)

        st.markdown("---")

        st.markdown("#### Current Risk vs 14-Day Escalation Probability")
        fig = px.scatter(df_pred, x="risk_score", y="escalation_probability",
                         color="trajectory",
                         color_discrete_map={"escalating":"#E74C3C","stable":"#3498DB","recovering":"#27AE60"},
                         hover_data=["student_id","risk_level"],
                         labels={"risk_score":"Current Risk Score (0–100)","escalation_probability":"14-Day Escalation Probability"},
                         opacity=0.65)
        fig.update_layout(**CHART_THEME, height=460, margin=dict(t=20,b=30,l=30,r=20), legend_title_text="Trajectory")
        fig.update_traces(marker=dict(size=7))
        fig.add_hline(y=50, line_dash="dot", line_color="#BDC3C7", line_width=1)
        fig.add_vline(x=50, line_dash="dot", line_color="#BDC3C7", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(df_pred, x="risk_score", nbins=30,
                                    color_discrete_sequence=["#2D3561"],
                                    title="Risk Score Distribution",
                                    labels={"risk_score":"Risk Score"})
            fig_hist.update_layout(**CHART_THEME, title_font_size=14, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_esc = px.histogram(df_pred, x="escalation_probability", nbins=30,
                                   color_discrete_sequence=["#E74C3C"],
                                   title="Escalation Probability Distribution",
                                   labels={"escalation_probability":"Escalation Probability"})
            fig_esc.update_layout(**CHART_THEME, title_font_size=14, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_esc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 6 — CLUSTER VIEW  (with K explorer)
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "🔵 Cluster View":
    st.markdown('<div class="section-header">🔵 Student Behavioural Clusters — K Explorer</div>', unsafe_allow_html=True)

    if not clusters:
        st.warning("No cluster data found. Run `python run_pipeline.py` first.")
    else:
        summaries        = clusters.get("cluster_summaries", {})
        assignments      = clusters.get("student_assignments", {})
        metrics_cmp      = clusters.get("metrics_comparison", [])
        best_k           = clusters.get("best_k", 2)
        algorithm        = clusters.get("algorithm", "K-Means")
        all_k_results    = clusters.get("all_k_results", {})

        # ── Top KPI row (best-K results) ───────────────────────────────
        total_clustered = sum(s.get("n_students", 0) for s in summaries.values())
        distressed_n = sum(
            s.get("n_students", 0) for s in summaries.values()
            if "distressed" in s.get("archetype","").lower() or "emotionally" in s.get("archetype","").lower()
        )
        stable_n = total_clustered - distressed_n

        ck1, ck2, ck3, ck4 = st.columns(4)
        ck1.metric("Total Students Clustered",  f"{total_clustered:,}")
        ck2.metric("Stable",                    f"{stable_n:,}",      delta=f"{stable_n/max(total_clustered,1):.1%}")
        ck3.metric("Emotionally Distressed",    f"{distressed_n:,}",  delta=f"{distressed_n/max(total_clustered,1):.1%}", delta_color="inverse")
        ck4.metric("Optimal K (Best Silhouette)", best_k)

        st.markdown("---")

        # ── SECTION A: Best-K results (canonical) ─────────────────────
        st.markdown(f"### Best K = {best_k} Results  ({algorithm})")
        sum_rows = []
        for cid, cs in summaries.items():
            if cid == "-1": continue
            sum_rows.append({
                "Cluster":              cid,
                "Archetype":            cs.get("archetype","N/A"),
                "Students":             cs.get("n_students", 0),
                "Avg Unpleasant Ratio": f"{cs.get('mean_unpleasant_ratio',0):.1%}",
                "Avg Intensity":        f"{cs.get('mean_avg_intensity',0):.3f}",
                "Avg Tiredness":        f"{cs.get('mean_avg_tiredness',0):.2f}",
                "Avg Absence Rate":     f"{cs.get('mean_absence_rate',0):.1%}",
            })
        if sum_rows:
            st.dataframe(pd.DataFrame(sum_rows), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            pie_data = pd.DataFrame([
                {"Archetype": cs.get("archetype","N/A"), "Count": cs.get("n_students",0)}
                for cid, cs in summaries.items() if cid != "-1"
            ])
            if not pie_data.empty:
                fig = px.pie(pie_data, names="Archetype", values="Count",
                             color="Archetype",
                             color_discrete_map={"Stable":"#27AE60","Emotionally Distressed":"#E74C3C"},
                             title=f"Cluster Distribution — Best K={best_k}", hole=0.4)
                fig.update_layout(**CHART_THEME, title_font_size=13, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if metrics_cmp:
                st.markdown("#### 4-Algorithm Comparison (at K={})".format(best_k))
                mc_df = pd.DataFrame([{
                    "Algorithm":      m.get("algorithm","N/A"),
                    "K":              m.get("n_clusters","N/A"),
                    "Silhouette":     m.get("silhouette"),
                    "Calinski-H":     m.get("calinski_harabasz"),
                    "Davies-Bouldin": m.get("davies_bouldin"),
                } for m in metrics_cmp])
                best_idx = mc_df["Silhouette"].idxmax() if mc_df["Silhouette"].notna().any() else None
                def highlight_best(row):
                    return ["background-color:#EAFAF1"]*len(row) if row.name == best_idx else [""]*len(row)
                st.dataframe(mc_df.style.apply(highlight_best, axis=1), use_container_width=True)

        st.markdown("---")

        # ── SECTION B: K Explorer — explore any K from 2 to 9 ─────────
        st.markdown("### K Explorer — Compare Different Cluster Solutions")
        st.caption("Use the slider to explore how the data splits at different K values. "
                   "K=2 is the statistically optimal choice (highest silhouette). "
                   "Higher K reveals sub-groups within Stable students.")

        if all_k_results:
            k_options = sorted([int(k) for k in all_k_results.keys()])
            selected_k = st.select_slider(
                "Select K to explore",
                options=k_options,
                value=best_k,
                format_func=lambda k: f"K={k}{'  ← Best' if k == best_k else ''}",
            )
            k_data = all_k_results[str(selected_k)]
            k_sil  = k_data.get("silhouette", "N/A")
            k_sums = k_data.get("cluster_summaries", {})

            st.markdown(f"**K={selected_k}** — Silhouette score: `{k_sil}`")

            # PCA scatter image for this K
            img_path = OUTPUTS_DIR / k_data.get("pca_image", "")
            if img_path.exists():
                st.image(str(img_path), caption=f"K-Means PCA Projection — K={selected_k}", use_column_width=True)
            else:
                st.info("PCA image not found — re-run pipeline to generate all-K visualisations.")

            # Cluster composition bar chart for selected K
            bar_rows = [
                {"Cluster": f"C{cid} — {cs['archetype']}", "Students": cs["n_students"]}
                for cid, cs in k_sums.items()
            ]
            if bar_rows:
                bar_df = pd.DataFrame(bar_rows)
                arch_palette = {"Stable":"#27AE60","Emotionally Distressed":"#E74C3C"}
                bar_colors   = [arch_palette.get(cs["archetype"], "#3498DB") for cs in k_sums.values()]
                fig = px.bar(bar_df, x="Cluster", y="Students",
                             title=f"Cluster Sizes — K={selected_k}",
                             color="Cluster",
                             color_discrete_sequence=bar_colors)
                fig.update_layout(**CHART_THEME, title_font_size=13, showlegend=False,
                                  margin=dict(t=40,b=30,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)

            # Detailed cluster table for selected K
            k_sum_rows = []
            for cid, cs in k_sums.items():
                k_sum_rows.append({
                    "Cluster":              cid,
                    "Archetype":            cs.get("archetype","N/A"),
                    "Students":             cs.get("n_students",0),
                    "Avg Unpleasant Ratio": f"{cs.get('mean_unpleasant_ratio',0):.1%}",
                    "Avg Intensity":        f"{cs.get('mean_avg_intensity',0):.3f}",
                    "Avg Tiredness":        f"{cs.get('mean_avg_tiredness',0):.2f}",
                })
            if k_sum_rows:
                st.dataframe(pd.DataFrame(k_sum_rows), use_container_width=True)
        else:
            st.info("K Explorer data not available — re-run pipeline to generate multi-K results.")

        st.markdown("---")

        # ── SECTION C: Distressed students list ───────────────────────
        st.markdown("#### Emotionally Distressed Students (Best K-Means)")
        distressed_sids = [
            sid for sid, asgn in assignments.items()
            if "distressed" in asgn.get("archetype","").lower() or "emotionally" in asgn.get("archetype","").lower()
        ]
        # Apply class filter to distressed list
        if active_sids is not None:
            distressed_sids = [s for s in distressed_sids if s in active_sids]

        if distressed_sids:
            dist_rows = []
            for sid in distressed_sids[:200]:
                pred = pred_lookup.get(sid, {})
                sp   = scoped_stu_profiles.get(sid, stu_profiles.get(sid, {}))
                em   = sp.get("emotions", {})
                res  = sp.get("resilience", {})
                fl   = sp.get("dysregulation_flags", {})
                dist_rows.append({
                    "student_id":       sid,
                    "risk_level":       pred.get("risk_level","N/A"),
                    "risk_score":       pred.get("risk_score",0),
                    "dominant_emotion": em.get("dominant_emotion","N/A"),
                    "unpleasant_ratio": em.get("unpleasant_ratio",0),
                    "avg_intensity":    em.get("avg_intensity",0),
                    "bounce_back":      res.get("bounce_back_avg_checkins"),
                    "compound_flag_%":  f"{fl.get('compound_flag_rate',0)*100:.1f}%",
                })
            dist_df = pd.DataFrame(dist_rows).sort_values("risk_score", ascending=False)

            def colour_risk(val):
                return "background-color: " + {"HIGH":"#FDECEA","MEDIUM":"#FEF9E7","LOW":"#EAFAF1"}.get(val,"white")

            st.dataframe(dist_df.style.applymap(colour_risk, subset=["risk_level"]),
                         use_container_width=True, height=380)
            st.caption(f"Showing up to 200 of {len(distressed_sids)} distressed students in view.")
        else:
            st.success("No Emotionally Distressed students in the current class/filter selection.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#9EA8CC; font-size:0.78rem; letter-spacing:0.5px;">'
    'Switch4Schools Wellbeing Intelligence Platform &nbsp;|&nbsp; IFN735 Team 29 &nbsp;|&nbsp; Quantum Hustle'
    '</p>',
    unsafe_allow_html=True,
)
