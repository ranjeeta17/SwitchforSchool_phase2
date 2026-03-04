"""
================================================================
Author  : @Ranjeeta
Module  : app/app.py
Purpose : Streamlit web dashboard for teachers
          Interactive view of class profiles, risk tiers, 
          trend charts, and intervention recommendations
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29

Usage:
    streamlit run app/app.py
================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Switch4Schools — Wellbeing Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
PROFILES_PATH     = OUTPUTS_DIR / "class_emotional_profiles.json"
RECS_PATH         = OUTPUTS_DIR / "class_recommendations.json"
PREDICTIONS_PATH  = OUTPUTS_DIR / "risk_predictions.json"

PRIORITY_COLORS = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
EMOTION_COLORS  = {
    "HAPPY": "#F1C40F", "EXCITED": "#E67E22", "CALM": "#3498DB",
    "SAD": "#9B59B6",   "ANXIOUS": "#E74C3C", "ANGRY": "#C0392B",
    "SCARED": "#8E44AD", "BORED": "#95A5A6", "CONFUSED": "#1ABC9C",
}


# ── Data loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_profiles():
    if not PROFILES_PATH.exists():
        return {}
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    return data.get("class_profiles", {})


@st.cache_data
def load_recommendations():
    if not RECS_PATH.exists():
        return {}
    with open(RECS_PATH) as f:
        data = json.load(f)
    return data.get("class_recommendations", {})


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)
    return data.get("predictions", [])


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60?text=Switch4Schools", use_column_width=True)
st.sidebar.title("🎛️ Filters")
view_mode = st.sidebar.radio("View", ["📊 Class Overview", "🔴 At-Risk Students", "🔮 Risk Predictions"])

profiles = load_profiles()
recs     = load_recommendations()
preds    = load_predictions()

# ── Header ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
high_count   = sum(1 for r in recs.values() if r.get("priority_level") == "HIGH")
medium_count = sum(1 for r in recs.values() if r.get("priority_level") == "MEDIUM")
low_count    = sum(1 for r in recs.values() if r.get("priority_level") == "LOW")

col1.metric("🏫 Classes Monitored", len(profiles))
col2.metric("🔴 High Priority",  high_count,  delta=f"Needs attention")
col3.metric("🟡 Medium Priority", medium_count)
col4.metric("🟢 Low Priority",   low_count)

st.markdown("---")

# ── View: Class Overview ──────────────────────────────────────────────────
if view_mode == "📊 Class Overview":
    st.header("📊 Class Emotional Profiles")

    if not profiles:
        st.warning("No class profiles found. Run the analysis pipeline first.")
    else:
        # Flatten for dataframe
        rows = []
        for cid, p in profiles.items():
            rec = recs.get(cid, {})
            rows.append({
                "Class":          p.get("class_name", cid),
                "Emotion":        p["emotions"].get("dominant_emotion", "N/A"),
                "Avg Intensity":  p["emotions"].get("avg_intensity", 0),
                "Avg Tiredness":  p["wellness_indicators"].get("avg_tiredness", 0),
                "Absence Rate":   p["wellness_indicators"].get("absence_rate", 0),
                "Trend":          p["trend_analysis"].get("trend", "N/A"),
                "Priority":       rec.get("priority_level", "N/A"),
                "Activities":     "; ".join(rec.get("recommended_activities", [])[:2]),
            })

        df = pd.DataFrame(rows)

        # Priority filter
        priority_filter = st.sidebar.multiselect(
            "Priority Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
        )
        df_filtered = df[df["Priority"].isin(priority_filter)]

        # Colour-coded table
        def colour_priority(val):
            c = {"HIGH": "#ffcccc", "MEDIUM": "#fff3cc", "LOW": "#ccffcc"}.get(val, "white")
            return f"background-color: {c}"

        st.dataframe(
            df_filtered.style.applymap(colour_priority, subset=["Priority"]),
            use_container_width=True,
        )

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names="Priority", color="Priority",
                         color_discrete_map=PRIORITY_COLORS, title="Priority Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(df["Emotion"].value_counts().reset_index(),
                         x="Emotion", y="count",
                         color="Emotion", title="Dominant Emotions Across Classes")
            st.plotly_chart(fig, use_container_width=True)

# ── View: At-Risk Students ─────────────────────────────────────────────────
elif view_mode == "🔴 At-Risk Students":
    st.header("🔴 At-Risk Student Overview")
    if preds:
        high_risk = [p for p in preds if p.get("risk_level") == "HIGH"]
        st.info(f"**{len(high_risk)}** students classified as HIGH risk out of **{len(preds)}** total.")
        df_risk = pd.DataFrame(preds)[["student_id", "risk_level", "risk_score", "risk_factors"]]
        st.dataframe(df_risk, use_container_width=True)
    else:
        st.warning("No risk prediction data found. Run `src/risk/risk_classifier.py` first.")

# ── View: Risk Predictions ─────────────────────────────────────────────────
elif view_mode == "🔮 Risk Predictions":
    st.header("🔮 Phase 2 – Risk Trajectory Predictions")
    if preds:
        df_pred = pd.DataFrame(preds)
        fig = px.scatter(
            df_pred,
            x="risk_score",
            y="escalation_probability",
            color="trajectory",
            hover_data=["student_id"],
            title="Current Risk vs Escalation Probability",
            labels={"risk_score": "Current Risk Score", "escalation_probability": "Escalation Probability (14 days)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df_pred[["student_id", "risk_level", "risk_score", "escalation_probability", "trajectory", "suggested_action"]]
            if "suggested_action" in df_pred.columns
            else df_pred[["student_id", "risk_level", "risk_score", "escalation_probability", "trajectory"]],
            use_container_width=True,
        )
    else:
        st.warning("No prediction data found. Run the Phase 2 risk prediction pipeline.")

st.markdown("---")
st.caption("🧠 Switch4Schools Wellbeing Intelligence Platform | IFN735 Team 29 | Author: @Ranjeeta")
