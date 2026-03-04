"""
================================================================
Author  : @Ranjeeta
Module  : src/profiling/class_profiler.py
Purpose : Create class-level emotional profiles from merged data
          (Extracted & modularised from main_updated.ipynb Cell 9)
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Any


def create_class_profile(class_id: str, class_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a comprehensive emotional profile for a single class.

    Parameters
    ----------
    class_id   : Unique class identifier
    class_data : Subset of the merged dataframe for this class only

    Returns
    -------
    dict with schema:
        class_id, class_name, total_sessions, total_students,
        emotions, wellness_indicators, trend_analysis, last_session
    """
    class_name = class_data["Class Name"].iloc[0] if "Class Name" in class_data.columns else f"Class_{class_id[:8]}"
    total_sessions = class_data["Session ID"].nunique()
    total_students = class_data["Student ID"].nunique()

    # ── Emotion distribution ──────────────────────────────────────────────
    emotion_counter = Counter(class_data["Emotion"].dropna())
    dominant_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "UNKNOWN"

    avg_intensity = float(class_data["Emotion Intensity Percentage"].mean()) if "Emotion Intensity Percentage" in class_data.columns else 0.5

    recent_emotions = (
        class_data.sort_values("Created At", ascending=False)["Emotion"]
        .dropna()
        .head(10)
        .tolist()
    ) if "Created At" in class_data.columns else []

    # ── Wellness indicators ───────────────────────────────────────────────
    avg_tiredness = float(class_data["Tiredness"].mean()) if "Tiredness" in class_data.columns else 3.0
    chat_requests = int(class_data["Chat Requested"].sum()) if "Chat Requested" in class_data.columns else 0
    absence_rate  = float(1 - class_data["Student ID"].nunique() / max(total_sessions, 1))

    # ── Trend analysis (recent vs earlier intensity) ──────────────────────
    trend_info = _compute_trend(class_data)

    last_session = str(class_data["Created At"].max()) if "Created At" in class_data.columns else "N/A"

    return {
        "class_id":   class_id,
        "class_name": class_name,
        "total_sessions": total_sessions,
        "total_students": total_students,
        "emotions": {
            "dominant_emotion":    dominant_emotion,
            "avg_intensity":       round(avg_intensity, 3),
            "emotion_distribution": dict(emotion_counter),
            "recent_emotions":     recent_emotions,
        },
        "wellness_indicators": {
            "avg_tiredness":  round(avg_tiredness, 3),
            "chat_requests":  chat_requests,
            "absence_rate":   round(absence_rate, 3),
        },
        "trend_analysis": trend_info,
        "last_session": last_session,
    }


def _compute_trend(class_data: pd.DataFrame) -> Dict[str, Any]:
    """Compute recent vs earlier avg intensity to determine trend direction."""
    if "Created At" not in class_data.columns or "Emotion Intensity Percentage" not in class_data.columns:
        return {"trend": "insufficient_data", "recent_avg_intensity": None, "earlier_avg_intensity": None, "volatility": None}

    sorted_data = class_data.sort_values("Created At")
    mid = len(sorted_data) // 2
    earlier = sorted_data.iloc[:mid]["Emotion Intensity Percentage"].mean()
    recent  = sorted_data.iloc[mid:]["Emotion Intensity Percentage"].mean()
    volatility = float(class_data["Emotion Intensity Percentage"].std())

    if pd.isna(recent) or pd.isna(earlier):
        trend = "insufficient_data"
    elif recent > earlier + 0.05:
        trend = "improving"
    elif recent < earlier - 0.05:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "trend":                trend,
        "recent_avg_intensity": round(float(recent), 3) if not pd.isna(recent) else None,
        "earlier_avg_intensity": round(float(earlier), 3) if not pd.isna(earlier) else None,
        "volatility":           round(volatility, 3) if not pd.isna(volatility) else None,
    }


def build_all_class_profiles(
    merged_df: pd.DataFrame,
    top_n: int = 100,
) -> Dict[str, Dict]:
    """
    Build emotional profiles for the top N most recent classes.

    Parameters
    ----------
    merged_df : Full merged relationship dataframe
    top_n     : How many classes to profile (default 100)

    Returns
    -------
    dict of { class_id: profile_dict }
    """
    print(f"[ClassProfiler] Selecting top {top_n} most recent classes ...")

    if "Created At" not in merged_df.columns or "Class ID" not in merged_df.columns:
        raise ValueError("merged_df must have 'Created At' and 'Class ID' columns.")

    recent_date = merged_df["Created At"].max()
    class_last_seen = merged_df.groupby("Class ID")["Created At"].max().reset_index()
    class_last_seen.columns = ["Class ID", "last_seen"]
    class_last_seen = class_last_seen.sort_values("last_seen", ascending=False).head(top_n)

    selected_class_ids = set(class_last_seen["Class ID"].tolist())
    filtered_df = merged_df[merged_df["Class ID"].isin(selected_class_ids)]

    profiles = {}
    groups = list(filtered_df.groupby("Class ID"))
    print(f"[ClassProfiler] Profiling {len(groups)} classes ...")

    for class_id, class_data in groups:
        profiles[class_id] = create_class_profile(class_id, class_data)

    print(f"[ClassProfiler] Done. {len(profiles)} class profiles created.")
    return profiles
