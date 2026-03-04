"""
================================================================
Author  : @Ranjeeta
Module  : src/profiling/student_profiler.py
Purpose : Create per-student emotional profiles over time
          (Extracted & modularised from recomandation.ipynb)
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Any, List


NEGATIVE_EMOTIONS = {"SAD", "ANXIOUS", "ANGRY", "SCARED"}


def create_student_profile(student_id: str, student_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a comprehensive emotional profile for a single student.

    Parameters
    ----------
    student_id   : Unique student identifier
    student_data : Subset of merged dataframe for this student only

    Returns
    -------
    dict with schema:
        student_id, total_checkins, emotions, wellness_indicators,
        trend_analysis, recent_emotions
    """
    student_data = student_data.sort_values("Created At") if "Created At" in student_data.columns else student_data

    # ── Emotion summary ───────────────────────────────────────────────────
    emotion_counts = Counter(student_data["Emotion"].dropna())
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "UNKNOWN"
    avg_intensity = float(student_data["Emotion Intensity Percentage"].mean()) if "Emotion Intensity Percentage" in student_data.columns else 0.5

    recent_emotions: List[str] = (
        student_data["Emotion"].dropna().tail(10).tolist()
    )

    negative_count = sum(1 for e in student_data["Emotion"].dropna() if e in NEGATIVE_EMOTIONS)
    negative_ratio  = negative_count / max(len(student_data), 1)

    # ── Wellness ──────────────────────────────────────────────────────────
    avg_tiredness = float(student_data["Tiredness"].mean()) if "Tiredness" in student_data.columns else 3.0
    chat_requests = int(student_data["Chat Requested"].sum()) if "Chat Requested" in student_data.columns else 0

    # ── Trend ─────────────────────────────────────────────────────────────
    trend = _analyze_trend(student_data)

    return {
        "student_id":   student_id,
        "total_checkins": len(student_data),
        "emotions": {
            "dominant_emotion":  dominant_emotion,
            "avg_intensity":     round(avg_intensity, 3),
            "emotion_counts":    dict(emotion_counts),
            "negative_ratio":    round(negative_ratio, 3),
            "recent_emotions":   recent_emotions,
        },
        "wellness_indicators": {
            "avg_tiredness": round(avg_tiredness, 3),
            "chat_requests": chat_requests,
        },
        "trend_analysis": trend,
    }


def _analyze_trend(student_data: pd.DataFrame) -> Dict[str, Any]:
    """Compute emotional intensity trend direction for one student."""
    if "Emotion Intensity Percentage" not in student_data.columns or len(student_data) < 4:
        return {"trend": "insufficient_data", "slope": None}

    intensities = student_data["Emotion Intensity Percentage"].dropna().tolist()
    if len(intensities) < 4:
        return {"trend": "insufficient_data", "slope": None}

    x = np.arange(len(intensities))
    slope = float(np.polyfit(x, intensities, 1)[0])

    if slope > 0.005:
        trend = "worsening"   # intensity of whatever dominant emotion is increasing
    elif slope < -0.005:
        trend = "improving"
    else:
        trend = "stable"

    return {"trend": trend, "slope": round(slope, 5)}


def build_all_student_profiles(merged_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run profiling on every student in the merged dataset.

    Returns
    -------
    dict of { student_id: profile_dict }
    """
    print("[StudentProfiler] Building per-student emotional profiles ...")
    unique_students = merged_df["Student ID"].dropna().unique()
    print(f"[StudentProfiler] {len(unique_students):,} unique students found.")

    profiles = {}
    for student_id in unique_students:
        student_data = merged_df[merged_df["Student ID"] == student_id]
        profiles[str(student_id)] = create_student_profile(str(student_id), student_data)

    print(f"[StudentProfiler] Done. {len(profiles):,} student profiles created.")
    return profiles
