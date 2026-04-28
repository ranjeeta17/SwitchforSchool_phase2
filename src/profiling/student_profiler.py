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
from typing import Dict, Any, List, Optional


UNPLEASANT_EMOTIONS = {"SAD", "ANXIOUS", "ANGRY", "SCARED"}


def create_student_profile(student_id: str, student_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a comprehensive emotional profile for a single student.

    Returns
    -------
    dict with schema:
        student_id, total_checkins, emotions, wellness_indicators,
        trend_analysis, resilience, dysregulation_flags
    """
    student_data = student_data.sort_values("Created At") if "Created At" in student_data.columns else student_data

    # ── Emotion summary ───────────────────────────────────────────────────
    emotion_counts   = Counter(student_data["Emotion"].dropna())
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "UNKNOWN"
    avg_intensity    = float(student_data["Emotion Intensity Percentage"].mean()) if "Emotion Intensity Percentage" in student_data.columns else 0.5

    recent_emotions: List[str] = student_data["Emotion"].dropna().tail(10).tolist()

    unpleasant_count = sum(1 for e in student_data["Emotion"].dropna() if e in UNPLEASANT_EMOTIONS)
    unpleasant_ratio = unpleasant_count / max(len(student_data), 1)

    # ── Wellness ──────────────────────────────────────────────────────────
    avg_tiredness = float(pd.to_numeric(student_data["Tiredness"], errors="coerce").mean()) if "Tiredness" in student_data.columns else 3.0
    chat_requests = int(student_data["Chat Requested"].astype(str).str.upper().eq("TRUE").sum()) if "Chat Requested" in student_data.columns else 0
    absence_rate  = float(
        student_data["Absencce"].astype(str).str.upper().eq("TRUE").sum() / max(len(student_data), 1)
    ) if "Absencce" in student_data.columns else 0.0

    # ── Trend (linear slope + 7-day rolling) ──────────────────────────────
    trend = _analyze_trend(student_data)

    # ── Resilience: bounce-back rate after dysregulation ──────────────────
    resilience = _compute_bounce_back_rate(student_data)

    # ── Flag-based co-occurrence model ────────────────────────────────────
    dysr_flags = _compute_dysregulation_flags(student_data)

    return {
        "student_id":     student_id,
        "total_checkins": len(student_data),
        "emotions": {
            "dominant_emotion": dominant_emotion,
            "avg_intensity":    round(avg_intensity, 3),
            "emotion_counts":   dict(emotion_counts),
            "unpleasant_ratio": round(unpleasant_ratio, 3),
            "recent_emotions":  recent_emotions,
        },
        "wellness_indicators": {
            "avg_tiredness": round(avg_tiredness, 3),
            "chat_requests": chat_requests,
            "absence_rate":  round(absence_rate, 3),
        },
        "trend_analysis":      trend,
        "resilience":          resilience,
        "dysregulation_flags": dysr_flags,
    }


# ── Helper: linear slope + 7-day rolling average ─────────────────────────

def _analyze_trend(student_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute emotional intensity trend direction (linear slope) and
    a 7-day rolling average for longitudinal pattern detection.

    rolling_7d_avg         — most recent 7-day rolling mean of daily intensity
    longitudinal_direction — compares last-7d rolling avg vs prior 7d window
    """
    base: Dict[str, Any] = {
        "trend":                  "insufficient_data",
        "slope":                  None,
        "rolling_7d_avg":         None,
        "longitudinal_direction": "insufficient_data",
    }

    if "Emotion Intensity Percentage" not in student_data.columns or len(student_data) < 4:
        return base

    intensities = student_data["Emotion Intensity Percentage"].dropna().tolist()
    if len(intensities) < 4:
        return base

    # Linear slope across all check-ins
    x     = np.arange(len(intensities))
    slope = float(np.polyfit(x, intensities, 1)[0])
    trend = "worsening" if slope > 0.005 else ("improving" if slope < -0.005 else "stable")

    # 7-day rolling average (requires Created At timestamps)
    rolling_7d_avg         = None
    longitudinal_direction = "insufficient_data"

    if "Created At" in student_data.columns:
        try:
            daily = (
                student_data.set_index("Created At")["Emotion Intensity Percentage"]
                .resample("D").mean().dropna()
            )
            if len(daily) >= 7:
                rolling = daily.rolling(window=7, min_periods=3).mean().dropna()
                if len(rolling) > 0:
                    rolling_7d_avg = round(float(rolling.iloc[-1]), 3)
                if len(rolling) >= 14:
                    recent_avg  = float(rolling.iloc[-7:].mean())
                    earlier_avg = float(rolling.iloc[-14:-7].mean())
                    if recent_avg > earlier_avg + 0.05:
                        longitudinal_direction = "worsening"
                    elif recent_avg < earlier_avg - 0.05:
                        longitudinal_direction = "improving"
                    else:
                        longitudinal_direction = "stable"
        except Exception:
            pass  # malformed timestamps — skip rolling computation

    return {
        "trend":                  trend,
        "slope":                  round(slope, 5),
        "rolling_7d_avg":         rolling_7d_avg,
        "longitudinal_direction": longitudinal_direction,
    }


# ── Helper: bounce-back rate (resilience metric) ─────────────────────────

def _compute_bounce_back_rate(student_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Resilience metric: average check-ins needed to recover after a
    dysregulation event (unpleasant emotion at intensity >= 0.6).

    Recovery = intensity drops below 0.4 in a subsequent check-in.
    Lower bounce_back_avg_checkins = faster recovery = more resilient.
    None = no dysregulation events found in the student's history.
    """
    if (
        "Emotion" not in student_data.columns
        or "Emotion Intensity Percentage" not in student_data.columns
        or len(student_data) < 3
    ):
        return {"bounce_back_avg_checkins": None, "dysregulation_events": 0}

    sd         = student_data.reset_index(drop=True)
    emotions   = sd["Emotion"].fillna("").astype(str).tolist()
    intensities = pd.to_numeric(sd["Emotion Intensity Percentage"], errors="coerce").fillna(0).tolist()

    dysreg_events   = 0
    recovery_counts = []

    i = 0
    while i < len(sd):
        if emotions[i] in UNPLEASANT_EMOTIONS and intensities[i] >= 0.6:
            dysreg_events += 1
            recovered = False
            for j in range(i + 1, len(sd)):
                if intensities[j] < 0.4:
                    recovery_counts.append(j - i)
                    recovered = True
                    i = j      # advance past recovery point to avoid re-counting
                    break
            if not recovered:
                recovery_counts.append(len(sd) - i)  # never recovered in data window
        i += 1

    avg_bb = round(float(np.mean(recovery_counts)), 2) if recovery_counts else None
    return {
        "bounce_back_avg_checkins": avg_bb,
        "dysregulation_events":     dysreg_events,
    }


# ── Helper: compound dysregulation flag model ─────────────────────────────

def _compute_dysregulation_flags(student_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Flag-based co-occurrence model. For each check-in row, raise flags:
      F1 — unpleasant emotion at high intensity (>= 0.6)
      F2 — high tiredness (>= 4.0 on 1–5 scale)
      F3 — chat support requested

    compound_flag_sessions = rows where 2+ flags fire simultaneously.
    compound_flag_rate     = compound_flag_sessions / total_checkins.
    High compound rate signals students who are distressed across multiple
    dimensions at the same time — a stronger signal than any single flag.
    """
    total = max(len(student_data), 1)

    emotions    = student_data["Emotion"].fillna("").astype(str).tolist() if "Emotion" in student_data.columns else [""] * total
    intensities = pd.to_numeric(student_data["Emotion Intensity Percentage"], errors="coerce").fillna(0).tolist() if "Emotion Intensity Percentage" in student_data.columns else [0.0] * total
    tiredness   = pd.to_numeric(student_data["Tiredness"], errors="coerce").fillna(0).tolist() if "Tiredness" in student_data.columns else [0.0] * total
    chats       = student_data["Chat Requested"].astype(str).str.upper().tolist() if "Chat Requested" in student_data.columns else ["FALSE"] * total

    f1 = f2 = f3 = compound = 0
    for idx in range(total):
        flags = 0
        if emotions[idx] in UNPLEASANT_EMOTIONS and intensities[idx] >= 0.6:
            f1 += 1; flags += 1
        if tiredness[idx] >= 4.0:
            f2 += 1; flags += 1
        if chats[idx] == "TRUE":
            f3 += 1; flags += 1
        if flags >= 2:
            compound += 1

    return {
        "flag_dysregulation_sessions":  f1,
        "flag_high_tiredness_sessions": f2,
        "flag_chat_sessions":           f3,
        "compound_flag_sessions":       compound,
        "compound_flag_rate":           round(compound / total, 3),
    }


# ── Public: build all profiles ────────────────────────────────────────────

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
