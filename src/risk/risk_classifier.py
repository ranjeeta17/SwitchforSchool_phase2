"""
================================================================
Author  : @Ranjeeta
Module  : src/risk/risk_classifier.py
Purpose : Phase 1 rule-based risk classification (Low/Medium/High)
          Rule logic from project report Section 1.3
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

from typing import Dict, Any, List

# ── Thresholds (school-tunable) ────────────────────────────────────────────
# HIGH intensity of an UNPLEASANT emotion = dysregulation (risk).
# LOW intensity = regulation (calm/peace) — NOT a risk signal.
HIGH_RISK_UNPLEASANT_INTENSITY = 0.6    # avg intensity >= this WITH unpleasant dominant = dysregulation
HIGH_RISK_TIREDNESS_THRESHOLD  = 4.5    # avg tiredness above this (scale 1–5)
HIGH_RISK_UNPLEASANT_RATIO     = 0.4    # >40 % check-ins are unpleasant emotions
HIGH_RISK_ABSENCE_RATE         = 0.15   # >15 % absence rate

MEDIUM_RISK_TIREDNESS_THRESHOLD  = 3.5
MEDIUM_RISK_UNPLEASANT_RATIO     = 0.2
MEDIUM_RISK_ABSENCE_RATE         = 0.05

UNPLEASANT_EMOTIONS = {"SAD", "ANXIOUS", "ANGRY", "SCARED"}
DECLINING_TRENDS    = {"declining", "worsening"}


def classify_student_risk(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply rule-based risk classification to a student profile.

    Score components:
    ──────────────────────────────────────────────────────────────
    +20  Dominant emotion is unpleasant
    +25  High intensity unpleasant (dysregulation signal)
    +20  High tiredness (>= 4.5)
    +20  High unpleasant ratio (>= 40%)
    +15  High absence rate (>= 15%)
    +15  Linear trend is worsening/declining
    +10  7-day longitudinal trend is worsening
    +10–20 Compound flag rate (2+ risk flags co-occurring)
    +8–15  Slow bounce-back after dysregulation
    +5–15  Chat support requests

    Thresholds: HIGH >= 50, MEDIUM >= 25, LOW < 25

    Returns
    -------
    dict: { risk_level, risk_score (0–100), risk_factors }
    """
    emotions   = profile.get("emotions", {})
    wellness   = profile.get("wellness_indicators", {})
    trend      = profile.get("trend_analysis", {})
    resilience = profile.get("resilience", {})
    dysr_flags = profile.get("dysregulation_flags", {})

    avg_intensity        = emotions.get("avg_intensity", 0.5)
    unpleasant_ratio     = emotions.get("unpleasant_ratio", 0.0)
    dominant_emotion     = emotions.get("dominant_emotion", "HAPPY")
    avg_tiredness        = wellness.get("avg_tiredness", 3.0)
    absence_rate         = wellness.get("absence_rate", 0.0)
    chat_requests        = wellness.get("chat_requests", 0)
    trend_direction      = trend.get("trend", "stable")
    longitudinal         = trend.get("longitudinal_direction", "insufficient_data")
    bounce_back          = resilience.get("bounce_back_avg_checkins")
    compound_rate        = dysr_flags.get("compound_flag_rate", 0.0)

    risk_score   = 0
    risk_factors: List[str] = []

    # ── Score accumulation ─────────────────────────────────────────────────

    if dominant_emotion in UNPLEASANT_EMOTIONS:
        risk_score += 20
        risk_factors.append(f"Dominant emotion is unpleasant ({dominant_emotion})")

    # Primary dysregulation signal: HIGH intensity + unpleasant dominant
    if dominant_emotion in UNPLEASANT_EMOTIONS and avg_intensity >= HIGH_RISK_UNPLEASANT_INTENSITY:
        risk_score += 25
        risk_factors.append(
            f"High intensity unpleasant emotion ({dominant_emotion} at {avg_intensity:.2f}) — dysregulation signal"
        )

    if avg_tiredness >= HIGH_RISK_TIREDNESS_THRESHOLD:
        risk_score += 20
        risk_factors.append(f"High tiredness level ({avg_tiredness:.1f}/5)")

    if unpleasant_ratio >= HIGH_RISK_UNPLEASANT_RATIO:
        risk_score += 20
        risk_factors.append(f"High proportion of unpleasant check-ins ({unpleasant_ratio:.0%})")

    if absence_rate >= HIGH_RISK_ABSENCE_RATE:
        risk_score += 15
        risk_factors.append(f"High absence rate ({absence_rate:.0%})")

    if trend_direction in DECLINING_TRENDS:
        risk_score += 15
        risk_factors.append(f"Emotional trend is {trend_direction}")

    # 7-day longitudinal direction (recent window — separate from overall slope)
    if longitudinal == "worsening":
        risk_score += 10
        risk_factors.append("7-day rolling trend is worsening")

    # Compound flag co-occurrence: multiple stressors in the same session
    if compound_rate >= 0.10:
        risk_score += 20
        risk_factors.append(f"High compound dysregulation rate ({compound_rate:.0%} of sessions have 2+ risk flags)")
    elif compound_rate >= 0.05:
        risk_score += 10
        risk_factors.append(f"Moderate compound dysregulation rate ({compound_rate:.0%} of sessions)")

    # Bounce-back rate: slow recovery = low resilience
    if bounce_back is not None and bounce_back >= 5:
        risk_score += 15
        risk_factors.append(f"Slow emotional recovery after dysregulation (avg {bounce_back:.1f} check-ins to recover)")
    elif bounce_back is not None and bounce_back >= 3:
        risk_score += 8
        risk_factors.append(f"Moderate emotional recovery speed (avg {bounce_back:.1f} check-ins)")

    if chat_requests > 0:
        risk_score += min(chat_requests * 5, 15)
        risk_factors.append(f"Student requested {chat_requests} wellbeing chat(s)")

    # ── Classification ─────────────────────────────────────────────────────
    if risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_score": min(risk_score, 100),
        "risk_factors": risk_factors,
    }


def classify_all_students(profiles: Dict[str, Dict]) -> List[Dict]:
    """
    Run risk classification across all student profiles.

    Returns list of dicts sorted by risk_score (highest first).
    """
    results = []
    for student_id, profile in profiles.items():
        classification = classify_student_risk(profile)
        results.append({
            "student_id":  student_id,
            **classification,
        })

    return sorted(results, key=lambda x: x["risk_score"], reverse=True)
