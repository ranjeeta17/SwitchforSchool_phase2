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
HIGH_RISK_INTENSITY_THRESHOLD  = 0.7    # avg intensity above this → distress signal
HIGH_RISK_TIREDNESS_THRESHOLD  = 4.0    # avg tiredness above this (scale 1–5)
HIGH_RISK_NEGATIVE_RATIO       = 0.4    # >40 % check-ins are negative
HIGH_RISK_ABSENCE_RATE         = 0.3    # >30 % sessions missed

MEDIUM_RISK_INTENSITY_THRESHOLD = 0.5
MEDIUM_RISK_TIREDNESS_THRESHOLD  = 3.0
MEDIUM_RISK_NEGATIVE_RATIO       = 0.2
MEDIUM_RISK_ABSENCE_RATE         = 0.15

NEGATIVE_EMOTIONS = {"SAD", "ANXIOUS", "ANGRY", "SCARED"}
DECLINING_TRENDS  = {"declining", "worsening"}


def classify_student_risk(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply Phase 1 rule-based risk classification to a student profile.

    Rules (in order of priority):
    ──────────────────────────────
    HIGH Risk:   low avg mood AND (high tiredness OR high negative ratio OR high absence)
    MEDIUM Risk: occasional negative emotions OR mild fatigue/disengagement
    LOW Risk:    consistent positive emotions and steady energy

    Returns
    -------
    dict: { risk_level, risk_score (0–100), risk_factors, recommendations }
    """
    emotions   = profile.get("emotions", {})
    wellness   = profile.get("wellness_indicators", {})
    trend      = profile.get("trend_analysis", {})

    avg_intensity    = emotions.get("avg_intensity", 0.5)
    negative_ratio   = emotions.get("negative_ratio", 0.0)
    dominant_emotion = emotions.get("dominant_emotion", "HAPPY")
    avg_tiredness    = wellness.get("avg_tiredness", 3.0)
    absence_rate     = wellness.get("absence_rate", 0.0)
    trend_direction  = trend.get("trend", "stable")

    risk_score   = 0
    risk_factors: List[str] = []

    # ── Score accumulation ─────────────────────────────────────────────────
    if dominant_emotion in NEGATIVE_EMOTIONS:
        risk_score += 30
        risk_factors.append(f"Dominant emotion is negative ({dominant_emotion})")

    if avg_intensity > HIGH_RISK_INTENSITY_THRESHOLD:
        risk_score += 20
        risk_factors.append(f"High avg emotion intensity ({avg_intensity:.2f})")

    if avg_tiredness >= HIGH_RISK_TIREDNESS_THRESHOLD:
        risk_score += 20
        risk_factors.append(f"High tiredness level ({avg_tiredness:.1f}/5)")

    if negative_ratio >= HIGH_RISK_NEGATIVE_RATIO:
        risk_score += 20
        risk_factors.append(f"High proportion of negative check-ins ({negative_ratio:.0%})")

    if absence_rate >= HIGH_RISK_ABSENCE_RATE:
        risk_score += 15
        risk_factors.append(f"High absence rate ({absence_rate:.0%})")

    if trend_direction in DECLINING_TRENDS:
        risk_score += 15
        risk_factors.append(f"Emotional trend is {trend_direction}")

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
