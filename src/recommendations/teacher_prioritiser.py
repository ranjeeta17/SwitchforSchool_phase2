"""
================================================================
Author  : @Ranjeeta
Module  : src/recommendations/teacher_prioritiser.py
Purpose : Phase 2 – Teacher Prioritisation Engine
          Computes a composite Priority Score to rank who to 
          see first, saving teacher time and driving engagement.
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

from typing import Dict, Any, List


# ── Weighting formula ──────────────────────────────────────────────────────
# Priority Score = (Current Risk × 0.40) + (Escalation Probability × 0.35)
#                + (Absence Rate × 0.15)  + (Volatility × 0.10)
WEIGHT_CURRENT_RISK  = 0.40
WEIGHT_ESCALATION    = 0.35
WEIGHT_ABSENCE       = 0.15
WEIGHT_VOLATILITY    = 0.10


def compute_priority_score(
    current_risk_score: float,
    escalation_probability: float,
    absence_rate: float,
    volatility: float,
) -> float:
    """
    Compute a 0–100 composite Priority Score for teacher triage.

    Parameters
    ----------
    current_risk_score     : Phase 1 risk score (0–100)
    escalation_probability : Phase 2 predicted escalation score (0–100)
    absence_rate           : Proportion of sessions missed (0–1), scaled ×100
    volatility             : Emotional intensity std dev (0–1), scaled ×100

    Returns
    -------
    float : Priority Score (0–100)
    """
    score = (
        current_risk_score          * WEIGHT_CURRENT_RISK
        + escalation_probability    * WEIGHT_ESCALATION
        + (absence_rate    * 100)   * WEIGHT_ABSENCE
        + (volatility      * 100)   * WEIGHT_VOLATILITY
    )
    return round(min(score, 100), 1)


def build_priority_list(
    student_predictions: List[Dict],
    student_profiles: Dict[str, Dict],
) -> List[Dict]:
    """
    Generate a ranked, teacher-facing priority list.

    Parameters
    ----------
    student_predictions : Output from risk_predictor.predict_all_students()
    student_profiles    : Output from student_profiler.build_all_student_profiles()

    Returns
    -------
    List of dicts sorted by priority_score (highest first), each with:
        student_id, priority_score, risk_level, trajectory,
        suggested_action, reasons
    """
    ranked = []

    for prediction in student_predictions:
        student_id  = prediction["student_id"]
        profile     = student_profiles.get(student_id, {})
        wellness    = profile.get("wellness_indicators", {})
        trend_info  = profile.get("trend_analysis", {})

        current_risk    = prediction.get("risk_score", 0)
        escalation_prob = prediction.get("escalation_probability", 0)
        absence_rate    = wellness.get("absence_rate", 0.0)
        volatility      = trend_info.get("slope", 0.0) or 0.0
        volatility      = abs(volatility)  # use magnitude

        priority_score = compute_priority_score(
            current_risk, escalation_prob, absence_rate, volatility
        )

        # ── Suggested action ───────────────────────────────────────────────
        if priority_score >= 70:
            action = "Immediate 1-on-1 wellbeing check-in today"
        elif priority_score >= 45:
            action = "Schedule wellbeing conversation this week"
        elif priority_score >= 20:
            action = "Keep on watch list; passive monitoring"
        else:
            action = "No immediate action needed"

        # ── Reasons summary (human-readable) ──────────────────────────────
        reasons = prediction.get("risk_factors", [])
        trajectory = prediction.get("trajectory", "stable")
        if trajectory == "escalating":
            reasons.append(f"Risk trajectory is escalating (velocity: {prediction.get('risk_velocity', 0):+.2f})")

        ranked.append({
            "student_id":         student_id,
            "priority_score":     priority_score,
            "risk_level":         prediction.get("risk_level", "LOW"),
            "trajectory":         trajectory,
            "escalation_probability": escalation_prob,
            "suggested_action":   action,
            "reasons":            reasons,
        })

    return sorted(ranked, key=lambda x: x["priority_score"], reverse=True)
