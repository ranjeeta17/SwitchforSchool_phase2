"""
================================================================
Author  : @Ranjeeta
Module  : src/risk/risk_predictor.py
Purpose : Phase 2 – Risk Trajectory Prediction
          Calculates Risk Velocity and Escalation Probability Score
          to shift from reactive → preventative wellbeing management
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


# ── Config ─────────────────────────────────────────────────────────────────
PREDICTION_WINDOW_DAYS  = 14   # predict 2 weeks ahead
MIN_DATA_POINTS         = 5    # minimum check-ins needed for prediction
VELOCITY_WEIGHT         = 0.6  # contribution of slope to escalation score
CURRENT_RISK_WEIGHT     = 0.4  # contribution of current risk score


def calculate_risk_velocity(risk_score_series: List[float]) -> float:
    """
    Calculate the slope of a student's risk score over time.

    A positive slope means risk is WORSENING (escalating).
    A negative slope means risk is IMPROVING (recovering).

    Parameters
    ----------
    risk_score_series : List of historical risk scores (oldest → newest)

    Returns
    -------
    float : slope (units: risk score points per check-in)
    """
    if len(risk_score_series) < MIN_DATA_POINTS:
        return 0.0

    x = np.arange(len(risk_score_series)).reshape(-1, 1)
    y = np.array(risk_score_series)

    # Simple least-squares slope (avoids scikit-learn dependency)
    x_flat = x.flatten()
    slope = float(np.polyfit(x_flat, y, 1)[0])
    return round(slope, 4)


def predict_escalation_score(
    current_risk_score: float,
    risk_velocity: float,
    days_ahead: int = PREDICTION_WINDOW_DAYS,
) -> float:
    """
    Predict the Escalation Probability Score (0–100).

    Formula:
        predicted_score = current_score + (velocity × days_ahead)
        escalation_score = clamp(predicted_score, 0, 100)

    Parameters
    ----------
    current_risk_score : Current risk score (0–100)
    risk_velocity      : Risk slope from calculate_risk_velocity()
    days_ahead         : How many days into future to project

    Returns
    -------
    float : Escalation probability score (0–100)
    """
    predicted = current_risk_score + (risk_velocity * days_ahead)
    return round(float(np.clip(predicted, 0, 100)), 1)


def build_risk_history_series(student_data: pd.DataFrame) -> List[float]:
    """
    Build a time-ordered risk score series from raw student check-in data
    for use in calculate_risk_velocity().

    Approximates daily risk from emotion + tiredness data.

    Parameters
    ----------
    student_data : Filtered dataframe for a single student

    Returns
    -------
    List[float] : Risk score per day (0–100 scale)
    """
    from src.data.preprocessor import NEGATIVE_EMOTIONS  # avoid circular import

    student_data = student_data.sort_values("Created At").copy()

    # Group by day
    student_data["date"] = pd.to_datetime(student_data["Created At"]).dt.date
    daily = student_data.groupby("date")

    series = []
    for date, day_data in daily:
        score = 0.0
        neg_ratio = day_data["Emotion"].isin(NEGATIVE_EMOTIONS).mean()
        avg_intensity = day_data["Emotion Intensity Percentage"].mean() if "Emotion Intensity Percentage" in day_data else 0.5
        avg_tiredness = day_data["Tiredness"].mean() if "Tiredness" in day_data else 3.0

        score += neg_ratio * 40
        if avg_tiredness >= 4:
            score += 20
        if avg_intensity > 0.7:
            score += 20

        series.append(min(score, 100))

    return series


def predict_all_students(
    classified_students: List[Dict],
    student_checkins_df: pd.DataFrame,
) -> List[Dict]:
    """
    Augment risk classification results with trajectory predictions.

    Parameters
    ----------
    classified_students : Output from risk_classifier.classify_all_students()
    student_checkins_df : Raw check-in dataframe (must have Student ID, Created At)

    Returns
    -------
    List[Dict] : Each student dict extended with:
        risk_velocity, escalation_probability, prediction_horizon_days
    """
    print(f"[RiskPredictor] Predicting trajectories for {len(classified_students)} students ...")
    enriched = []

    for student in classified_students:
        student_id   = student["student_id"]
        current_risk = student.get("risk_score", 0)

        student_df = student_checkins_df[
            student_checkins_df["Student ID"].astype(str) == str(student_id)
        ]

        history_series = build_risk_history_series(student_df)
        velocity = calculate_risk_velocity(history_series)
        escalation = predict_escalation_score(current_risk, velocity)

        enriched.append({
            **student,
            "risk_velocity":           velocity,
            "escalation_probability":  escalation,
            "prediction_horizon_days": PREDICTION_WINDOW_DAYS,
            "trajectory": "escalating" if velocity > 0.5 else "stable" if abs(velocity) <= 0.5 else "recovering",
        })

    # Sort by escalation probability (highest risk first)
    return sorted(enriched, key=lambda x: x["escalation_probability"], reverse=True)
