"""
================================================================
Author  : @Ranjeeta
Module  : tests/test_risk_classifier.py
Purpose : Unit tests for Phase 1 rule-based risk classification
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
Run:
    pytest tests/test_risk_classifier.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk.risk_classifier import classify_student_risk


def _make_profile(
    dominant_emotion: str = "HAPPY",
    avg_intensity: float = 0.5,
    negative_ratio: float = 0.1,
    avg_tiredness: float = 2.5,
    absence_rate: float = 0.05,
    trend: str = "stable",
) -> dict:
    """Helper: build a minimal student profile dict for testing."""
    return {
        "emotions": {
            "dominant_emotion": dominant_emotion,
            "avg_intensity":    avg_intensity,
            "negative_ratio":   negative_ratio,
        },
        "wellness_indicators": {
            "avg_tiredness": avg_tiredness,
            "absence_rate":  absence_rate,
        },
        "trend_analysis": {
            "trend": trend,
        },
    }


# ── LOW Risk ──────────────────────────────────────────────────────────────
def test_low_risk_happy_student():
    profile = _make_profile(dominant_emotion="HAPPY", avg_intensity=0.4, negative_ratio=0.05)
    result  = classify_student_risk(profile)
    assert result["risk_level"] == "LOW", f"Expected LOW, got {result['risk_level']}"
    assert result["risk_score"] < 25


def test_low_risk_excited_student():
    profile = _make_profile(dominant_emotion="EXCITED", avg_intensity=0.6, negative_ratio=0.08)
    result  = classify_student_risk(profile)
    assert result["risk_level"] == "LOW"


# ── MEDIUM Risk ───────────────────────────────────────────────────────────
def test_medium_risk_declining_trend():
    profile = _make_profile(dominant_emotion="HAPPY", avg_intensity=0.4, negative_ratio=0.25, trend="declining")
    result  = classify_student_risk(profile)
    assert result["risk_level"] in {"MEDIUM", "HIGH"}, f"Expected MEDIUM/HIGH, got {result['risk_level']}"


def test_medium_risk_moderate_tiredness():
    profile = _make_profile(dominant_emotion="CALM", avg_intensity=0.45, avg_tiredness=3.5, negative_ratio=0.22)
    result  = classify_student_risk(profile)
    assert result["risk_level"] in {"MEDIUM", "LOW"}


# ── HIGH Risk ─────────────────────────────────────────────────────────────
def test_high_risk_anxious_student():
    profile = _make_profile(
        dominant_emotion="ANXIOUS",
        avg_intensity=0.8,
        negative_ratio=0.5,
        avg_tiredness=4.5,
        absence_rate=0.35,
        trend="declining",
    )
    result = classify_student_risk(profile)
    assert result["risk_level"] == "HIGH", f"Expected HIGH, got {result['risk_level']}"
    assert result["risk_score"] >= 50


def test_high_risk_sad_high_absence():
    profile = _make_profile(
        dominant_emotion="SAD",
        avg_intensity=0.75,
        absence_rate=0.40,
        negative_ratio=0.55,
    )
    result = classify_student_risk(profile)
    assert result["risk_level"] == "HIGH"


# ── Edge cases ────────────────────────────────────────────────────────────
def test_unknown_emotion_defaults():
    profile = _make_profile(dominant_emotion="UNKNOWN")
    result  = classify_student_risk(profile)
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= result["risk_score"] <= 100


def test_risk_score_bounded():
    """Risk score should never exceed 100."""
    profile = _make_profile(
        dominant_emotion="ANGRY",
        avg_intensity=1.0,
        negative_ratio=1.0,
        avg_tiredness=5.0,
        absence_rate=1.0,
        trend="declining",
    )
    result = classify_student_risk(profile)
    assert result["risk_score"] <= 100


def test_risk_factors_provided():
    """Risk factors list should be populated for non-LOW students."""
    profile = _make_profile(dominant_emotion="SAD", negative_ratio=0.6, avg_tiredness=4.5)
    result  = classify_student_risk(profile)
    assert isinstance(result["risk_factors"], list)
    assert len(result["risk_factors"]) > 0
