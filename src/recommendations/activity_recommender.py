"""
================================================================
Author  : @Ranjeeta
Module  : src/recommendations/activity_recommender.py
Purpose : Phase 1 emotion-to-activity rule-based recommendation engine
          (Extracted & modularised from main_updated.ipynb Cell 13)
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

from typing import Dict, Any, List


# ── Activity catalogue (maps dominant emotion → intervention) ──────────────
ACTIVITY_RECOMMENDATIONS = {
    "HAPPY": {
        "activities":    ["Creative projects", "Group discussions", "Peer teaching", "Achievement sharing"],
        "focus":         "Channel positive energy into productive learning",
        "duration":      "5-10 minutes",
        "priority":      "LOW",
        "benefits":      "Maintains positive momentum and reinforces learning",
    },
    "EXCITED": {
        "activities":    ["Energising activities", "Physical movement breaks", "Collaborative challenges", "Quick competitions"],
        "focus":         "Direct high energy into structured learning tasks",
        "duration":      "10-15 minutes",
        "priority":      "LOW",
        "benefits":      "Harnesses excitement for enhanced engagement",
    },
    "CALM": {
        "activities":    ["Deep focus tasks", "Individual reading", "Reflective writing", "Mindful learning"],
        "focus":         "Leverage calm state for deep learning",
        "duration":      "15-20 minutes",
        "priority":      "LOW",
        "benefits":      "Optimal state for complex problem-solving",
    },
    "SAD": {
        "activities":    ["Gentle discussions", "Mindfulness exercises", "Art therapy", "Support group activities"],
        "focus":         "Provide emotional support and gentle engagement",
        "duration":      "10-15 minutes",
        "priority":      "HIGH",
        "benefits":      "Addresses emotional needs while maintaining learning momentum",
    },
    "ANXIOUS": {
        "activities":    ["Breathing exercises", "Structured routines", "Clear instructions", "Stress relief techniques"],
        "focus":         "Create calming environment and reduce stress",
        "duration":      "5-10 minutes",
        "priority":      "HIGH",
        "benefits":      "Reduces anxiety and improves focus and participation",
    },
    "SCARED": {
        "activities":    ["Safe space discussions", "Gradual exposure activities", "Trust building exercises", "Encouragement sessions"],
        "focus":         "Build confidence and create safe learning environment",
        "duration":      "10-20 minutes",
        "priority":      "HIGH",
        "benefits":      "Addresses fears and builds student confidence",
    },
    "ANGRY": {
        "activities":    ["Conflict resolution", "Emotion regulation techniques", "Physical activities", "Cooling down exercises"],
        "focus":         "Help students manage anger and redirect energy",
        "duration":      "15-20 minutes",
        "priority":      "HIGH",
        "benefits":      "Teaches emotional regulation and conflict management",
    },
    "BORED": {
        "activities":    ["Brain teasers", "Creative challenges", "Interest-based tasks", "Hands-on projects"],
        "focus":         "Re-engage students with stimulating activities",
        "duration":      "10-15 minutes",
        "priority":      "MEDIUM",
        "benefits":      "Reignites curiosity and intrinsic motivation",
    },
    "CONFUSED": {
        "activities":    ["Step-by-step explanations", "Peer tutoring", "Visual aids", "Q&A sessions"],
        "focus":         "Clarify concepts and build understanding",
        "duration":      "10-15 minutes",
        "priority":      "MEDIUM",
        "benefits":      "Restores confidence and comprehension",
    },
    "UNKNOWN": {
        "activities":    ["Check-in conversation", "General wellbeing chat", "Open classroom discussion"],
        "focus":         "Gather more information about student state",
        "duration":      "5-10 minutes",
        "priority":      "MEDIUM",
        "benefits":      "Improves data completeness and teacher awareness",
    },
}


def generate_class_recommendation(class_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured activity recommendation for one class.

    Parameters
    ----------
    class_profile : Dict from class_profiler.create_class_profile()

    Returns
    -------
    Dict with: class_id, class_name, priority_level, emotional_state,
               recommended_activities, recommended_duration, rationale
    """
    class_id   = class_profile.get("class_id", "unknown")
    class_name = class_profile.get("class_name") or f"Unnamed_{class_id[:8]}"

    dominant_emotion = class_profile["emotions"].get("dominant_emotion", "UNKNOWN")
    avg_intensity    = class_profile["emotions"].get("avg_intensity", 0.5)
    avg_tiredness    = class_profile["wellness_indicators"].get("avg_tiredness", 3.0)
    trend            = class_profile["trend_analysis"].get("trend", "stable")
    absence_rate     = class_profile["wellness_indicators"].get("absence_rate", 0.0)

    base_rec = ACTIVITY_RECOMMENDATIONS.get(dominant_emotion, ACTIVITY_RECOMMENDATIONS["UNKNOWN"])

    # ── Priority override ─────────────────────────────────────────────────
    # Base priority is set by dominant emotion (SAD/ANXIOUS/ANGRY/SCARED = HIGH,
    # BORED/CONFUSED = MEDIUM, HAPPY/EXCITED/CALM = LOW)
    priority = base_rec["priority"]

    # Declining trend upgrades LOW → MEDIUM (not to HIGH by itself)
    if trend == "declining" and priority == "LOW":
        priority = "MEDIUM"

    # High tiredness upgrades LOW → MEDIUM, MEDIUM stays MEDIUM
    if avg_tiredness >= 4.5 and priority == "LOW":
        priority = "MEDIUM"

    # Severe absence rate (>15%) upgrades to HIGH
    if absence_rate >= 0.15:
        priority = "HIGH"

    # ── Rationale ─────────────────────────────────────────────────────────
    rationale_parts = [
        f"Class is predominantly {dominant_emotion.lower()} (intensity: {avg_intensity:.0%})",
        f"Trend: {trend}",
        f"Avg tiredness: {avg_tiredness:.1f}/5",
    ]
    if absence_rate > 0.1:
        rationale_parts.append(f"Notable absence rate: {absence_rate:.0%}")

    return {
        "class_id":              class_id,
        "class_name":            class_name,
        "priority_level":        priority,
        "emotional_state":       dominant_emotion,
        "recommended_activities": base_rec["activities"],
        "recommended_duration":  base_rec["duration"],
        "activity_focus":        base_rec["focus"],
        "expected_benefits":     base_rec["benefits"],
        "rationale":             ". ".join(rationale_parts),
    }


def generate_all_recommendations(class_profiles: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Generate recommendations for every class profile.

    Returns
    -------
    dict of { class_id: recommendation_dict }
    """
    print(f"[Recommender] Generating recommendations for {len(class_profiles)} classes ...")
    recommendations = {}
    for class_id, profile in class_profiles.items():
        recommendations[class_id] = generate_class_recommendation(profile)

    high  = sum(1 for r in recommendations.values() if r["priority_level"] == "HIGH")
    med   = sum(1 for r in recommendations.values() if r["priority_level"] == "MEDIUM")
    low   = sum(1 for r in recommendations.values() if r["priority_level"] == "LOW")
    print(f"[Recommender] Done. HIGH={high}  MEDIUM={med}  LOW={low}")
    return recommendations
