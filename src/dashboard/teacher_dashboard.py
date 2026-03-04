"""
================================================================
Author  : @Ranjeeta
Module  : src/dashboard/teacher_dashboard.py
Purpose : Matplotlib + Plotly visualisation builder for teachers
          (Extracted & modularised from main_updated.ipynb Cell 11 & 15)
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from collections import Counter
from typing import Dict, List, Any


PRIORITY_COLORS = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
EMOTION_COLORS  = {
    "HAPPY": "#F1C40F", "EXCITED": "#E67E22", "CALM": "#3498DB",
    "SAD": "#9B59B6",   "ANXIOUS": "#E74C3C", "ANGRY": "#C0392B",
    "SCARED": "#8E44AD", "BORED": "#95A5A6",  "CONFUSED": "#1ABC9C",
}


def plot_teacher_dashboard(
    class_profiles: Dict[str, Dict],
    recommendations: Dict[str, Dict],
    save_path: str = None,
) -> None:
    """
    Render the 6-panel Teacher Dashboard.

    Panels
    ------
    [0,0] Priority level distribution (pie)
    [0,1] Emotional state distribution (bar)
    [0,2] High-priority classes (horizontal bar)
    [1,0] Most recommended activities (horizontal bar)
    [1,1] Duration recommendations (bar)
    [1,2] Trend status overview (bar)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Teacher Dashboard — Class Wellbeing Overview", fontsize=20, fontweight="bold", y=0.98)

    recs = list(recommendations.values())

    # ── [0,0] Priority pie ────────────────────────────────────────────────
    ax = axes[0, 0]
    priority_counts = Counter(r["priority_level"] for r in recs)
    colors = [PRIORITY_COLORS.get(k, "grey") for k in priority_counts.keys()]
    ax.pie(priority_counts.values(), labels=priority_counts.keys(),
           autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Class Priority Distribution", fontweight="bold")

    # ── [0,1] Emotion distribution ────────────────────────────────────────
    ax = axes[0, 1]
    emotion_counts = Counter(r["emotional_state"] for r in recs)
    bar_colors = [EMOTION_COLORS.get(e, "#BDC3C7") for e in emotion_counts.keys()]
    bars = ax.bar(emotion_counts.keys(), emotion_counts.values(), color=bar_colors)
    ax.set_title("Emotional States Across Classes", fontweight="bold")
    ax.set_ylabel("Number of Classes")
    ax.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, emotion_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha="center", va="bottom", fontweight="bold")

    # ── [0,2] High-priority classes ───────────────────────────────────────
    ax = axes[0, 2]
    high_priority = [r["class_name"] for r in recs if r["priority_level"] == "HIGH"]
    if high_priority:
        ax.barh(range(len(high_priority)), [1] * len(high_priority), color=PRIORITY_COLORS["HIGH"], alpha=0.7)
        ax.set_yticks(range(len(high_priority)))
        ax.set_yticklabels([n[:28] + "…" if len(n) > 28 else n for n in high_priority])
        ax.set_title(f"High Priority Classes ({len(high_priority)})", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No High Priority Classes ✓", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="green")
        ax.set_title("High Priority Classes", fontweight="bold")

    # ── [1,0] Top activities ──────────────────────────────────────────────
    ax = axes[1, 0]
    all_activities = []
    for r in recs:
        all_activities.extend(r.get("recommended_activities", []))
    activity_counts = pd.Series(all_activities).value_counts().head(10)
    ax.barh(range(len(activity_counts)), activity_counts.values, color="#5DADE2")
    ax.set_yticks(range(len(activity_counts)))
    ax.set_yticklabels([a[:22] + "…" if len(a) > 22 else a for a in activity_counts.index])
    ax.set_title("Most Recommended Activities", fontweight="bold")
    ax.set_xlabel("Frequency")

    # ── [1,1] Duration distribution ───────────────────────────────────────
    ax = axes[1, 1]
    duration_counts = Counter(r.get("recommended_duration", "N/A") for r in recs)
    ax.bar(duration_counts.keys(), duration_counts.values(), color="#A29BFE")
    ax.set_title("Duration Recommendations", fontweight="bold")
    ax.set_ylabel("Number of Classes")
    ax.tick_params(axis="x", rotation=30)

    # ── [1,2] Trend overview ──────────────────────────────────────────────
    ax = axes[1, 2]
    trend_counts = Counter(
        p["trend_analysis"].get("trend", "unknown")
        for p in class_profiles.values()
    )
    trend_colors = {"improving": "#27AE60", "stable": "#3498DB", "declining": "#E74C3C", "insufficient_data": "#95A5A6"}
    colors_list = [trend_colors.get(t, "#BDC3C7") for t in trend_counts.keys()]
    ax.bar(trend_counts.keys(), trend_counts.values(), color=colors_list)
    ax.set_title("Class Emotional Trend Overview", fontweight="bold")
    ax.set_ylabel("Number of Classes")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Dashboard] Saved to {save_path}")
    plt.show()


def build_teacher_dashboard_data(
    class_profiles: Dict[str, Dict],
    recommendations: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Build the serialisable teacher dashboard dict (for JSON export).

    Returns
    -------
    dict with: overview, priority_breakdown, emotional_breakdown, high_priority_list
    """
    recs = list(recommendations.values())
    high   = [r for r in recs if r["priority_level"] == "HIGH"]
    medium = [r for r in recs if r["priority_level"] == "MEDIUM"]
    low    = [r for r in recs if r["priority_level"] == "LOW"]

    return {
        "overview": {
            "total_classes":      len(class_profiles),
            "high_priority":      len(high),
            "medium_priority":    len(medium),
            "low_priority":       len(low),
        },
        "priority_breakdown": {
            "HIGH":   len(high),
            "MEDIUM": len(medium),
            "LOW":    len(low),
        },
        "emotional_breakdown": dict(Counter(r["emotional_state"] for r in recs)),
        "high_priority_list": [
            {"class_id": r["class_id"], "class_name": r["class_name"], "emotional_state": r["emotional_state"]}
            for r in high
        ],
    }
