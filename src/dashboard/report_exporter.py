"""
================================================================
Author  : @Ranjeeta
Module  : src/dashboard/report_exporter.py
Purpose : Export analysis results to JSON, CSV, and Excel formats
          (Extracted & modularised from main_updated.ipynb Cell 17)
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def _ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def export_class_profiles(class_profiles: Dict[str, Dict], filename: str = "class_emotional_profiles.json") -> str:
    """Save class emotional profiles to JSON."""
    _ensure_outputs_dir()
    output = {
        "analysis_metadata": {
            "analysis_date":          datetime.now().isoformat(),
            "total_classes_analyzed": len(class_profiles),
            "data_source":            "studentCheckIn_cleaned.csv",
            "analysis_type":          "class_based_emotional_profiling",
            "author":                 "@Ranjeeta",
        },
        "class_profiles": class_profiles,
    }
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Exporter] Class profiles saved → {path}")
    return str(path)


def export_recommendations(
    recommendations: Dict[str, Dict],
    teacher_dashboard: Dict,
    filename: str = "class_recommendations.json",
) -> str:
    """Save class recommendations to JSON."""
    _ensure_outputs_dir()
    output = {
        "analysis_metadata": {
            "analysis_date":        datetime.now().isoformat(),
            "total_recommendations": len(recommendations),
            "high_priority_count":  sum(1 for r in recommendations.values() if r.get("priority_level") == "HIGH"),
            "medium_priority_count": sum(1 for r in recommendations.values() if r.get("priority_level") == "MEDIUM"),
            "low_priority_count":   sum(1 for r in recommendations.values() if r.get("priority_level") == "LOW"),
            "author":               "@Ranjeeta",
        },
        "teacher_dashboard":     teacher_dashboard,
        "class_recommendations": recommendations,
    }
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Exporter] Recommendations saved → {path}")
    return str(path)


def export_class_student_map(
    merged_df,
    class_profiles: Dict[str, Dict],
    filename: str = "class_student_map.json",
) -> str:
    """
    Export a mapping of class_id → {class_name, student_ids} so the
    dashboard can filter every student-level view by the selected class.
    Only includes the top-N classes that were profiled (i.e. in class_profiles).
    """
    _ensure_outputs_dir()
    class_map = {}
    for class_id, group in merged_df.groupby("Class ID"):
        str_cid = str(class_id)
        if str_cid not in class_profiles:
            continue
        profile = class_profiles[str_cid]
        class_map[str_cid] = {
            "class_name": profile.get("class_name", str_cid),
            "student_ids": [str(sid) for sid in group["Student ID"].dropna().unique().tolist()],
        }

    output = {
        "analysis_metadata": {
            "analysis_date":   datetime.now().isoformat(),
            "total_classes":   len(class_map),
            "author":          "@Ranjeeta",
        },
        "class_map": class_map,
    }
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Exporter] Class→Student map saved → {path}")
    return str(path)


def export_student_profiles(student_profiles: Dict[str, Dict], filename: str = "student_profiles.json") -> str:
    """
    Save full per-student profiles (including resilience + dysregulation_flags)
    to JSON so the dashboard can display deep-dive and alert views.
    """
    _ensure_outputs_dir()
    output = {
        "analysis_metadata": {
            "analysis_date":     datetime.now().isoformat(),
            "total_students":    len(student_profiles),
            "data_source":       "studentCheckIn_cleaned.csv",
            "analysis_type":     "per_student_emotional_profiling",
            "author":            "@Ranjeeta",
        },
        "student_profiles": student_profiles,
    }
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Exporter] Student profiles saved → {path}")
    return str(path)


def export_risk_predictions(predictions: List[Dict], filename: str = "risk_predictions.json") -> str:
    """Save Phase 2 risk trajectory predictions to JSON."""
    _ensure_outputs_dir()
    output = {
        "analysis_metadata": {
            "analysis_date":      datetime.now().isoformat(),
            "total_students":     len(predictions),
            "escalating_count":   sum(1 for p in predictions if p.get("trajectory") == "escalating"),
            "recovering_count":   sum(1 for p in predictions if p.get("trajectory") == "recovering"),
            "stable_count":       sum(1 for p in predictions if p.get("trajectory") == "stable"),
            "author":             "@Ranjeeta",
        },
        "predictions": predictions,
    }
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Exporter] Risk predictions saved → {path}")
    return str(path)


def export_summary_csv(
    class_profiles: Dict[str, Dict],
    recommendations: Dict[str, Dict],
    filename: str = "class_analysis_summary.csv",
) -> str:
    """Export flat summary table as CSV for spreadsheet review."""
    _ensure_outputs_dir()
    rows = []
    for class_id, profile in class_profiles.items():
        rec = recommendations.get(class_id, {})
        rows.append({
            "class_id":             profile.get("class_id"),
            "class_name":           profile.get("class_name"),
            "total_sessions":       profile.get("total_sessions"),
            "total_students":       profile.get("total_students"),
            "dominant_emotion":     profile["emotions"].get("dominant_emotion"),
            "avg_intensity":        profile["emotions"].get("avg_intensity"),
            "avg_tiredness":        profile["wellness_indicators"].get("avg_tiredness"),
            "absence_rate":         profile["wellness_indicators"].get("absence_rate"),
            "trend":                profile["trend_analysis"].get("trend"),
            "priority_level":       rec.get("priority_level", "UNKNOWN"),
            "recommended_activities": "; ".join(rec.get("recommended_activities", [])),
            "recommended_duration": rec.get("recommended_duration", "N/A"),
            "last_session":         profile.get("last_session"),
        })

    path = OUTPUTS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exporter] Summary CSV saved → {path}")
    return str(path)
