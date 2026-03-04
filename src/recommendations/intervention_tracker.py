"""
================================================================
Author  : @Ranjeeta
Module  : src/recommendations/intervention_tracker.py
Purpose : Phase 2 – Intervention Effectiveness Tracking
          Records before/after emotional state for each intervention
          and calculates success rates per student profile type
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import json
import os
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from pathlib import Path


LOG_PATH = Path(__file__).resolve().parents[3] / "outputs" / "intervention_log.json"
NEGATIVE_EMOTIONS = {"SAD", "ANXIOUS", "ANGRY", "SCARED"}


def _load_log() -> List[Dict]:
    """Load the intervention log from disk (creates empty file if missing)."""
    if not LOG_PATH.exists():
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w") as f:
            json.dump([], f)
        return []
    with open(LOG_PATH, "r") as f:
        return json.load(f)


def _save_log(log: List[Dict]) -> None:
    """Persist the intervention log to disk."""
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def log_intervention(
    student_id: str,
    activity_name: str,
    emotion_before: str,
    intensity_before: float,
) -> Dict:
    """
    Record that an intervention was applied and capture the baseline state.

    Call this when a teacher applies a recommended activity.
    Follow up with record_outcome() after observing the student.

    Returns the log entry dict (with null outcome fields until record_outcome is called).
    """
    log = _load_log()
    entry = {
        "id":               f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "student_id":       student_id,
        "activity_name":    activity_name,
        "applied_date":     str(date.today()),
        "emotion_before":   emotion_before,
        "intensity_before": intensity_before,
        "emotion_after":    None,
        "intensity_after":  None,
        "outcome":          None,   # "improved" | "no_change" | "worsened"
        "recorded_at":      None,
    }
    log.append(entry)
    _save_log(log)
    print(f"[InterventionTracker] Logged: {activity_name} for student {student_id}")
    return entry


def record_outcome(
    intervention_id: str,
    emotion_after: str,
    intensity_after: float,
) -> Optional[Dict]:
    """
    Record the emotional state observed AFTER applying an intervention.

    Parameters
    ----------
    intervention_id : The 'id' field from the log entry returned by log_intervention()
    emotion_after   : Observed emotion label post-intervention
    intensity_after : Observed intensity (0–1) post-intervention

    Returns
    -------
    Updated log entry, or None if ID not found
    """
    log = _load_log()
    for entry in log:
        if entry["id"] == intervention_id:
            entry["emotion_after"]   = emotion_after
            entry["intensity_after"] = intensity_after
            entry["recorded_at"]     = str(datetime.now())

            # Determine outcome
            was_negative_before = entry["emotion_before"] in NEGATIVE_EMOTIONS
            is_negative_after   = emotion_after in NEGATIVE_EMOTIONS
            if was_negative_before and not is_negative_after:
                entry["outcome"] = "improved"
            elif was_negative_before and is_negative_after and intensity_after < entry["intensity_before"]:
                entry["outcome"] = "improved"
            elif intensity_after < entry["intensity_before"] - 0.1:
                entry["outcome"] = "improved"
            elif intensity_after > entry["intensity_before"] + 0.1:
                entry["outcome"] = "worsened"
            else:
                entry["outcome"] = "no_change"

            _save_log(log)
            print(f"[InterventionTracker] Outcome recorded: {entry['outcome']}")
            return entry

    print(f"[InterventionTracker] Warning: intervention_id {intervention_id} not found.")
    return None


def calculate_effectiveness(activity_name: str, student_id: Optional[str] = None) -> float:
    """
    Calculate the success rate (0.0–1.0) for a given activity.

    Parameters
    ----------
    activity_name : Name of the activity/intervention
    student_id    : If provided, filter to this student only; otherwise class-wide

    Returns
    -------
    float : success_rate (proportion of 'improved' outcomes)
    """
    log = _load_log()
    relevant = [
        e for e in log
        if e["activity_name"] == activity_name
        and e["outcome"] is not None
        and (student_id is None or e["student_id"] == student_id)
    ]
    if not relevant:
        return 0.0
    improved = sum(1 for e in relevant if e["outcome"] == "improved")
    return round(improved / len(relevant), 3)


def get_best_interventions(student_id: str, top_n: int = 3) -> List[Dict]:
    """
    Return the top N most effective interventions for a student,
    based on historical outcome data.

    Returns
    -------
    List of dicts with { activity_name, success_rate, sample_size }
    """
    log = _load_log()
    student_log = [e for e in log if e["student_id"] == student_id and e["outcome"] is not None]
    activities = set(e["activity_name"] for e in student_log)

    results = []
    for activity in activities:
        rate = calculate_effectiveness(activity, student_id)
        count = sum(1 for e in student_log if e["activity_name"] == activity)
        results.append({"activity_name": activity, "success_rate": rate, "sample_size": count})

    return sorted(results, key=lambda x: x["success_rate"], reverse=True)[:top_n]
