"""
================================================================
Author  : @Ranjeeta
Module  : src/data/loader.py
Purpose : Centralised dataset loader for the EI project
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import pandas as pd
import os
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"


def load_student_checkins(path: str = None) -> pd.DataFrame:
    """Load the cleaned student check-in CSV (757,112 records)."""
    file = path or DATA_DIR / "studentCheckIn_cleaned.csv"
    print(f"[Loader] Loading student check-ins from {file} ...")
    df = pd.read_csv(file)
    print(f"[Loader] Loaded {len(df):,} check-in records.")
    return df


def load_classes(path: str = None) -> pd.DataFrame:
    """Load class reference data (583 classes)."""
    file = path or DATA_DIR / "class.csv"
    return pd.read_csv(file)


def load_sessions(path: str = None) -> pd.DataFrame:
    """Load check-in sessions (31,149 sessions)."""
    file = path or DATA_DIR / "checkInSession.csv"
    return pd.read_csv(file)


def load_students(path: str = None) -> pd.DataFrame:
    """Load student master data (12,467 students)."""
    file = path or DATA_DIR / "students.csv"
    return pd.read_csv(file)


def load_teacher_class(path: str = None) -> pd.DataFrame:
    """Load teacher-class mappings (3,754 records)."""
    file = path or DATA_DIR / "teacherClass.csv"
    return pd.read_csv(file)


def load_wellbeing_responses(path: str = None) -> pd.DataFrame:
    """Load wellbeing survey responses."""
    file = path or DATA_DIR / "wellbeingResponse.csv"
    return pd.read_csv(file)


def load_wellbeing_questions(path: str = None) -> pd.DataFrame:
    """Load wellbeing survey question schema."""
    file = path or DATA_DIR / "wellbeingQuestion.csv"
    return pd.read_csv(file)


def load_switches_summary(path: str = None) -> pd.DataFrame:
    """Load Switch4Schools activity/intervention catalogue."""
    file = path or DATA_DIR / "switches-summary.csv"
    return pd.read_csv(file)


def load_all() -> dict:
    """
    Convenience function — loads all datasets and returns them as a dict.

    Returns
    -------
    dict with keys:
        student_checkins, classes, sessions, students,
        teacher_class, wellbeing_responses, wellbeing_questions, activities
    """
    return {
        "student_checkins":    load_student_checkins(),
        "classes":             load_classes(),
        "sessions":            load_sessions(),
        "students":            load_students(),
        "teacher_class":       load_teacher_class(),
        "wellbeing_responses": load_wellbeing_responses(),
        "wellbeing_questions": load_wellbeing_questions(),
        "activities":          load_switches_summary(),
    }
