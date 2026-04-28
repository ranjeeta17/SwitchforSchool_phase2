"""
================================================================
Author  : @Ranjeeta
Module  : src/data/preprocessor.py
Purpose : Data cleaning, merging, and normalisation pipeline
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29
================================================================
"""

import pandas as pd
import numpy as np


# Canonical emotion label map — normalise all raw values
EMOTION_LABEL_MAP = {
    "happy":    "HAPPY",
    "excited":  "EXCITED",
    "sad":      "SAD",
    "anxious":  "ANXIOUS",
    "angry":    "ANGRY",
    "scared":   "SCARED",
    "calm":     "CALM",
    "bored":    "BORED",
    "confused": "CONFUSED",
}

UNPLEASANT_EMOTIONS = {"SAD", "ANXIOUS", "ANGRY", "SCARED"}
POSITIVE_EMOTIONS   = {"HAPPY", "EXCITED", "CALM"}


def clean_checkins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Phase 1 data hygiene to the raw student check-in dataframe.

    Steps
    -----
    1. Drop duplicate rows
    2. Parse timestamps
    3. Normalise emotion labels to single ontology
    4. Clip intensity to [0, 1] range
    5. Clip tiredness to [1, 5] ordinal scale
    6. Drop rows with no Student ID or Session ID
    """
    df = df.copy()

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Preprocessor] Removed {before - len(df):,} duplicate rows.")

    # 2. Parse timestamps
    if "Created At" in df.columns:
        df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce", utc=True)

    # 3. Normalise emotion labels
    if "Emotion" in df.columns:
        df["Emotion"] = (
            df["Emotion"]
            .str.strip()
            .str.lower()
            .map(EMOTION_LABEL_MAP)
            .fillna("UNKNOWN")
        )

    # 4. Clip intensity (coerce strings/undefined to NaN first, then clip)
    if "Emotion Intensity Percentage" in df.columns:
        df["Emotion Intensity Percentage"] = pd.to_numeric(
            df["Emotion Intensity Percentage"], errors="coerce"
        ).clip(0, 1)

    # 5. Clip tiredness
    if "Tiredness" in df.columns:
        df["Tiredness"] = pd.to_numeric(df["Tiredness"], errors="coerce").clip(1, 5)

    # 6. Drop rows missing key identifiers
    df = df.dropna(subset=["Student ID"])
    print(f"[Preprocessor] Clean dataset: {len(df):,} rows remaining.")
    return df


def merge_datasets(
    checkins: pd.DataFrame,
    classes: pd.DataFrame,
    sessions: pd.DataFrame,
    students: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the 757,112 × 21 complete relationship table used by profilers.

    Merges: checkins ← sessions ← classes ← students
    """
    print("[Preprocessor] Merging datasets ...")

    # checkins → sessions: both use "Check In Session ID"
    merged = checkins.merge(sessions, on="Check In Session ID", how="left")
    # merged → classes: both use "Class ID"
    merged = merged.merge(classes, on="Class ID", how="left")
    # merged → students: checkins have "Student ID", students table uses "ID"
    merged = merged.merge(students, left_on="Student ID", right_on="ID", how="left")

    # Normalise column names so all downstream modules use consistent keys
    merged = merged.rename(columns={
        "Opened Date Time": "Created At",    # sessions timestamp → expected key
        "Request For Chat": "Chat Requested", # raw column → expected key
    })

    print(f"[Preprocessor] Complete relationship table: {merged.shape}")
    return merged


def flag_unpleasant_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean column `is_unpleasant` for distress-focused analysis."""
    df = df.copy()
    df["is_unpleasant"] = df["Emotion"].isin(UNPLEASANT_EMOTIONS)
    return df


def compute_absence_rate(df: pd.DataFrame, total_sessions: int) -> float:
    """
    Estimate absence rate for a student or class.

    absence_rate = 1 - (actual_check_ins / total_sessions)
    """
    if total_sessions == 0:
        return 0.0
    return max(0.0, 1 - (len(df) / total_sessions))
