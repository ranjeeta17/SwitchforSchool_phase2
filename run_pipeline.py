"""
================================================================
Author  : @Ranjeeta
Module  : run_pipeline.py
Purpose : End-to-end pipeline runner
          Executes Phase 1 + Phase 2 in sequence and saves outputs
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29

Usage:
    python run_pipeline.py
================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_all
from src.data.preprocessor import clean_checkins, merge_datasets
from src.profiling.class_profiler import build_all_class_profiles
from src.profiling.student_profiler import build_all_student_profiles
from src.risk.risk_classifier import classify_all_students
from src.recommendations.activity_recommender import generate_all_recommendations
from src.dashboard.teacher_dashboard import build_teacher_dashboard_data
from src.dashboard.report_exporter import (
    export_class_profiles,
    export_recommendations,
    export_summary_csv,
)


def run_phase1():
    """Execute the complete Phase 1 pipeline."""
    print("\n" + "=" * 60)
    print("  PHASE 1 — Data Processing & Profiling")
    print("=" * 60 + "\n")

    # Step 1: Load data
    datasets = load_all()

    # Step 2: Clean check-ins
    clean_df = clean_checkins(datasets["student_checkins"])

    # Step 3: Merge datasets
    merged_df = merge_datasets(
        clean_df,
        datasets["classes"],
        datasets["sessions"],
        datasets["students"],
    )

    # Step 4: Build class profiles
    class_profiles = build_all_class_profiles(merged_df, top_n=100)

    # Step 5: Build student profiles
    student_profiles = build_all_student_profiles(merged_df)

    # Step 6: Classify student risk
    classified_students = classify_all_students(student_profiles)
    print(f"\n[Pipeline] Risk classification complete: {len(classified_students)} students")

    # Step 7: Generate class recommendations
    recommendations = generate_all_recommendations(class_profiles)

    # Step 8: Build teacher dashboard summary
    dashboard = build_teacher_dashboard_data(class_profiles, recommendations)

    # Step 9: Export all outputs
    export_class_profiles(class_profiles)
    export_recommendations(recommendations, dashboard)
    export_summary_csv(class_profiles, recommendations)

    print("\n✅ Phase 1 complete. Check the outputs/ folder for results.")
    return merged_df, class_profiles, student_profiles, classified_students


def run_phase2(merged_df, classified_students, student_profiles):
    """Execute the Phase 2 pipeline — predictive intelligence."""
    print("\n" + "=" * 60)
    print("  PHASE 2 — Predictive Wellbeing Intelligence")
    print("=" * 60 + "\n")

    try:
        from src.risk.risk_predictor import predict_all_students
        from src.recommendations.teacher_prioritiser import build_priority_list
        from src.dashboard.report_exporter import export_risk_predictions

        # Step 1: Predict trajectory for all students
        predictions = predict_all_students(classified_students, merged_df)

        # Step 2: Build teacher priority list
        priority_list = build_priority_list(predictions, student_profiles)
        print(f"\n[Pipeline] Priority list built for {len(priority_list)} students")
        top5 = priority_list[:5]
        for i, s in enumerate(top5, 1):
            print(f"  {i}. Student {s['student_id'][:8]}… | Score: {s['priority_score']} | {s['suggested_action']}")

        # Step 3: Export predictions with priority scores merged
        for pred in predictions:
            sid = pred["student_id"]
            match = next((p for p in priority_list if p["student_id"] == sid), {})
            pred["priority_score"]   = match.get("priority_score", 0)
            pred["suggested_action"] = match.get("suggested_action", "")

        export_risk_predictions(predictions)
        print("\n✅ Phase 2 complete. Check outputs/risk_predictions.json")

    except ImportError as e:
        print(f"[Phase 2] Skipped — missing dependency: {e}")
        print("Run: pip install scikit-learn scipy statsmodels")


if __name__ == "__main__":
    merged_df, class_profiles, student_profiles, classified_students = run_phase1()
    run_phase2(merged_df, classified_students, student_profiles)
    print("\n🎉 Full pipeline complete. Launch dashboard with:")
    print("   streamlit run app/app.py\n")
