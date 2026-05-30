"""
================================================================
Author  : @Quantum hustle
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
import time
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
    export_student_profiles,
    export_class_student_map,
)


def _banner(step, total, msg):
    """Print a clearly visible step banner with step counter."""
    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  [{step}/{total}]  {msg}")
    print(f"{bar}")


def run_phase1():
    """Execute the complete Phase 1 pipeline."""
    t_phase = time.time()
    print("\n" + "═" * 60)
    print("  PHASE 1 — Data Processing & Profiling")
    print("═" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    _banner(1, 9, "Loading raw datasets from datasets/ folder")
    t = time.time()
    datasets = load_all()
    print(f"\n  ✅  {len(datasets)} dataset(s) loaded: {', '.join(datasets.keys())}")
    for name, df in datasets.items():
        if hasattr(df, "__len__"):
            print(f"       • {name}: {len(df):,} rows")
    print(f"  ⏱   Step 1 done in {time.time() - t:.2f}s")

    # ── Step 2: Clean check-ins ────────────────────────────────────────────
    _banner(2, 9, "Cleaning & validating student check-ins")
    t = time.time()
    raw_rows = len(datasets["student_checkins"])
    clean_df = clean_checkins(datasets["student_checkins"])
    dropped  = raw_rows - len(clean_df)
    pct      = dropped / max(raw_rows, 1) * 100
    print(f"\n  ✅  {len(clean_df):,} clean rows  ({dropped:,} dropped — {pct:.1f}% removed)")
    print(f"  ⏱   Step 2 done in {time.time() - t:.2f}s")

    # ── Step 3: Merge datasets ─────────────────────────────────────────────
    _banner(3, 9, "Merging check-ins + classes + sessions + students")
    t = time.time()
    merged_df = merge_datasets(
        clean_df,
        datasets["classes"],
        datasets["sessions"],
        datasets["students"],
    )
    n_classes  = merged_df["Class ID"].nunique()   if "Class ID"   in merged_df.columns else "?"
    n_students = merged_df["Student ID"].nunique() if "Student ID" in merged_df.columns else "?"
    print(f"\n  ✅  Merged dataset: {len(merged_df):,} rows × {len(merged_df.columns)} columns")
    print(f"       Unique classes:  {n_classes}")
    print(f"       Unique students: {n_students:,}")
    print(f"  ⏱   Step 3 done in {time.time() - t:.2f}s")

    # ── Step 4: Build class profiles ──────────────────────────────────────
    _banner(4, 9, "Building class emotional profiles (top 100 classes)")
    t = time.time()
    class_profiles = build_all_class_profiles(merged_df, top_n=100)
    print(f"\n  ✅  {len(class_profiles):,} class profiles built")
    print(f"  ⏱   Step 4 done in {time.time() - t:.2f}s")

    # ── Step 5: Build student profiles ────────────────────────────────────
    _banner(5, 9, "Building per-student profiles (resilience + flags + 7d trend)")
    print("  ⏳  Processing each student — resilience, compound dysregulation flags,")
    print("      7-day rolling intensity average. This may take 30–90 seconds ...")
    t = time.time()
    student_profiles = build_all_student_profiles(merged_df)
    has_rolling    = sum(1 for sp in student_profiles.values()
                         if sp.get("trend_analysis", {}).get("rolling_7d_avg") is not None)
    has_resilience = sum(1 for sp in student_profiles.values()
                         if sp.get("resilience", {}).get("bounce_back_avg_checkins") is not None)
    n_stu = len(student_profiles)
    print(f"\n  ✅  {n_stu:,} student profiles created")
    print(f"       7d rolling avg populated:   {has_rolling:,} / {n_stu:,} ({has_rolling/max(n_stu,1)*100:.0f}%)")
    print(f"       Resilience score populated:  {has_resilience:,} / {n_stu:,} ({has_resilience/max(n_stu,1)*100:.0f}%)")
    print(f"  ⏱   Step 5 done in {time.time() - t:.2f}s")

    # ── Step 6: Classify student risk ─────────────────────────────────────
    _banner(6, 9, "Classifying student risk levels (rule-based scorer)")
    t = time.time()
    classified_students = classify_all_students(student_profiles)
    high_n   = sum(1 for s in classified_students if s.get("risk_level") == "HIGH")
    medium_n = sum(1 for s in classified_students if s.get("risk_level") == "MEDIUM")
    low_n    = sum(1 for s in classified_students if s.get("risk_level") == "LOW")
    total_c  = max(len(classified_students), 1)
    print(f"\n  ✅  {len(classified_students):,} students classified")
    print(f"       🔴 HIGH:   {high_n:,}  ({high_n/total_c*100:.1f}%)")
    print(f"       🟡 MEDIUM: {medium_n:,}  ({medium_n/total_c*100:.1f}%)")
    print(f"       🟢 LOW:    {low_n:,}  ({low_n/total_c*100:.1f}%)")
    print(f"  ⏱   Step 6 done in {time.time() - t:.2f}s")

    # ── Step 7: Generate class recommendations ─────────────────────────────
    _banner(7, 9, "Generating class activity recommendations")
    t = time.time()
    recommendations = generate_all_recommendations(class_profiles)
    h = sum(1 for r in recommendations.values() if r.get("priority_level") == "HIGH")
    m = sum(1 for r in recommendations.values() if r.get("priority_level") == "MEDIUM")
    l = sum(1 for r in recommendations.values() if r.get("priority_level") == "LOW")
    print(f"\n  ✅  {len(recommendations):,} recommendations generated")
    print(f"       HIGH: {h}  |  MEDIUM: {m}  |  LOW: {l}")
    print(f"  ⏱   Step 7 done in {time.time() - t:.2f}s")

    # ── Step 8: Build teacher dashboard summary ────────────────────────────
    _banner(8, 9, "Building teacher dashboard summary")
    t = time.time()
    dashboard = build_teacher_dashboard_data(class_profiles, recommendations)
    print(f"\n  ✅  Dashboard summary built")
    print(f"  ⏱   Step 8 done in {time.time() - t:.2f}s")

    # ── Step 9: Export all outputs ─────────────────────────────────────────
    _banner(9, 9, "Exporting all outputs to outputs/ folder")
    t = time.time()
    export_class_profiles(class_profiles)
    export_recommendations(recommendations, dashboard)
    export_summary_csv(class_profiles, recommendations)
    export_student_profiles(student_profiles)
    export_class_student_map(merged_df, class_profiles)
    print(f"\n  ✅  All output files saved")
    print(f"       • outputs/class_emotional_profiles.json")
    print(f"       • outputs/class_recommendations.json")
    print(f"       • outputs/class_analysis_summary.csv")
    print(f"       • outputs/student_profiles.json")
    print(f"       • outputs/class_student_map.json")
    print(f"  ⏱   Step 9 done in {time.time() - t:.2f}s")

    t_total = time.time() - t_phase
    print(f"\n{'═'*60}")
    print(f"  ✅  PHASE 1 COMPLETE in {t_total:.1f}s")
    print(f"  📁  Check the outputs/ folder for results")
    print(f"{'═'*60}")
    return merged_df, class_profiles, student_profiles, classified_students


def run_phase2(merged_df, classified_students, student_profiles):
    """Execute the Phase 2 pipeline — predictive intelligence."""
    t_phase = time.time()
    print("\n" + "═" * 60)
    print("  PHASE 2 — Predictive Wellbeing Intelligence")
    print("═" * 60)

    try:
        from src.risk.risk_predictor import predict_all_students
        from src.recommendations.teacher_prioritiser import build_priority_list
        from src.dashboard.report_exporter import export_risk_predictions

        # ── Step 1: Predict 14-day trajectories ───────────────────────────
        _banner(1, 4, "Predicting 14-day risk trajectories (velocity + escalation model)")
        t = time.time()
        predictions = predict_all_students(classified_students, merged_df)
        esc  = sum(1 for p in predictions if p.get("trajectory") == "escalating")
        rec  = sum(1 for p in predictions if p.get("trajectory") == "recovering")
        stab = sum(1 for p in predictions if p.get("trajectory") == "stable")
        print(f"\n  ✅  {len(predictions):,} predictions generated")
        print(f"       📈 Escalating: {esc:,}  |  📉 Recovering: {rec:,}  |  ➡ Stable: {stab:,}")
        print(f"  ⏱   Step 1 done in {time.time() - t:.2f}s")

        # ── Step 2: Build priority list ───────────────────────────────────
        _banner(2, 4, "Building teacher priority list (composite scoring)")
        t = time.time()
        priority_list = build_priority_list(predictions, student_profiles)
        print(f"\n  ✅  Priority list: {len(priority_list):,} students ranked")
        print(f"  🔴  Top 5 Students Requiring Immediate Attention:")
        for i, s in enumerate(priority_list[:5], 1):
            print(f"       {i}. {s['student_id'][:12]}…  "
                  f"Score: {s['priority_score']}  |  {s['suggested_action']}")
        print(f"  ⏱   Step 2 done in {time.time() - t:.2f}s")

        # ── Step 3: Export risk predictions ───────────────────────────────
        _banner(3, 4, "Merging priority scores and exporting risk predictions")
        t = time.time()
        for pred in predictions:
            sid   = pred["student_id"]
            match = next((p for p in priority_list if p["student_id"] == sid), {})
            pred["priority_score"]   = match.get("priority_score", 0)
            pred["suggested_action"] = match.get("suggested_action", "")
        export_risk_predictions(predictions)
        print(f"\n  ✅  Risk predictions exported → outputs/risk_predictions.json")
        print(f"  ⏱   Step 3 done in {time.time() - t:.2f}s")

        # ── Step 4: Student clustering (K=2 to K=9) ───────────────────────
        _banner(4, 4, "Clustering students by behavioural pattern (K=2 through K=9)")
        t = time.time()
        try:
            from src.profiling.student_clusterer import cluster_students
            print("  ⏳  Running K-Means, Agglomerative, DBSCAN, GMM ...")
            print("      Generating PCA scatter visualisations for K=2 through K=9 ...")
            cluster_students(student_profiles, output_dir="outputs")
            print(f"\n  ✅  Clustering complete → outputs/student_clusters.json")
            print(f"       PCA images: outputs/cluster_kmeans_k2_pca.png  …  cluster_kmeans_k9_pca.png")
            print(f"  ⏱   Step 4 done in {time.time() - t:.2f}s")
        except ImportError as cluster_err:
            print(f"\n  ⚠️   Clustering skipped — missing dependency: {cluster_err}")
            print(f"       Run: pip install scikit-learn scipy statsmodels")

        t_total = time.time() - t_phase
        print(f"\n{'═'*60}")
        print(f"  ✅  PHASE 2 COMPLETE in {t_total:.1f}s")
        print(f"  📁  outputs/risk_predictions.json  +  outputs/student_clusters.json")
        print(f"{'═'*60}")

    except ImportError as e:
        print(f"\n  ⚠️   Phase 2 Skipped — missing dependency: {e}")
        print("       Run: pip install scikit-learn scipy statsmodels")


if __name__ == "__main__":
    t_start = time.time()
    print(f"\n{'█' * 60}")
    print(f"  Switch4Schools — AI Wellbeing Pipeline")
    print(f"  IFN735 Team 29  |  @Ranjeeta")
    print(f"{'█' * 60}")

    merged_df, class_profiles, student_profiles, classified_students = run_phase1()
    run_phase2(merged_df, classified_students, student_profiles)

    t_all = time.time() - t_start
    print(f"\n{'█' * 60}")
    print(f"  🎉  FULL PIPELINE COMPLETE in {t_all:.1f}s")
    print(f"  🚀  Launch dashboard with:")
    print(f"       streamlit run app/app.py")
    print(f"{'█' * 60}\n")
