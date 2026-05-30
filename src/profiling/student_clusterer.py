"""
================================================================
Author  : @Ranjeeta
Module  : src/profiling/student_clusterer.py
Purpose : Cluster students by emotional/behavioural patterns
          using multiple algorithms — K-Means, DBSCAN,
          Agglomerative (Hierarchical), and GMM — with full
          visualisation and algorithm comparison.
Project : IFN735 – AI-Powered Emotional Intelligence Support
          Switch4Schools | Team 29 | Epic E2
================================================================

Cluster Archetypes (K=2, confirmed by client):
  • Stable              — balanced across all features (~95% of students)
  • Emotionally Distressed — high unpleasant ratio + high intensity (~5% of students)
================================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm

from typing import Dict, Any, List, Tuple
from collections import Counter

# ── Project root (works whether called as a script or imported) ──
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, "..", ".."))


def _log(msg: str, step: str = "Clusterer"):
    """Print a timestamped progress line so the user knows what is happening."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {msg}", flush=True)

# ── sklearn clustering ────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)

warnings.filterwarnings("ignore")

# ── Palette & constants ───────────────────────────────────────
CLUSTER_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#6A4C93", "#8AC926", "#FF595E",
]
ALL_EMOTIONS = ["HAPPY", "EXCITED", "CALM", "SAD", "ANXIOUS",
                "ANGRY", "SCARED", "BORED", "CONFUSED"]
ARCHETYPE_LABELS = {
    "distressed": "Emotionally Distressed",
    "stable":     "Stable",
}
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — Feature Engineering
# ══════════════════════════════════════════════════════════════

def build_feature_matrix(
    student_profiles: Dict[str, Dict],
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Convert raw student profiles into a numerical feature matrix.

    Features (8 total):
        unpleasant_ratio        — fraction of check-ins with unpleasant emotion
        avg_intensity           — mean emotion intensity (0–1)
        avg_tiredness           — mean tiredness (1–5)
        absence_rate            — fraction of sessions absent
        chat_requests_norm      — chat requests / total check-ins
        trend_slope             — linear slope of intensity over time
        emotion_diversity       — number of distinct emotions shown
        dominant_unpleasant_flag — 1 if dominant emotion is unpleasant

    Returns
    -------
    feature_df   : raw (unscaled) DataFrame with student_id index
    X_scaled     : StandardScaler-normalised numpy array
    scaler       : fitted StandardScaler instance
    """
    rows = []
    for sid, profile in student_profiles.items():
        em  = profile.get("emotions", {})
        wl  = profile.get("wellness_indicators", {})
        tr  = profile.get("trend_analysis", {})
        total = max(profile.get("total_checkins", 1), 1)

        unpleasant_ratio = em.get("unpleasant_ratio", 0.0)
        avg_int          = em.get("avg_intensity",    0.5)
        avg_tired        = wl.get("avg_tiredness",    3.0)
        absence          = wl.get("absence_rate",     0.0)
        chat_norm        = wl.get("chat_requests",    0) / total
        slope            = tr.get("slope", 0.0) if tr.get("slope") is not None else 0.0
        dom_emotion      = em.get("dominant_emotion", "HAPPY")
        emotion_counts   = em.get("emotion_counts",   {})
        diversity        = len(emotion_counts)
        dom_unpleasant_flag = 1 if dom_emotion in {"SAD", "ANXIOUS", "ANGRY", "SCARED"} else 0

        rows.append({
            "student_id":            sid,
            "unpleasant_ratio":      unpleasant_ratio,
            "avg_intensity":         avg_int,
            "avg_tiredness":         avg_tired,
            "absence_rate":          absence,
            "chat_requests_norm":    chat_norm,
            "trend_slope":           slope,
            "emotion_diversity":     diversity,
            "dominant_unpleasant_flag": dom_unpleasant_flag,
        })

    feature_df = pd.DataFrame(rows).set_index("student_id")
    feature_cols = [c for c in feature_df.columns]
    feature_df[feature_cols] = feature_df[feature_cols].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df[feature_cols])

    _log(f"Feature matrix ready: {X_scaled.shape[0]:,} students x {X_scaled.shape[1]} features")
    return feature_df, X_scaled, scaler


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — Optimal K Selection (Elbow + Silhouette)
# ══════════════════════════════════════════════════════════════

def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    output_dir: str = OUTPUT_DIR,
) -> int:
    """
    Run K-Means for each K in k_range and plot the Elbow curve
    alongside Silhouette scores to recommend the best K.

    Returns the K with the highest Silhouette score.
    """
    _log("Searching for optimal K — testing K=2 to K=9 ...")
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        silhouettes.append(sil)
        _log(f"  K={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}", step="K-Search")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    _log(f"Best K selected: {best_k} (highest silhouette)")

    # ── Plot ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimal K Selection — K-Means", fontsize=15, fontweight="bold", y=1.02)

    ks = list(k_range)
    ax1.plot(ks, inertias, "o-", color="#E63946", linewidth=2, markersize=7)
    ax1.axvline(best_k, color="#457B9D", linestyle="--", alpha=0.7, label=f"Best K = {best_k}")
    ax1.set_title("Elbow Curve (Inertia)")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    bar_colors = ["#2A9D8F" if k == best_k else "#A8DADC" for k in ks]
    ax2.bar(ks, silhouettes, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_title("Silhouette Score per K")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score (higher = better)")
    ax2.set_xticks(ks)
    for i, (k, s) in enumerate(zip(ks, silhouettes)):
        ax2.text(k, s + 0.002, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_optimal_k.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved elbow/silhouette chart -> {path}")
    return best_k


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — Clustering Algorithms
# ══════════════════════════════════════════════════════════════

def run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    _log(f"Running K-Means (k={k}) ...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X_scaled)


def run_dbscan(X_scaled: np.ndarray, eps: float = 1.2, min_samples: int = 10) -> np.ndarray:
    _log(f"Running DBSCAN (eps={eps}, min_samples={min_samples}) ...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    _log(f"DBSCAN done -> {n_clusters} clusters, {n_noise:,} noise/outlier points")
    return labels


def run_agglomerative(X_scaled: np.ndarray, k: int) -> np.ndarray:
    _log(f"Running Agglomerative / Hierarchical Clustering (k={k}, linkage=ward) ...")
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return agg.fit_predict(X_scaled)


def run_gmm(X_scaled: np.ndarray, k: int) -> np.ndarray:
    _log(f"Running Gaussian Mixture Model (k={k}) ...")
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    return gmm.fit_predict(X_scaled)


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — PCA Projection (for 2-D visualisation)
# ══════════════════════════════════════════════════════════════

def compute_pca(X_scaled: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    _log(f"PCA 2D projection done — variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}")
    return X_pca, pca


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — Metric Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_labels(X_scaled: np.ndarray, labels: np.ndarray, name: str) -> Dict[str, Any]:
    """Compute three standard internal clustering quality metrics."""
    valid = labels[labels != -1]   # exclude DBSCAN noise
    X_valid = X_scaled[labels != -1]
    n_clusters = len(set(valid))
    if n_clusters < 2 or len(X_valid) < 2:
        return {"algorithm": name, "n_clusters": n_clusters,
                "silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {
        "algorithm":          name,
        "n_clusters":         n_clusters,
        "silhouette":         round(float(silhouette_score(X_valid, valid)),      4),
        "calinski_harabasz":  round(float(calinski_harabasz_score(X_valid, valid)), 2),
        "davies_bouldin":     round(float(davies_bouldin_score(X_valid, valid)),  4),
    }


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — Archetype Labelling
# ══════════════════════════════════════════════════════════════

def assign_archetypes(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
) -> Dict[int, str]:
    """
    Inspect cluster centroids in the original feature space and assign
    a human-readable archetype label to each cluster ID.
    """
    feature_df = feature_df.copy()
    feature_df["cluster"] = labels
    centroids = feature_df.groupby("cluster").mean(numeric_only=True)

    archetype_map: Dict[int, str] = {}
    for cid, row in centroids.iterrows():
        if cid == -1:
            archetype_map[cid] = "Noise / Outlier"
            continue
        unpleasant = row.get("unpleasant_ratio", 0)
        intensity  = row.get("avg_intensity", 0.5)

        # HIGH intensity of unpleasant emotions = dysregulation = Emotionally Distressed
        if unpleasant >= 0.30 and intensity >= 0.50:
            label = "Emotionally Distressed"
        else:
            label = "Stable"
        archetype_map[cid] = label

    return archetype_map


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — Visualisations
# ══════════════════════════════════════════════════════════════

def _scatter_pca(
    ax: plt.Axes,
    X_pca: np.ndarray,
    labels: np.ndarray,
    title: str,
    archetype_map: Dict[int, str],
    pca: PCA,
):
    """Draw a 2-D PCA scatter coloured by cluster on the given Axes."""
    unique_labels = sorted(set(labels))
    for i, cid in enumerate(unique_labels):
        mask  = labels == cid
        color = "#BBBBBB" if cid == -1 else CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        lname = archetype_map.get(cid, f"Cluster {cid}")
        cnt   = int(mask.sum())
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, alpha=0.55, s=12, label=f"{lname} (n={cnt:,})", edgecolors="none",
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=7, markerscale=2, loc="upper right")
    ax.grid(True, alpha=0.2)


def plot_all_algorithms(
    X_pca: np.ndarray,
    pca: PCA,
    all_results: Dict[str, Dict],
    output_dir: str = OUTPUT_DIR,
):
    """
    2×2 grid of PCA scatter plots — one per algorithm.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Student Clusters — PCA Projection by Algorithm", fontsize=16, fontweight="bold", y=1.01)

    pairs = list(all_results.items())
    for ax, (algo_name, res) in zip(axes.flat, pairs):
        _scatter_pca(ax, X_pca, res["labels"], algo_name, res["archetypes"], pca)

    # hide unused axes if fewer than 4 algorithms
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_all_algorithms_pca.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved all-algorithms PCA grid -> {path}")


def plot_silhouette_samples(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    algo_name: str,
    archetype_map: Dict[int, str],
    output_dir: str = OUTPUT_DIR,
):
    """
    Per-sample silhouette plot — shows within-cluster cohesion quality
    as a horizontal bar chart grouped by cluster.
    """
    valid_mask = labels != -1
    X_v = X_scaled[valid_mask]
    L_v = labels[valid_mask]
    if len(set(L_v)) < 2:
        return

    sample_sils = silhouette_samples(X_v, L_v)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for cid in sorted(set(L_v)):
        vals = np.sort(sample_sils[L_v == cid])
        y_upper = y_lower + len(vals)
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=color, edgecolor=color, alpha=0.8)
        ax.text(-0.05, y_lower + len(vals) / 2,
                f"{archetype_map.get(cid, cid)}", ha="right", fontsize=8)
        y_lower = y_upper + 10

    mean_sil = float(silhouette_score(X_v, L_v))
    ax.axvline(mean_sil, color="red", linestyle="--", label=f"Mean = {mean_sil:.3f}")
    ax.set_title(f"Silhouette Plot — {algo_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Students")
    ax.legend(); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    safe = algo_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"cluster_silhouette_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved -> {os.path.basename(path)}")


def plot_cluster_profiles(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    algo_name: str,
    archetype_map: Dict[int, str],
    output_dir: str = OUTPUT_DIR,
):
    """
    Radar (spider) chart showing mean feature values per cluster,
    plus a heatmap of centroid values.
    """
    df = feature_df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]
    centroids = df.groupby("cluster").mean(numeric_only=True)

    features = list(centroids.columns)
    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
    ax_radar = fig.add_subplot(gs[0], polar=True)
    ax_heat  = fig.add_subplot(gs[1])

    # ── Radar ────────────────────────────────────────────────
    for cid, row in centroids.iterrows():
        vals = row[features].tolist()
        vals += vals[:1]
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        label = archetype_map.get(cid, f"Cluster {cid}")
        ax_radar.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax_radar.fill(angles, vals, alpha=0.12, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(
        [f.replace("_", "\n") for f in features], fontsize=8
    )
    ax_radar.set_title(f"Cluster Profiles — {algo_name}\n(Radar Chart)",
                       fontsize=11, fontweight="bold", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # ── Heatmap ──────────────────────────────────────────────
    heat_data = centroids[features].copy()
    heat_data.index = [archetype_map.get(i, f"C{i}") for i in heat_data.index]
    im = ax_heat.imshow(heat_data.values, cmap="RdYlGn_r", aspect="auto")
    ax_heat.set_xticks(range(len(features)))
    ax_heat.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
    ax_heat.set_yticks(range(len(heat_data)))
    ax_heat.set_yticklabels(heat_data.index, fontsize=9)
    ax_heat.set_title("Cluster Centroid Heatmap\n(Brighter = Higher Value)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, shrink=0.8)
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            val = heat_data.values[i, j]
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(val) > heat_data.values.std() else "black")

    plt.tight_layout()
    safe = algo_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"cluster_profiles_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved -> {os.path.basename(path)}")


def plot_cluster_size_distribution(
    all_results: Dict[str, Dict],
    output_dir: str = OUTPUT_DIR,
):
    """
    Side-by-side bar charts showing how many students fall in each
    cluster per algorithm, coloured by archetype.
    """
    n_algos = len(all_results)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 6))
    if n_algos == 1:
        axes = [axes]

    fig.suptitle("Cluster Size Distribution by Algorithm",
                 fontsize=14, fontweight="bold")

    for ax, (algo_name, res) in zip(axes, all_results.items()):
        labels = res["labels"]
        arch   = res["archetypes"]
        counts = Counter(labels)
        sorted_cids = sorted([c for c in counts if c != -1])
        if -1 in counts:
            sorted_cids.append(-1)
        names = [arch.get(c, f"C{c}") for c in sorted_cids]
        sizes = [counts[c] for c in sorted_cids]
        colors = [CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)] if c != -1 else "#BBBBBB"
                  for c in sorted_cids]

        bars = ax.bar(range(len(names)), sizes, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_title(algo_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Students")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, sz in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                    f"{sz:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_size_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved -> {os.path.basename(path)}")


def plot_algorithm_comparison(
    metrics: List[Dict[str, Any]],
    output_dir: str = OUTPUT_DIR,
):
    """
    Compare all algorithms on three metrics:
      • Silhouette Score     (higher is better)
      • Calinski-Harabasz    (higher is better)
      • Davies-Bouldin       (lower is better)
    """
    valid = [m for m in metrics if m["silhouette"] is not None]
    algos = [m["algorithm"] for m in valid]
    sil   = [m["silhouette"] for m in valid]
    ch    = [m["calinski_harabasz"] for m in valid]
    db    = [m["davies_bouldin"] for m in valid]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Clustering Algorithm Comparison", fontsize=14, fontweight="bold")
    x = np.arange(len(algos))
    bar_kw = dict(edgecolor="white", linewidth=1.2)

    # Silhouette — higher better
    best_sil = np.argmax(sil)
    colors = ["#2A9D8F" if i == best_sil else "#A8DADC" for i in range(len(algos))]
    ax1.bar(x, sil, color=colors, **bar_kw)
    ax1.set_xticks(x); ax1.set_xticklabels(algos, rotation=15, ha="right")
    ax1.set_title("Silhouette Score\n(higher = better)", fontweight="bold")
    ax1.set_ylabel("Score"); ax1.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(sil):
        ax1.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    # Calinski-Harabász — higher better
    best_ch = np.argmax(ch)
    colors = ["#E9C46A" if i == best_ch else "#FAE8C0" for i in range(len(algos))]
    ax2.bar(x, ch, color=colors, **bar_kw)
    ax2.set_xticks(x); ax2.set_xticklabels(algos, rotation=15, ha="right")
    ax2.set_title("Calinski-Harabász Index\n(higher = better)", fontweight="bold")
    ax2.set_ylabel("Score"); ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(ch):
        ax2.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # Davies-Bouldin — lower better
    best_db = np.argmin(db)
    colors = ["#E63946" if i == best_db else "#F4BFBF" for i in range(len(algos))]
    ax3.bar(x, db, color=colors, **bar_kw)
    ax3.set_xticks(x); ax3.set_xticklabels(algos, rotation=15, ha="right")
    ax3.set_title("Davies-Bouldin Score\n(lower = better)", fontweight="bold")
    ax3.set_ylabel("Score"); ax3.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(db):
        ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_algorithm_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved -> {os.path.basename(path)}")


def plot_emotion_breakdown_per_cluster(
    feature_df: pd.DataFrame,
    student_profiles: Dict[str, Dict],
    labels: np.ndarray,
    algo_name: str,
    archetype_map: Dict[int, str],
    output_dir: str = OUTPUT_DIR,
):
    """
    Stacked bar chart showing dominant emotion distribution within each cluster.
    Reveals what emotional patterns drive each archetype.
    """
    rows = []
    df = feature_df.copy()
    df["cluster"] = labels
    for sid in df.index:
        cid = int(df.loc[sid, "cluster"])
        if cid == -1:
            continue
        dom = student_profiles[sid].get("emotions", {}).get("dominant_emotion", "UNKNOWN")
        rows.append({"cluster": cid, "emotion": dom})

    if not rows:
        return

    emotion_df = pd.DataFrame(rows)
    pivot = emotion_df.groupby(["cluster", "emotion"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)   # normalise to %
    pivot.index = [archetype_map.get(i, f"C{i}") for i in pivot.index]

    # Colour map for emotions
    emotion_colors = {
        "HAPPY":   "#FFD166", "EXCITED": "#06D6A0", "CALM":    "#118AB2",
        "SAD":     "#EF476F", "ANXIOUS": "#FF8C00", "ANGRY":   "#C1121F",
        "SCARED":  "#9B5DE5", "BORED":   "#AAAAAA", "CONFUSED":"#5C677D",
    }
    cols_in_data = [e for e in ALL_EMOTIONS if e in pivot.columns]
    remaining    = [c for c in pivot.columns if c not in cols_in_data]
    pivot = pivot[cols_in_data + remaining]

    colors = [emotion_colors.get(c, "#999999") for c in pivot.columns]
    ax = pivot.plot(kind="bar", stacked=True, color=colors, figsize=(12, 6),
                    edgecolor="white", linewidth=0.5)
    plt.title(f"Dominant Emotion Breakdown per Cluster — {algo_name}",
              fontsize=13, fontweight="bold")
    plt.xlabel("Cluster Archetype")
    plt.ylabel("Proportion of Students")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Dominant Emotion", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    safe = algo_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"cluster_emotion_breakdown_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved -> {os.path.basename(path)}")


def plot_3d_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    algo_name: str,
    archetype_map: Dict[int, str],
    output_dir: str = OUTPUT_DIR,
):
    """3-D PCA scatter for added depth of view."""
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    pca3 = PCA(n_components=3, random_state=42)
    X3   = pca3.fit_transform(X_scaled)

    fig  = plt.figure(figsize=(10, 7))
    ax   = fig.add_subplot(111, projection="3d")
    for cid in sorted(set(labels)):
        mask  = labels == cid
        color = "#BBBBBB" if cid == -1 else CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        name  = archetype_map.get(cid, f"Cluster {cid}")
        ax.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2],
                   c=color, alpha=0.5, s=8, label=name, edgecolors="none")

    var = pca3.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var[2]:.1%})")
    ax.set_title(f"3-D PCA — {algo_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()

    safe = algo_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"cluster_3d_pca_{safe}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    _log(f"Saved -> {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════
#  SECTION 8 — Output Serialisation
# ══════════════════════════════════════════════════════════════

def build_cluster_output(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    archetype_map: Dict[int, str],
    metrics: Dict[str, Any],
    algo_name: str,
) -> Dict[str, Any]:
    """
    Build a serialisable dict that maps every student_id to their
    cluster id and archetype label, plus aggregate stats.
    """
    student_assignments = {}
    for sid, cid in zip(feature_df.index, labels):
        student_assignments[str(sid)] = {
            "cluster_id": int(cid),
            "archetype":  archetype_map.get(int(cid), "Unknown"),
        }

    cluster_summaries = {}
    tmp = feature_df.copy()
    tmp["cluster"] = labels
    for cid, grp in tmp.groupby("cluster"):
        cluster_summaries[str(int(cid))] = {
            "archetype":   archetype_map.get(int(cid), "Unknown"),
            "n_students":  int(len(grp)),
            "mean_unpleasant_ratio": round(float(grp["unpleasant_ratio"].mean()), 3),
            "mean_avg_intensity":  round(float(grp["avg_intensity"].mean()),  3),
            "mean_avg_tiredness":  round(float(grp["avg_tiredness"].mean()),  3),
            "mean_absence_rate":   round(float(grp["absence_rate"].mean()),   3),
        }

    return {
        "algorithm":         algo_name,
        "metrics":           metrics,
        "cluster_summaries": cluster_summaries,
        "student_assignments": student_assignments,
    }


# ══════════════════════════════════════════════════════════════
#  SECTION 8b — Multi-K Visualisations (all K values explored)
# ══════════════════════════════════════════════════════════════

def run_all_k_visualizations(
    X_scaled: np.ndarray,
    feature_df: pd.DataFrame,
    X_pca: np.ndarray,
    pca: PCA,
    k_range: range = range(2, 10),
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Run K-Means for every K in k_range, save a PCA scatter PNG per K,
    and return cluster composition data for each K so the dashboard
    can let teachers explore K=2, K=3, K=4 … interactively.

    Saves: cluster_kmeans_k{K}_pca.png for each K.
    Returns: { "2": {silhouette, cluster_summaries, pca_image}, "3": {...}, ... }
    """
    _log(f"Generating per-K visualisations (K={k_range.start}–{k_range.stop - 1}) ...")
    all_k_data: Dict[str, Any] = {}

    for k in k_range:
        labels = run_kmeans(X_scaled, k)
        arch   = assign_archetypes(feature_df, labels)
        sil    = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

        # PCA scatter for this specific K
        fig, ax = plt.subplots(figsize=(9, 6))
        _scatter_pca(ax, X_pca, labels, f"K-Means — K={k}  (silhouette={sil:.4f})", arch, pca)
        img_name = f"cluster_kmeans_k{k}_pca.png"
        img_path = os.path.join(output_dir, img_name)
        plt.savefig(img_path, dpi=130, bbox_inches="tight")
        plt.close()
        _log(f"  K={k}  silhouette={sil:.4f}  → {img_name}", step="MultiK")

        # Cluster composition summary
        counts = Counter(labels.tolist())
        cluster_summaries = {}
        for cid in sorted(counts.keys()):
            grp = feature_df[labels == cid]
            cluster_summaries[str(cid)] = {
                "archetype":            arch.get(int(cid), f"Cluster {cid}"),
                "n_students":           int(counts[cid]),
                "mean_unpleasant_ratio": round(float(grp["unpleasant_ratio"].mean()), 3),
                "mean_avg_intensity":   round(float(grp["avg_intensity"].mean()), 3),
                "mean_avg_tiredness":   round(float(grp["avg_tiredness"].mean()), 3),
            }

        all_k_data[str(k)] = {
            "k":                int(k),
            "silhouette":       round(sil, 4),
            "cluster_summaries": cluster_summaries,
            "pca_image":        img_name,
        }

    _log(f"Multi-K visualisations complete — {len(all_k_data)} K values saved.")
    return all_k_data


# ══════════════════════════════════════════════════════════════
#  SECTION 9 — Main Entry Point
# ══════════════════════════════════════════════════════════════

def cluster_students(
    student_profiles: Dict[str, Dict],
    output_dir: str = OUTPUT_DIR,
    best_algo: str = "K-Means",
) -> Dict[str, Any]:
    """
    Full clustering pipeline:
      1. Build feature matrix
      2. Find optimal K
      3. Run 4 algorithms
      4. Evaluate + compare
      5. Generate all visualisations
      6. Return serialisable results for JSON export

    Parameters
    ----------
    student_profiles : output of build_all_student_profiles()
    output_dir       : directory to save plots and JSON
    best_algo        : which algorithm's assignments to use as the
                       canonical cluster column in downstream modules

    Returns
    -------
    dict with keys: best_algorithm, student_assignments, cluster_summaries,
                    metrics_comparison, all_results
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print("\n" + "="*60)
    print("  STUDENT CLUSTERING — Switch4Schools")
    print(f"  Output folder: {output_dir}")
    print("="*60)

    # 1 ── Feature matrix ──────────────────────────────────────
    _log(f"Step 1/7 — Building feature matrix from {len(student_profiles):,} student profiles ...")
    feature_df, X_scaled, scaler = build_feature_matrix(student_profiles)

    # 2 ── Optimal K ───────────────────────────────────────────
    _log("Step 2/7 — Finding optimal number of clusters (K) ...")
    best_k = find_optimal_k(X_scaled, k_range=range(2, 10), output_dir=output_dir)

    # 3 ── PCA (shared) ────────────────────────────────────────
    _log("Step 3/7 — Computing PCA 2D/3D projections for visualisation ...")
    X_pca, pca = compute_pca(X_scaled)

    # 3b ── Multi-K visualisations (all K=2..9 for dashboard exploration) ──
    _log("Step 3b — Generating per-K PCA scatter plots for all K values ...")
    all_k_results = run_all_k_visualizations(
        X_scaled, feature_df, X_pca, pca,
        k_range=range(2, 10), output_dir=output_dir,
    )

    # 4 ── Run algorithms ──────────────────────────────────────
    _log("Step 4/7 — Running 4 clustering algorithms ...")
    all_results: Dict[str, Dict] = {}

    for algo_name, labels in [
        ("K-Means",          run_kmeans(X_scaled, best_k)),
        ("Agglomerative",    run_agglomerative(X_scaled, best_k)),
        ("GMM",              run_gmm(X_scaled, best_k)),
        ("DBSCAN",           run_dbscan(X_scaled)),
    ]:
        arch    = assign_archetypes(feature_df, labels)
        metrics = evaluate_labels(X_scaled, labels, algo_name)
        all_results[algo_name] = {
            "labels":    labels,
            "archetypes": arch,
            "metrics":   metrics,
        }
        _log(f"  {algo_name:<18} -> {metrics['n_clusters']} clusters  "
             f"silhouette={metrics['silhouette']}", step="Eval")

    # 5 ── Visualisations ──────────────────────────────────────
    _log("Step 5/7 — Generating visualisations (this takes ~30s) ...")

    # 5a. All algorithms PCA side by side
    _log("  Plotting all-algorithm PCA grid ...")
    plot_all_algorithms(X_pca, pca, all_results, output_dir)

    # 5b. Per-algorithm: silhouette, profiles, emotion breakdown, 3D
    for i, (algo_name, res) in enumerate(all_results.items(), 1):
        _log(f"  [{i}/{len(all_results)}] {algo_name} — silhouette / radar+heatmap / emotion / 3D ...")
        plot_silhouette_samples(X_scaled, res["labels"], algo_name,
                                res["archetypes"], output_dir)
        plot_cluster_profiles(feature_df, res["labels"], algo_name,
                              res["archetypes"], output_dir)
        plot_emotion_breakdown_per_cluster(feature_df, student_profiles,
                                           res["labels"], algo_name,
                                           res["archetypes"], output_dir)
        plot_3d_pca(X_scaled, res["labels"], algo_name,
                    res["archetypes"], output_dir)

    # 5c. Size distribution & algorithm comparison
    _log("  Plotting cluster size distribution ...")
    plot_cluster_size_distribution(all_results, output_dir)
    all_metrics = [r["metrics"] for r in all_results.values()]
    _log("  Plotting algorithm comparison chart ...")
    plot_algorithm_comparison(all_metrics, output_dir)

    # 6 ── Build JSON output ───────────────────────────────────
    _log("Step 6/7 — Serialising results to JSON ...")
    if best_algo not in all_results:
        best_algo = "K-Means"
    best = all_results[best_algo]
    result = build_cluster_output(
        feature_df,
        best["labels"],
        best["archetypes"],
        best["metrics"],
        best_algo,
    )
    result["metrics_comparison"] = all_metrics
    result["best_k"]             = best_k
    result["all_k_results"]      = all_k_results   # per-K data for dashboard K explorer

    out_path = os.path.join(output_dir, "student_clusters.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    _log(f"JSON saved -> {out_path}")

    # 7 ── Summary table ───────────────────────────────────────
    _log("Step 7/7 — Done. Summary:")
    print("\n" + "─"*60)
    print(f"  {'Algorithm':<20} {'K':>3}  {'Silhouette':>10}  {'C-H Index':>12}  {'D-B Score':>10}")
    print("─"*60)
    for m in all_metrics:
        sil_str = f"{m['silhouette']:.4f}" if m["silhouette"] is not None else "  N/A  "
        ch_str  = f"{m['calinski_harabasz']:.1f}" if m["calinski_harabasz"] is not None else "  N/A  "
        db_str  = f"{m['davies_bouldin']:.4f}" if m["davies_bouldin"] is not None else "  N/A  "
        print(f"  {m['algorithm']:<20} {m['n_clusters']:>3}  {sil_str:>10}  {ch_str:>12}  {db_str:>10}")
    print("─"*60)
    elapsed = time.time() - t0
    print(f"\n  Best algorithm : {best_algo}")
    print(f"  Output folder  : {output_dir}")
    print(f"  Total time     : {elapsed:.1f}s")
    print("="*60 + "\n")

    return result


# ══════════════════════════════════════════════════════════════
#  SECTION 10 — Standalone Runner
#  Run directly:  python src/profiling/student_clusterer.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, PROJECT_ROOT)

    print("\n" + "="*60)
    print("  Running student_clusterer.py as standalone script")
    print("="*60)

    _log("Loading data pipeline — this may take a minute ...")
    try:
        from src.data.loader import load_all
        from src.data.preprocessor import clean_checkins, merge_datasets
        from src.profiling.student_profiler import build_all_student_profiles

        datasets   = load_all()
        clean_df   = clean_checkins(datasets["student_checkins"])
        merged_df  = merge_datasets(
            clean_df,
            datasets["classes"],
            datasets["sessions"],
            datasets["students"],
        )
        _log(f"Merged dataset loaded: {len(merged_df):,} rows")

        student_profiles = build_all_student_profiles(merged_df)
        _log(f"Student profiles built: {len(student_profiles):,} students")

    except Exception as e:
        print(f"\n[ERROR] Could not load data: {e}")
        print("Make sure you run this from the project root:")
        print("  python src/profiling/student_clusterer.py")
        sys.exit(1)

    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    result = cluster_students(student_profiles, output_dir=output_dir)

    print(f"\nDone! Check {output_dir}/ for all charts and student_clusters.json")
