#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMS cluster profiles: sizes, confidence, silhouette, and feature fingerprints.

Inputs
------
--features     outputs/features/features_cases_scaled.csv   (index=sample_id; numeric)
--assignments  outputs/clustering_k13/assignments_cases.csv (from refit_k13_gmm.py)
--latent       outputs/clustering_k13/latent_pca.csv        (from refit_k13_gmm.py) [optional but recommended]
--out          outputs/cluster_profiles

What you get
------------
tables/
  - cluster_sizes.csv                 (# per GMS, median post_max, % low_conf)
  - per_cluster_feature_means.csv     (means of interpretable features per GMS)
  - top_features_per_cluster.csv      (top 10 ↑ and ↓ z-diffs per GMS)
  - per_sample_silhouette.csv         (if latent provided)
  - per_cluster_silhouette.csv        (if latent provided)
  - sample_manifest.csv               (sample_id, GMS, post_max, low_confidence)

figs/
  - cluster_sizes.png                 (bar)
  - posterior_by_cluster.png          (boxplot of post_max)
  - feature_heatmap.png               (GMS × key features, z-scored)
  - spectrum_6ch_by_cluster.png       (stacked bars if 6-channel features present)
  - silhouette_bars.png               (if latent provided)

Dependencies: pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_samples, silhouette_score

# ---------------- utils ----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_indexed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df

def pick_present(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]

def zscore_colwise(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0) + 1e-12)

def palette(n: int) -> List[str]:
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7'])
    return [base[i % len(base)] for i in range(n)]

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="GMS cluster profiles")
    ap.add_argument("--features", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--latent", default=None, help="optional latent CSV for silhouette (index=sample_id)")
    ap.add_argument("--gms-col", default="GMS_label")
    ap.add_argument("--out", default="outputs/cluster_profiles")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir); ensure_dir(outdir/"figs"); ensure_dir(outdir/"tables")

    # Load
    X = read_indexed_csv(args.features)
    A = read_indexed_csv(args.assignments)
    if args.gms_col not in A.columns:
        raise SystemExit(f"'{args.gms_col}' not found in assignments. Columns: {A.columns.tolist()}")
    # align
    idx = sorted(set(X.index) & set(A.index))
    if not idx:
        raise SystemExit("No overlapping IDs between features and assignments.")
    X = X.loc[idx]
    A = A.loc[idx]

    # Pull basics
    gms = A[args.gms_col].astype(str)
    post = A["post_max"] if "post_max" in A.columns else pd.Series(1.0, index=A.index)
    lowc = A["low_confidence"] if "low_confidence" in A.columns else (post < 0.70).astype(int)

    # Cluster sizes & confidence
    sizes = gms.value_counts().sort_index()
    med_post = post.groupby(gms).median()
    pct_low = (lowc.groupby(gms).mean() * 100.0)
    summary = pd.DataFrame({"n": sizes, "median_post": med_post, "pct_low_conf": pct_low}).fillna(0.0)
    summary.index.name = "GMS"
    summary.to_csv(outdir/"tables/cluster_sizes.csv")

    # Sample manifest
    manifest = pd.DataFrame({"sample_id": A.index, "GMS": gms.values, "post_max": post.values, "low_confidence": lowc.values})
    manifest.to_csv(outdir/"tables/sample_manifest.csv", index=False)

    # Interpretable features (pick those present)
    key_feats = [
        "frac_frameshift","frac_nonsense","frac_synonymous","frac_missense",
        "frac_indel","frac_long_indel_ge5bp","frac_loftee_hc",
        "frac_vep_HIGH","frac_vep_MODERATE","frac_vep_LOW",
        "CADD_median","CADD_75th","REVEL_median","PolyPhen_mean","SIFT_median",
        "AF_min","AF_median","AF_max",
        "SpliceAI_max","SpliceAI_ge_0_2_count","dbscSNV_ada_max","dbscSNV_rf_max",
        "clinvar_plp_count",
        "prop_C>A","prop_C>G","prop_C>T","prop_T>A","prop_T>C","prop_T>G"
    ]
    present = pick_present(X, key_feats)
    F = X[present].copy()

    # Per-cluster means
    means = F.groupby(gms).mean(numeric_only=True)
    means.index.name = "GMS"
    means.to_csv(outdir/"tables/per_cluster_feature_means.csv")

    # Top features per cluster (z-diff from global)
    Z = zscore_colwise(F)  # per-feature z across cohort
    Zm = Z.groupby(gms).mean()
    rows = []
    for label in Zm.index:
        v = Zm.loc[label]
        top_pos = v.sort_values(ascending=False).head(10)
        top_neg = v.sort_values(ascending=True).head(10)
        for feat, val in top_pos.items():
            rows.append({"GMS": label, "feature": feat, "z_diff": float(val), "direction": "up"})
        for feat, val in top_neg.items():
            rows.append({"GMS": label, "feature": feat, "z_diff": float(val), "direction": "down"})
    pd.DataFrame(rows).to_csv(outdir/"tables/top_features_per_cluster.csv", index=False)

    # -------- Figures --------
    # 1) sizes
    fig = plt.figure(figsize=(max(7, 0.4*len(summary)+4), 4), dpi=300)
    plt.bar(summary.index.astype(str), summary["n"].values)
    plt.title("GMS sizes"); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(outdir/"figs/cluster_sizes.png"); plt.close(fig)

    # 2) posterior by cluster
    fig = plt.figure(figsize=(max(7, 0.4*len(summary)+4), 4), dpi=300)
    data = [post[gms == g].values for g in summary.index]
    plt.boxplot(data, labels=summary.index.astype(str), showfliers=False)
    plt.ylabel("max posterior"); plt.title("Assignment confidence by GMS")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(outdir/"figs/posterior_by_cluster.png"); plt.close(fig)

    # 3) feature heatmap (z-scored means)
    H = Zm.copy()
    # reorder columns: put burdens/impact first, then deleteriousness, AF, splice, 6-channel
    order = pick_present(H, ["frac_frameshift","frac_nonsense","frac_synonymous","frac_missense","frac_indel","frac_long_indel_ge5bp","frac_loftee_hc",
                             "frac_vep_HIGH","frac_vep_MODERATE","frac_vep_LOW",
                             "CADD_median","CADD_75th","REVEL_median","PolyPhen_mean","SIFT_median",
                             "AF_min","AF_median","AF_max",
                             "SpliceAI_max","SpliceAI_ge_0_2_count","dbscSNV_ada_max","dbscSNV_rf_max",
                             "clinvar_plp_count",
                             "prop_C>A","prop_C>G","prop_C>T","prop_T>A","prop_T>C","prop_T>G"])
    H = H[order]
    fig = plt.figure(figsize=(max(10, 0.35*H.shape[1]+4), max(6, 0.3*H.shape[0]+3)), dpi=300)
    im = plt.imshow(H.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="mean z (feature)")
    plt.yticks(range(H.shape[0]), H.index.astype(str))
    plt.xticks(range(H.shape[1]), H.columns, rotation=70, ha="right", fontsize=8)
    plt.title("Feature fingerprints per GMS (z-scored)")
    plt.tight_layout(); plt.savefig(outdir/"figs/feature_heatmap.png"); plt.close(fig)

    # 4) 6-channel spectrum by cluster (if present)
    six_cols = pick_present(means, ["prop_C>A","prop_C>G","prop_C>T","prop_T>A","prop_T>C","prop_T>G"])
    if six_cols:
        fig = plt.figure(figsize=(max(10, 0.5*means.shape[0]+4), 5), dpi=300)
        bottom = np.zeros(means.shape[0])
        for c in six_cols:
            vals = means[c].values
            plt.bar(means.index.astype(str), vals, bottom=bottom, label=c)
            bottom += vals
        plt.legend(title="6-channel", fontsize=8, ncol=3)
        plt.ylabel("Proportion"); plt.title("6-channel substitution spectrum by GMS")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(outdir/"figs/spectrum_6ch_by_cluster.png"); plt.close(fig)

    # 5) Silhouette (optional)
    if args.latent and Path(args.latent).exists():
        Zlatent = read_indexed_csv(args.latent).reindex(A.index).select_dtypes(include=[np.number])
        if not Zlatent.empty and len(Zlatent.columns) >= 2:
            labels, lab_names = pd.factorize(gms)
            s_all = silhouette_samples(Zlatent.values, labels, metric="euclidean")
            sil_overall = silhouette_score(Zlatent.values, labels, metric="euclidean")
            per_sample = pd.DataFrame({"sample_id": A.index, "GMS": gms.values, "silhouette": s_all})
            per_sample.to_csv(outdir/"tables/per_sample_silhouette.csv", index=False)
            per_cluster = per_sample.groupby("GMS")["silhouette"].mean().reindex(summary.index)
            per_cluster.to_csv(outdir/"tables/per_cluster_silhouette.csv", header=["mean_silhouette"])
            fig = plt.figure(figsize=(max(7, 0.4*len(per_cluster)+4), 4), dpi=300)
            plt.bar(per_cluster.index.astype(str), per_cluster.values)
            plt.ylim(0, 1); plt.ylabel("mean silhouette")
            plt.title(f"Silhouette by GMS (overall={sil_overall:.2f})")
            plt.xticks(rotation=45, ha="right"); plt.tight_layout()
            plt.savefig(outdir/"figs/silhouette_bars.png"); plt.close(fig)

    print("\n✅ Cluster profiles complete.")
    print(f"   Tables → {outdir/'tables'}")
    print(f"   Figs   → {outdir/'figs'}")

if __name__ == "__main__":
    main()
