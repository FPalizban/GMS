#!/usr/bin/env python3
"""
GMS rule induction: one-vs-rest thresholds (Youden) + depth-2 decision trees.

What this does
--------------
Given a per-sample feature matrix and final GMS assignments, this script:
  • Computes one-vs-rest AUROC for every feature per GMS
  • Finds the Youden-optimal cutoff (max TPR − FPR), with 95% bootstrap CIs
  • Records directionality (high_if_above vs high_if_below)
  • Fits a depth-2 decision tree surrogate (balanced) per GMS and prints readable rules
  • Saves tables + ROC curves + tree metrics for manuscript-ready reporting

Inputs
------
--features         CSV of per-sample features (rows=samples; index=sample_id).
                   Columns must be numeric/bool (categoricals should already be one-hot).
--assignments      CSV (index=sample_id) with a GMS label column (default: GMS_label).
--gms-col          Column name in assignments with the final GMS label.
--min-positive     Minimum # positives in a GMS to analyze (default: 15).
--bootstrap        B for bootstrap CIs on threshold & AUC (default: 200).
--zscore-in-script If set, z-score features inside this script (if not pre-scaled).
--topk-roc-plots   Plot ROC for top-K features per GMS (default: 12).
--out              Output directory (default: outputs/gms_rule_induction).
--out-suffix       Optional suffix to append to --out (e.g., outputs/gms_rule_induction_<suffix>).

Outputs (under --out or --out + '_' + --out-suffix)
---------------------------------------------------
- thresholds_per_signature.csv       # per GMS x feature with AUROC, cutoff, 95% CIs, dir, sens/spec
- tree_rules_per_signature.txt       # readable depth-2 rules per GMS
- top_features_per_signature.csv     # top features by AUROC per GMS
- roc_curves/GMS*/roc_<feature>.png  # ROC plots for top features
- tree_reports/GMS*/tree_report.json # confusion/metrics for each surrogate tree
- rule_induction_manifest.json

Notes
-----
* Assumes features are already z-scored (as in your Methods). If not, pass --zscore-in-script.
* Ignores non-numeric columns automatically; converts bool to {0,1}.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.tree import DecisionTreeClassifier


# -------------------- utils --------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # keep numeric/bool only
    num = df.select_dtypes(include=[np.number, bool]).copy()
    # cast bool→int
    for c in num.columns:
        if num[c].dtype == bool:
            num[c] = num[c].astype(int)
    # drop constant columns
    nunq = num.nunique(dropna=False)
    keep = nunq[nunq > 1].index.tolist()
    return num[keep]

def load_assignments(path: str, gms_col: str) -> pd.DataFrame:
    A = pd.read_csv(path, index_col=0)
    if gms_col not in A.columns:
        raise RuntimeError(f"'{gms_col}' not in assignments file: {path}")
    A.index = A.index.astype(str)
    A.index.name = "sample_id"
    return A[[gms_col]].rename(columns={gms_col: "GMS"})

def youden_cut(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Return best threshold by Youden J = TPR - FPR, along with sens/spec and AUC.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    j_best_idx = int(np.argmax(j))
    thr_best = float(thr[j_best_idx])
    sens = float(tpr[j_best_idx])
    spec = float(1 - fpr[j_best_idx])
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = float(auc(fpr, tpr))
    return thr_best, sens, spec, roc_auc, float(j[j_best_idx])

def bootstrap_ci(y: np.ndarray, s: np.ndarray, b: int = 200, rng: Optional[np.random.RandomState] = None) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Nonparametric bootstrap CIs for threshold and AUC (percentile).
    """
    if rng is None:
        rng = np.random.RandomState(1337)
    n = len(y)
    if n == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    idx = np.arange(n)
    thr_list: List[float] = []
    auc_list: List[float] = []
    for _ in range(b):
        bs = rng.choice(idx, size=n, replace=True)
        try:
            thr, _, _, roc_auc, _ = youden_cut(y[bs], s[bs])
            thr_list.append(thr)
            auc_list.append(roc_auc)
        except Exception:
            continue
    if len(thr_list) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    low_t, hi_t = np.percentile(thr_list, [2.5, 97.5])
    low_a, hi_a = np.percentile(auc_list, [2.5, 97.5])
    return (float(low_t), float(hi_t)), (float(low_a), float(hi_a))

def directionality(pos_vals: np.ndarray, neg_vals: np.ndarray) -> str:
    mpos = np.nanmean(pos_vals)
    mneg = np.nanmean(neg_vals)
    return "high_if_above" if mpos >= mneg else "high_if_below"

def plot_roc(y: np.ndarray, s: np.ndarray, title: str, out_png: Path) -> None:
    ensure_dir(out_png.parent)
    fpr, tpr, _ = roc_curve(y, s)
    try:
        roc_auc = roc_auc_score(y, s)
    except Exception:
        roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(fpr, tpr, lw=1.5)
    plt.plot([0, 1], [0, 1], ls="--", lw=1)
    plt.xlabel("1 − Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"{title}\nAUC={roc_auc:.3f}")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def tree_to_text(clf: DecisionTreeClassifier, feature_names: List[str]) -> List[str]:
    """
    Human-readable rules for a depth-2 tree.
    """
    from sklearn.tree import _tree
    t = clf.tree_
    rules: List[str] = []

    def recurse(node: int, path: str) -> None:
        if t.feature[node] != _tree.TREE_UNDEFINED:
            fname = feature_names[t.feature[node]]
            thr = t.threshold[node]
            recurse(t.children_left[node], path + f"{fname} <= {thr:.3f} AND ")
            recurse(t.children_right[node], path + f"{fname} > {thr:.3f} AND ")
        else:
            proba = clf.tree_.value[node][0]
            pred = int(np.argmax(proba))
            pos = float(proba[1]) if proba.size > 1 else float(proba[0])
            neg = float(proba[0]) if proba.size > 1 else 0.0
            rules.append(path[:-5] + f" ⇒ class={pred} (pos={pos:.1f}, neg={neg:.1f})")

    recurse(0, "")
    return rules


# -------------------- main --------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold induction + shallow rule surrogates for GMS")
    ap.add_argument("--features", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--gms-col", default="GMS_label")
    ap.add_argument("--min-positive", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--zscore-in-script", action="store_true", help="Z-score features here (if your input isn’t already scaled).")
    ap.add_argument("--topk-roc-plots", type=int, default=12, help="Plot ROC for top-K features per GMS.")
    ap.add_argument("--out", default="outputs/gms_rule_induction")
    ap.add_argument("--out-suffix", default=None, help="Optional suffix to append to --out (creates <out>_<suffix>).")
    args = ap.parse_args()

    outdir = Path(args.out if not args.out_suffix else f"{args.out}_{args.out_suffix}")
    ensure_dir(outdir)

    # ---- load data
    X = load_matrix(args.features)
    A = load_assignments(args.assignments, args.gms_col)

    # align on sample IDs
    common = X.index.astype(str).intersection(A.index.astype(str))
    if len(common) == 0:
        raise RuntimeError("No overlapping sample IDs between features and assignments.")
    X = X.loc[common].copy()
    y_all = A.loc[common, "GMS"].astype(str).values

    # optional z-score here
    if args.zscore_in_script:
        X = (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)
        X = X.fillna(0.0)

    feats = X.columns.tolist()
    def _gms_sort_key(s: str):
        u = str(s).upper()
        if u.startswith("GMS"):
            try:
                return (0, int(u.replace("GMS", "").strip()))
            except Exception:
                return (0, u)
        return (1, u)
    gms_labels = sorted(pd.Series(y_all).unique(), key=_gms_sort_key)

    rows: List[Dict] = []
    topfeat_rows: List[Dict] = []
    rules_txt_lines: List[str] = []
    rng = np.random.RandomState(2024)

    for gms in gms_labels:
        y = (pd.Series(y_all) == gms).astype(int).values
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        if n_pos < args.min_positive:
            rules_txt_lines.append(f"[{gms}] skipped (positives={n_pos} < {args.min_positive})")
            continue

        # Per-feature ROC/Youden
        auc_by_feat: List[Tuple[str, float]] = []
        for f in feats:
            s = X[f].values.astype(float)
            if np.isfinite(s).sum() < 5 or float(np.nanstd(s)) == 0.0:
                continue
            try:
                thr, sens, spec, aucv, jbest = youden_cut(y, s)
                (thr_lo, thr_hi), (auc_lo, auc_hi) = bootstrap_ci(y, s, b=args.bootstrap, rng=rng)
                dirn = directionality(s[y == 1], s[y == 0])
                rows.append({
                    "GMS": gms, "feature": f, "auc": aucv,
                    "youden_J": jbest, "threshold": thr, "thr_ci_low": thr_lo, "thr_ci_high": thr_hi,
                    "auc_ci_low": auc_lo, "auc_ci_high": auc_hi,
                    "sens_at_thr": sens, "spec_at_thr": spec, "direction": dirn,
                    "n_pos": n_pos, "n_neg": n_neg
                })
                auc_by_feat.append((f, aucv))
            except Exception:
                continue

        # top features for plotting / reporting
        auc_by_feat.sort(key=lambda t: t[1], reverse=True)
        topf = [f for f, _ in auc_by_feat[:max(1, args.topk_roc_plots)]]
        for f in topf:
            s = X[f].values.astype(float)
            plot_roc(y, s, title=f"{gms} vs rest — {f}", out_png=outdir / f"roc_curves/{gms}/roc_{f}.png")
        topfeat_rows.extend([{"GMS": gms, "feature": f, "rank": i + 1, "auc": a} for i, (f, a) in enumerate(auc_by_feat[:25])])

        # Surrogate tree (depth 2)
        clf = DecisionTreeClassifier(
            max_depth=2,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=max(5, int(0.01 * len(y)))
        )
        clf.fit(X.values, y)
        yhat = clf.predict(X.values)
        ypro = clf.predict_proba(X.values)[:, 1]
        acc = accuracy_score(y, yhat)
        try:
            auc_tree = roc_auc_score(y, ypro)
        except Exception:
            auc_tree = float("nan")
        # robust confusion matrix even if a class is missing in predictions
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)

        # record rules
        rules_txt_lines.append(f"\n[{gms}] depth-2 surrogate")
        for line in tree_to_text(clf, feats):
            line = line.replace("class=1", "class=IN_GMS").replace("class=0", "class=OTHER")
            rules_txt_lines.append("  - " + line)

        # metrics JSON per GMS
        ensure_dir(outdir / f"tree_reports/{gms}")
        report = {
            "GMS": gms,
            "n_pos": n_pos, "n_neg": n_neg,
            "accuracy": float(acc), "auc_tree": float(auc_tree),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)
        }
        with open(outdir / f"tree_reports/{gms}/tree_report.json", "w") as f:
            json.dump(report, f, indent=2)

    # Write tables and rules
    thr_df = pd.DataFrame(rows).sort_values(["GMS", "auc", "youden_J"], ascending=[True, False, False])
    top_df = pd.DataFrame(topfeat_rows).sort_values(["GMS", "rank"])
    ensure_dir(outdir)
    thr_df.to_csv(outdir / "thresholds_per_signature.csv", index=False)
    top_df.to_csv(outdir / "top_features_per_signature.csv", index=False)
    with open(outdir / "tree_rules_per_signature.txt", "w") as f:
        f.write("\n".join(rules_txt_lines) if rules_txt_lines else "No signatures met min-positive threshold.\n")

    # Manifest
    manifest = {
        "in": {
            "features": str(Path(args.features).resolve()),
            "assignments": str(Path(args.assignments).resolve()),
            "gms_col": args.gms_col
        },
        "params": {
            "min_positive": args.min_positive,
            "bootstrap": args.bootstrap,
            "zscore_in_script": bool(args.zscore_in_script),
            "topk_roc_plots": args.topk_roc_plots
        },
        "out": {
            "thresholds_csv": str(outdir / "thresholds_per_signature.csv"),
            "top_features_csv": str(outdir / "top_features_per_signature.csv"),
            "rules_txt": str(outdir / "tree_rules_per_signature.txt"),
            "roc_dir": str(outdir / "roc_curves"),
            "tree_reports_dir": str(outdir / "tree_reports")
        }
    }
    with open(outdir / "rule_induction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Rule induction complete.")
    print(f"   thresholds → {manifest['out']['thresholds_csv']}")
    print(f"   rules      → {manifest['out']['rules_txt']}")
    print(f"   top feats  → {manifest['out']['top_features_csv']}")
    print(f"   ROC figs   → {manifest['out']['roc_dir']}")
    print(f"   tree reps  → {manifest['out']['tree_reports_dir']}")


if __name__ == "__main__":
    main()
