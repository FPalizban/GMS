#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gms_pathway_enrichment.py
Auto-maps sample IDs from filenames (no flags needed) and runs pathway enrichment.

Usage:
  python scripts/gms_pathway_enrichment.py \
    --assignments outputs/clustering_k13/assignments_cases.csv \
    --variants-glob "data/cohorts/cases/*.txt" \
    --genesets resources/pathways/reactome.gmt resources/pathways/kegg.gmt \
    --out outputs/pathways_k13

Outputs:
  outputs/pathways_k13/
    tables/
      gene_universe.csv
      per_gms_gene_lists.csv
      gms_pathway_enrichment.csv
    figs/
      pathway_heatmap_top20.png
      top10_<GMS>.png
"""

from __future__ import annotations
import argparse
import gzip
import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def open_any(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r", errors="ignore")

def sniff_sep(sample: str) -> str:
    return "," if sample.count(",") > sample.count("\t") else "\t"

def read_table_auto(path: str) -> pd.DataFrame:
    with open_any(path) as fh:
        head = fh.read(1024)
    sep = sniff_sep(head)
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)

def read_assignments(path: str, gms_col: str = "GMS_label") -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    if gms_col not in df.columns:
        raise SystemExit(f"[assignments] '{gms_col}' not in columns: {df.columns.tolist()}")
    s = df[gms_col].astype(str)
    s.index = s.index.astype(str).str.strip()
    return s

# ---------------- gene detection ----------------

CANDIDATE_GENE_COLS = [
    "Gene_symbol", "CG_Gene", "_fkKV_geneID", "CG_Label_Gene",
    "CV_symbol", "CV_KV_symbol",
    "Hugo_Symbol", "Gene", "Gene_name", "SYMBOL", "symbol",
    "GENE", "gene", "Gene.refGene"
]

def guess_gene_col(cols: List[str], prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in cols:
        return prefer
    for c in CANDIDATE_GENE_COLS:
        if c in cols:
            return c
    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in cols}
    for cand in CANDIDATE_GENE_COLS:
        k = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if k in norm:
            return norm[k]
    return None

# ---------------- candidate variant rule ----------------

def _get_float(row: pd.Series, *names, default=np.nan) -> float:
    for n in names:
        if n in row and pd.notna(row[n]):
            try:
                return float(row[n])
            except Exception:
                pass
    return default

def _get_str(row: pd.Series, *names) -> str:
    for n in names:
        if n in row and pd.notna(row[n]):
            return str(row[n])
    return ""

def is_candidate_variant(row: pd.Series) -> bool:
    """
    Keep variant if ANY:
      • GDCross_score > 0
      • (IMPACT in {HIGH, MODERATE}) AND (CADD >= 20 OR REVEL >= 0.5 OR LOFTEE contains 'HC')
      • ClinVar contains 'pathogenic' / 'likely_pathogenic' (case-insensitive)
    """
    gd = _get_float(row, "GDCross_score", "KV_GDCross", "gd_cross")
    if pd.notna(gd) and gd > 0:
        return True
    impact = _get_str(row, "VEP_IMPACT", "IMPACT", "vep_impact").upper()
    cadd = _get_float(row, "CADD", "CADD_phred", "CADD_PHRED", "cadd_phred")
    revel = _get_float(row, "REVEL", "KV_REVEL", "revel")
    loftee = _get_str(row, "LOFTEE", "loftee", "LoF").upper()
    clin = _get_str(row, "CLNSIG", "ClinVar_Significance", "ClinVar_PLP_positional", "clinvar_significance").lower()
    if impact in {"HIGH", "MODERATE"} and (
        (pd.notna(cadd) and cadd >= 20) or
        (pd.notna(revel) and revel >= 0.5) or
        ("HC" in loftee)
    ):
        return True
    if "pathog" in clin:
        return True
    return False

# ---------------- hypergeometric + BH-FDR ----------------

try:
    from scipy.stats import hypergeom
    def hg_sf(k: int, K: int, N: int, n: int) -> float:
        return float(hypergeom.sf(k - 1, N, K, n))
except Exception:
    from math import comb
    def hg_sf(k: int, K: int, N: int, n: int) -> float:
        top = 0.0
        xmax = int(min(n, K))
        for x in range(int(k), xmax + 1):
            top += comb(K, x) * comb(N - K, n - x)
        den = comb(N, n) if N >= n else 1
        return float(top / max(den, 1))

def bh_fdr_series(pvals: pd.Series) -> pd.Series:
    p = pvals.astype(float).values
    m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan, dtype=float)
    rank = 0
    for idx in order:
        if not np.isfinite(p[idx]): continue
        rank += 1
        q[idx] = p[idx] * m / rank
    # monotone decreasing
    min_so_far = np.inf
    for idx in order[::-1]:
        if not np.isfinite(q[idx]): continue
        min_so_far = min(min_so_far, q[idx])
        q[idx] = min(min_so_far, 1.0)
    return pd.Series(q, index=pvals.index)

# ---------------- automatic sample_id derivation ----------------

INFILE_ID_CANDIDATES = [
    "Sample_ID","sample_id","Subject_ID","subject_id","Tumor_Sample_Barcode",
    "SAMPLE","Sample","ID","id"
]

def derive_sample_id_auto(file_path: str, df: pd.DataFrame, assignment_ids: set) -> str:
    """
    Best-effort automatic derivation:
      1) exact stem in assignments
      2) any token (split on non-alnum) in assignments
      3) any assignment id that is a substring of the stem (unique)
      4) any in-file ID column value matching assignments (unique)
      5) fallback: full stem (will be dropped if not in assignments)
    """
    stem = Path(file_path).stem
    stem_clean = stem.strip()
    # 1) exact
    if stem_clean in assignment_ids:
        return stem_clean
    # 2) token match
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", stem_clean) if t]
    for t in tokens:
        if t in assignment_ids:
            return t
    # 3) substring match (unique)
    subs = [aid for aid in assignment_ids if aid in stem_clean]
    if len(subs) == 1:
        return subs[0]
    # prefer tokens starting with common prefixes (KF, USACHP, X…)
    pref = [t for t in tokens if t in assignment_ids]
    if pref:
        return pref[0]
    # 4) in-file columns
    for col in INFILE_ID_CANDIDATES:
        if col in df.columns:
            vals = [str(v).strip() for v in df[col].dropna().unique().tolist() if str(v).strip() != ""]
            hits = [v for v in vals if v in assignment_ids]
            if len(hits) == 1:
                return hits[0]
    # 5) fallback
    return stem_clean

# ---------------- gene set loading ----------------

def open_gmt(path: str):
    with open_any(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                yield parts[0], parts[1], parts[2:]

def load_genesets(paths: List[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if p.endswith(".gmt") or p.endswith(".gmt.gz"):
            src = Path(p).stem.split(".")[0]
            for name, desc, genes in open_gmt(p):
                for g in genes:
                    g = str(g).strip()
                    if g:
                        rows.append((f"{src}:{name}", name, src, g))
        else:
            df = read_table_auto(p)
            lc = {c.lower(): c for c in df.columns}
            s_col = lc.get("set", lc.get("pathway", list(df.columns)[0]))
            g_col = lc.get("gene", list(df.columns)[-1])
            src_col = lc.get("source", None)
            for _, r in df.iterrows():
                set_name = str(r[s_col])
                gene = str(r[g_col])
                source = str(r[src_col]) if src_col else Path(p).stem
                rows.append((f"{source}:{set_name}", set_name, source, gene))
    GS = pd.DataFrame(rows, columns=["set_id", "set_name", "source", "gene"]).dropna()
    GS["gene"] = GS["gene"].astype(str).str.strip()
    GS = GS[GS["gene"] != ""].drop_duplicates()
    return GS

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Pathway enrichment per GMS with automatic sample_id mapping")
    ap.add_argument("--assignments", required=True, help="CSV with index=sample_id and GMS label column")
    ap.add_argument("--variants-glob", required=True, help="Glob to variant TXT/CSV/TSV files")
    ap.add_argument("--genesets", nargs="+", required=True, help="One or more GMT/TSV/CSV gene set files")
    ap.add_argument("--gms-col", default="GMS_label", help="Column name in assignments containing labels")
    ap.add_argument("--gene-col", default=None, help="Force a specific gene column name (optional)")
    ap.add_argument("--min-set-size", type=int, default=10)
    ap.add_argument("--max-set-size", type=int, default=2000)
    ap.add_argument("--out", default="outputs/pathways_k13")
    args = ap.parse_args()

    outdir = Path(args.out)
    ensure_dir(outdir); ensure_dir(outdir / "tables"); ensure_dir(outdir / "figs")

    # Assignments
    labels = read_assignments(args.assignments, gms_col=args.gms_col)  # Series
    assign_ids = set(labels.index.astype(str))
    print(f"[IO] Assignments: {len(labels)} samples, {labels.nunique()} unique labels")

    # Variant files
    var_files = sorted(glob(args.variants_glob))
    print(f"[SCAN] Matched {len(var_files)} variant files")
    if not var_files:
        print("[WARN] No files matched your --variants-glob pattern.")
        return

    # Build per-sample gene lists with auto-ID mapping
    sample2genes: Dict[str, List[str]] = {}
    no_gene_col = 0
    derived_pairs = []

    for p in var_files:
        df = read_table_auto(p)
        gcol = guess_gene_col(df.columns.tolist(), args.gene_col)
        if not gcol:
            print(f"[WARN] {os.path.basename(p)}: no gene column among {CANDIDATE_GENE_COLS}")
            no_gene_col += 1
            continue

        sid = derive_sample_id_auto(p, df, assign_ids)
        derived_pairs.append((Path(p).name, sid, sid in assign_ids))

        keep_genes = []
        for _, r in df.iterrows():
            try:
                if is_candidate_variant(r):
                    g = str(r[gcol]).strip()
                    if g and g != "nan":
                        keep_genes.append(g)
            except Exception:
                continue
        sample2genes[sid] = sorted(set(keep_genes))

    if no_gene_col:
        print(f"[WARN] Gene column not found in {no_gene_col} file(s). Consider --gene-col to override.")

    # Overlap
    have = sorted(set(sample2genes) & assign_ids)
    missing = [name for name, sid, ok in derived_pairs if not ok]
    print(f"[IO] Overlap after auto-mapping: {len(have)}/{len(sample2genes)} samples")

    if len(have) == 0:
        # Be graceful: show a short diagnostic and exit without raising
        print("[ERROR] Could not auto-map any sample IDs.")
        print("Examples (file → derived_id → in_assignments):")
        for name, sid, ok in derived_pairs[:15]:
            print(f"  {name} → {sid} → {'✓' if ok else '✗'}")
        print("Tip: Ensure your assignment index contains tokens like 'KF##########' or 'USACHP###########' that also appear in filenames.")
        return

    # Keep only overlapping samples
    labels = labels.loc[have]

    # Gene universe
    gene_universe = sorted(set().union(*[set(sample2genes[s]) for s in have]))
    pd.Series(gene_universe, name="gene").to_csv(outdir / "tables" / "gene_universe.csv", index=False)
    print(f"[UNIVERSE] {len(gene_universe)} unique genes")

    # Per-GMS aggregation
    per_gms = {}
    for gms in sorted(labels.unique(), key=lambda s: (len(s), s)):
        sids = [s for s in have if labels.loc[s] == gms]
        genes = sorted(set().union(*[set(sample2genes.get(s, [])) for s in sids]))
        per_gms[gms] = genes
    pd.DataFrame(
        [{"GMS": g, "n_genes": len(v), "n_samples": int((labels == g).sum()), "genes": ";".join(v)} for g, v in per_gms.items()]
    ).to_csv(outdir / "tables" / "per_gms_gene_lists.csv", index=False)

    # Load gene sets & filter by size
    GS = load_genesets(args.genesets)
    set_sizes = GS.groupby("set_id")["gene"].nunique()
    keep_ids = set_sizes[(set_sizes >= args.min_set_size) & (set_sizes <= args.max_set_size)].index
    GS = GS[GS["set_id"].isin(keep_ids)].copy()
    print(f"[PATHWAYS] Loaded {GS['set_id'].nunique()} sets after size filter ({args.min_set_size}–{args.max_set_size}).")

    set_to_genes = {sid: set(g) for sid, g in GS.groupby("set_id")["gene"].agg(set).items()}
    set_meta = GS.drop_duplicates("set_id")[["set_id", "set_name", "source"]].set_index("set_id")

    # Enrichment
    N = len(gene_universe)
    U = set(gene_universe)
    rows = []
    for gms_label, ggenes in per_gms.items():
        Gset = set(ggenes) & U
        n = len(Gset)
        if n == 0:
            continue
        for sid, S in set_to_genes.items():
            K = len(S & U)
            if K == 0:
                continue
            k = len(S & Gset)
            if k == 0:
                continue
            pval = hg_sf(k, K, N, n)
            rows.append({
                "GMS": gms_label,
                "set_id": sid,
                "set_name": set_meta.loc[sid, "set_name"],
                "source": set_meta.loc[sid, "source"],
                "k_overlap": int(k),
                "n_in_GMS": int(n),
                "set_size": int(K),
                "universe": int(N),
                "pval": float(pval),
            })
    E = pd.DataFrame(rows)
    if E.empty:
        print("[WARN] No pathway overlaps detected after filtering; check candidate variant rule or gene sets.")
        # Still write empty table to avoid hard crash
        (outdir / "tables").mkdir(parents=True, exist_ok=True)
        E.to_csv(outdir / "tables" / "gms_pathway_enrichment.csv", index=False)
        return

    # BH-FDR within each source family
    E["FDR"] = np.nan
    for src in E["source"].unique():
        m = (E["source"] == src)
        E.loc[m, "FDR"] = bh_fdr_series(E.loc[m, "pval"])

    E = E.sort_values(["GMS", "FDR", "pval"], kind="mergesort")
    E.to_csv(outdir / "tables" / "gms_pathway_enrichment.csv", index=False)
    print(f"[OK] Wrote enrichment table → {outdir/'tables'/'gms_pathway_enrichment.csv'}")

    # Figures
    X = E.copy()
    X["score"] = -np.log10(X["FDR"].replace(0, np.nextafter(0, 1)))
    top_sets = (
        X.groupby("set_name")["score"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index
    )
    H = (
        X[X["set_name"].isin(top_sets)]
        .pivot_table(index="GMS", columns="set_name", values="score", aggfunc="max")
        .fillna(0.0)
    )
    H = H.reindex(sorted(H.index, key=lambda s: (len(s), s)))
    fig = plt.figure(figsize=(max(10, 0.4 * H.shape[1] + 4), max(6, 0.3 * H.shape[0] + 3)), dpi=300)
    im = plt.imshow(H.values, aspect="auto")
    plt.colorbar(im, label="-log10(FDR)")
    plt.yticks(range(H.shape[0]), H.index)
    plt.xticks(range(H.shape[1]), H.columns, rotation=60, ha="right", fontsize=8)
    plt.title("Pathway enrichment per GMS (top sets)")
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "pathway_heatmap_top20.png")
    plt.close(fig)
    print(f"[OK] Wrote heatmap → {outdir/'figs'/'pathway_heatmap_top20.png'}")

    for g in sorted(E["GMS"].unique(), key=lambda s: (len(s), s)):
        Eg = E[E["GMS"] == g].copy()
        Eg["score"] = -np.log10(Eg["FDR"].replace(0, np.nextafter(0, 1)))
        top = Eg.sort_values(["FDR", "pval"]).head(10)
        if top.empty:
            continue
        fig = plt.figure(figsize=(9, 5), dpi=300)
        plt.barh(top["set_name"], top["score"])
        plt.gca().invert_yaxis()
        plt.xlabel("-log10(FDR)")
        plt.title(f"Top pathways for {g}")
        plt.tight_layout()
        plt.savefig(outdir / "figs" / f"top10_{g}.png")
        plt.close(fig)

    print("\n✅ Pathway enrichment complete.")
    print(f"   Tables → {outdir/'tables'}")
    print(f"   Figs   → {outdir/'figs'}")

if __name__ == "__main__":
    main()
