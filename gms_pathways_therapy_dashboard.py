#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gms_pathways_therapy_dashboard.py

Per-GMS pathway/GO enrichment + therapy mapping + figures.

Inputs
------
--assignments         CSV with columns: sample_id, gms  (labels like GMS1..GMS13)
--variants-glob       Glob to per-sample variant tables (TSV/CSV/TXT); auto-detects sep
                      and a gene column among: Gene_symbol, Hugo_Symbol, Gene, Gene_name.
                      Used to build sample→genes and GMS→genes (union).
--reactome            One or more Reactome gene set files (comma-separated). Accepted formats:
                      (a) GMT (MSigDB-like): set<TAB>desc<TAB>gene1<TAB>gene2...
                      (b) CSV/TSV with columns: pathway, gene   (order agnostic, case-insensitive)
--kegg                Same as --reactome but for KEGG pathway sets.
--go-bp / --go-mf / --go-cc   GO Biological Process / Molecular Function / Cellular Component.
--drug-gene-csv       Optional drug–gene KB (columns: gene, drug[, evidence_tier, source])
--outdir              Output dir (default: outputs/gms_pathways_dashboard)
--topn-heatmap        #terms shown in heatmaps (default 15)
--topn-bars           #terms in per-GMS barplots (default 10)

Outputs
-------
outdir/
  tables/
    per_sample_therapy.csv
    per_gms_summary.csv
    enrichment_reactome.csv  (all terms for all GMS; BH-FDR)
    enrichment_kegg.csv
    enrichment_go_bp.csv (if provided)
    enrichment_go_mf.csv (if provided)
    enrichment_go_cc.csv (if provided)
  figs/
    heatmap_reactome_topN.png   (−log10 FDR across GMS)
    heatmap_kegg_topN.png
    heatmap_go_bp_topN.png (if provided)
    ...
    bars_reactome_GMS1.png (per-GMS topN)
    bars_kegg_GMS1.png
    bars_go_bp_GMS1.png
    ...

Requires: numpy, pandas, scipy, matplotlib.
"""

import argparse
import glob
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

# ----------------------------
# GMS → pathways/therapy/drugs (curated)
# ----------------------------
GMS_MAP: Dict[str, Dict[str, List[str]]] = {
    "GMS1": {
        "pathways": ["Homologous recombination repair", "Mismatch repair", "DSB repair"],
        "therapy_classes": ["PARP inhibition", "Immune checkpoint (MMRd/MSI-like)"],
        "exemplar_drugs": ["Olaparib", "Niraparib", "Rucaparib", "Pembrolizumab"],
    },
    "GMS2": {
        "pathways": ["TC-NER", "Oxidative damage response"],
        "therapy_classes": ["DNA-damaging chemo (platinum sensitivity in NER defects)"],
        "exemplar_drugs": ["Cisplatin", "Carboplatin"],
    },
    "GMS3": {
        "pathways": ["Background (no focal pathway)"],
        "therapy_classes": ["No signature-directed therapy (standard of care)"],
        "exemplar_drugs": [],
    },
    "GMS4": {
        "pathways": ["Replication stress", "Checkpoint signaling (ATR/CHK1)", "Polymerase fidelity"],
        "therapy_classes": ["ATR/CHK1/WEE1 inhibition", "PARP combinations (context)"],
        "exemplar_drugs": ["Ceralasertib", "Prexasertib", "Adavosertib", "Olaparib (combo)"],
    },
    "GMS5": {
        "pathways": ["Common/ancestry-linked background"],
        "therapy_classes": ["No signature-directed therapy (ancestry-aware interpretation)"],
        "exemplar_drugs": [],
    },
    "GMS6": {
        "pathways": ["Protein dysfunction in constrained genes", "PI3K/AKT", "MAPK/ERK"],
        "therapy_classes": ["Targeted therapy if driver identified (gene-specific)"],
        "exemplar_drugs": ["Trametinib", "Selumetinib", "Alpelisib"],
    },
    "GMS7": {
        "pathways": ["Cytidine deamination (APOBEC-like)"],
        "therapy_classes": ["No established germline-directed therapy"],
        "exemplar_drugs": [],
    },
    "GMS8": {
        "pathways": ["Hereditary cancer predisposition pathways (gene-specific)"],
        "therapy_classes": ["Gene-directed therapy (e.g., PARP for BRCA; MEK for NF1)", "Surveillance"],
        "exemplar_drugs": ["Olaparib (BRCA)", "Selumetinib (NF1)"],
    },
    "GMS9": {
        "pathways": ["Spontaneous deamination (CpG C>T)", "Base excision repair (context)"],
        "therapy_classes": ["No signature-directed therapy"],
        "exemplar_drugs": [],
    },
    "GMS10": {
        "pathways": ["Driver-like missense/hotspots (constraint)"],
        "therapy_classes": ["Targeted therapy contingent on gene/hotspot"],
        "exemplar_drugs": ["Larotrectinib (NTRK)", "Selpercatinib (RET)", "Dabrafenib (BRAF)"],
    },
    "GMS11": {
        "pathways": ["Structural instability", "End-joining & resection balance"],
        "therapy_classes": ["ATR/CHK1/WEE1 inhibition (context)", "PARP sensitization (context)"],
        "exemplar_drugs": ["Ceralasertib", "Prexasertib", "Adavosertib", "Olaparib (combo)"],
    },
    "GMS12": {
        "pathways": ["Splicing regulation", "Promoter/enhancer dysregulation"],
        "therapy_classes": ["Splice-modulating agents (investigational)", "Epigenetic therapy (context)"],
        "exemplar_drugs": ["H3B-8800 (inv.)", "Tazemetostat (context)"],
    },
    "GMS13": {
        "pathways": ["Localized hypermutation (kataegis-like)"],
        "therapy_classes": ["No established germline-directed therapy"],
        "exemplar_drugs": [],
    },
}

GENE_COL_CANDIDATES = ["Gene_symbol", "Hugo_Symbol", "Gene", "Gene_name"]

# ----------------------------
# Utilities
# ----------------------------
def detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        line = fh.readline()
    return "\t" if line.count("\t") >= line.count(",") else ","

def read_assignments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize
    rename = {}
    if "sample_id" not in df.columns:
        for c in df.columns:
            if c.lower() in ("sample", "id", "case_id", "subject_id"):
                rename[c] = "sample_id"
    if "gms" not in df.columns:
        for c in df.columns:
            if c.lower() in ("gms_label", "cluster", "label", "gmsid"):
                rename[c] = "gms"
    if rename:
        df = df.rename(columns=rename)
    if not {"sample_id", "gms"}.issubset(df.columns):
        raise ValueError("Assignments must have columns: sample_id, gms")
    df["sample_id"] = df["sample_id"].astype(str)
    df["gms"] = df["gms"].astype(str).str.upper()
    return df[["sample_id", "gms"]]

def find_gene_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c in GENE_COL_CANDIDATES:
            return c
    for c in df.columns:
        if c.lower() in [x.lower() for x in GENE_COL_CANDIDATES]:
            return c
    return None

def scan_variant_genes(variants_glob: Optional[str], sample_ids: List[str]) -> Dict[str, Set[str]]:
    """Map sample_id -> set(genes) by substring matching sample_id in filename (case-insensitive)."""
    hits = {sid: set() for sid in sample_ids}
    if not variants_glob:
        return hits
    files = sorted(glob.glob(variants_glob))
    if not files:
        print(f"[WARN] No files matched: {variants_glob}")
        return hits

    sample_ids_low = [(sid, sid.lower(), re.sub(r"[^a-z0-9]", "", sid.lower())) for sid in sample_ids]
    for p in files:
        base = os.path.basename(p)
        base_low = base.lower()
        base_alnum = re.sub(r"[^a-z0-9]", "", base_low)
        matched = []
        for sid, low, alnum in sample_ids_low:
            if (low in base_low) or (alnum and alnum in base_alnum):
                matched.append(sid)
        if not matched:
            continue

        sep = detect_sep(p)
        try:
            df = pd.read_csv(p, sep=sep, dtype=str, low_memory=False)
        except Exception:
            df = pd.read_csv(p, sep="\t" if sep == "," else ",", dtype=str, low_memory=False)
        gene_col = find_gene_column(df)
        if not gene_col:
            print(f"[WARN] {base}: no gene column among {GENE_COL_CANDIDATES}")
            continue
        genes = set(df[gene_col].dropna().astype(str).str.strip())
        genes = {g for g in genes if g and g not in ("-", "nan", "NaN")}
        for sid in matched:
            hits[sid].update(genes)

    n_have = sum(1 for sid in sample_ids if hits[sid])
    print(f"[GENES] Collected gene lists for {n_have}/{len(sample_ids)} samples")
    return hits

def load_drug_gene_kb(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["gene", "drug", "evidence_tier", "source"])
    sep = detect_sep(path)
    kb = pd.read_csv(path, sep=sep, dtype=str)
    rename = {}
    for c in kb.columns:
        cl = c.lower()
        if cl == "gene_symbol": rename[c] = "gene"
        if cl in ("therapy", "drug_name"): rename[c] = "drug"
    if rename: kb = kb.rename(columns=rename)
    cols = {c.lower(): c for c in kb.columns}
    for need in ("gene", "drug"):
        if need not in cols:
            raise ValueError(f"drug-gene CSV missing column: {need}")
    kb = kb.rename(columns={cols["gene"]:"gene", cols["drug"]:"drug"})
    for col in ("evidence_tier", "source"):
        if col not in kb.columns: kb[col] = ""
    kb["gene"] = kb["gene"].astype(str).str.strip()
    kb["drug"] = kb["drug"].astype(str).str.strip()
    kb = kb.dropna(subset=["gene","drug"]).drop_duplicates(subset=["gene","drug"])
    return kb[["gene","drug","evidence_tier","source"]]

# ---------- gene set loaders ----------
def parse_gmt(path: str) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:  # name, desc, genes...
                continue
            name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if genes:
                m[name] = set(genes)
    return m

def parse_pathway_csv(path: str) -> Dict[str, Set[str]]:
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str)
    cols_low = {c.lower(): c for c in df.columns}
    # try pathway/gene or term/gene or name/gene
    pname = None
    if "pathway" in cols_low: pname = cols_low["pathway"]
    elif "term" in cols_low: pname = cols_low["term"]
    elif "name" in cols_low: pname = cols_low["name"]
    gcol = None
    if "gene" in cols_low: gcol = cols_low["gene"]
    elif "gene_symbol" in cols_low: gcol = cols_low["gene_symbol"]
    if not pname or not gcol:
        raise ValueError(f"{os.path.basename(path)} must have columns (pathway/term/name) and (gene/gene_symbol)")
    df = df[[pname, gcol]].dropna()
    m: Dict[str, Set[str]] = {}
    for pw, sub in df.groupby(pname):
        m[str(pw)] = set(str(x).strip() for x in sub[gcol].tolist() if str(x).strip())
    return m

def load_genesets(arg: Optional[str], label: str) -> Dict[str, Set[str]]:
    """arg may be None, a file path, or 'file1,file2,...'. Mix of GMT and CSV/TSV is allowed."""
    res: Dict[str, Set[str]] = {}
    if not arg:
        print(f"[{label}] no files provided.")
        return res
    for p in [x.strip() for x in arg.split(",") if x.strip()]:
        if not os.path.exists(p):
            print(f"[{label}] file not found: {p}")
            continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".gmt":
                sub = parse_gmt(p)
            else:
                sub = parse_pathway_csv(p)
            res.update(sub)
        except Exception as e:
            print(f"[{label}] failed to load {p}: {e}")
    print(f"[{label}] loaded {len(res)} gene sets")
    return res

# ---------- enrichment ----------
def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    ranked[order] = p[order] * n / (np.arange(n) + 1)
    # monotone
    for i in range(n - 2, -1, -1):
        ranked[order[i]] = min(ranked[order[i]], ranked[order[i + 1]])
    return np.clip(ranked, 0, 1)

def enrich_one_set(gms_genes: Set[str],
                   background: Set[str],
                   genesets: Dict[str, Set[str]],
                   label: str,
                   gms: str) -> pd.DataFrame:
    """Hypergeometric enrichment: population = background genes."""
    N = len(background)
    n = len(gms_genes)
    if N == 0 or n == 0:
        return pd.DataFrame(columns=["gms","set_label","k","K","n","N","p","fdr","overlap_genes"])
    rows = []
    gms_genes_u = {g.upper() for g in gms_genes}
    bg_u = {g.upper() for g in background}
    for set_name, gene_set in genesets.items():
        gs = {g.upper() for g in gene_set} & bg_u
        K = len(gs)
        if K == 0:
            continue
        k = len(gs & gms_genes_u)
        if k == 0:
            continue
        # P(X >= k) under Hypergeom(N, K, n)
        p = hypergeom.sf(k - 1, N, K, n)
        rows.append((gms, set_name, k, K, n, N, p, ";".join(sorted(gs & gms_genes_u))))
    if not rows:
        return pd.DataFrame(columns=["gms","set_label","k","K","n","N","p","fdr","overlap_genes"])
    df = pd.DataFrame(rows, columns=["gms","set_label","k","K","n","N","p","overlap_genes"])
    df["fdr"] = bh_fdr(df["p"].values)
    df = df.sort_values(["fdr","p","k"], ascending=[True, True, False]).reset_index(drop=True)
    df.insert(1, "collection", label)
    return df

# ---------- plotting ----------
def heatmap_topn(df: pd.DataFrame, gms_order: List[str], title: str, outpng: str, topn: int = 15):
    """Draw a heatmap of −log10(FDR) for topn terms overall (union across GMS)."""
    if df.empty:
        print(f"[FIG] skip heatmap (no rows) for {title}")
        return
    df2 = df.copy()
    df2["minuslog10fdr"] = -np.log10(df2["fdr"].replace(0, 1e-300))
    # pick topN by best FDR per term across all GMS
    best = df2.groupby("set_label")["fdr"].min().sort_values().head(topn).index.tolist()
    sub = df2[df2["set_label"].isin(best)]
    # pivot GMS × set_label
    mat = sub.pivot_table(index="gms", columns="set_label", values="minuslog10fdr", aggfunc="max").fillna(0.0)
    mat = mat.reindex(index=gms_order, fill_value=0.0)
    plt.figure(figsize=(max(8, 0.5 * mat.shape[1]), 0.5 * len(gms_order) + 2))
    im = plt.imshow(mat.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="-log10(FDR)")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()
    print(f"[FIG] heatmap → {outpng}")

def bars_per_gms(df: pd.DataFrame, gms: str, title: str, outpng: str, topn: int = 10):
    if df.empty:
        return
    sub = df[df["gms"] == gms].copy()
    if sub.empty:
        return
    sub["minuslog10fdr"] = -np.log10(sub["fdr"].replace(0, 1e-300))
    sub = sub.sort_values(["fdr","k"], ascending=[True, False]).head(topn)
    plt.figure(figsize=(8, max(3, 0.35 * len(sub))))
    plt.barh(sub["set_label"][::-1], sub["minuslog10fdr"][::-1])
    plt.xlabel("-log10(FDR)")
    plt.title(f"{title} — {gms}")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()
    print(f"[FIG] bars → {outpng}")

# ---------- main ----------
def uniq_join(items: Iterable[str]) -> str:
    return "; ".join(sorted({x for x in items if x}))

def main():
    ap = argparse.ArgumentParser(description="Per-GMS pathway/GO enrichment + therapy mapping + figures")
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--variants-glob", required=True, help="Glob to per-sample variant tables (TSV/CSV/TXT)")
    ap.add_argument("--reactome", default=None, help="Reactome gene sets (CSV/TSV with pathway,gene OR GMT). Comma-separated allowed.")
    ap.add_argument("--kegg", default=None, help="KEGG gene sets (CSV/TSV with pathway,gene OR GMT). Comma-separated allowed.")
    ap.add_argument("--go-bp", default=None, help="GO BP gene sets (CSV/TSV pathway,gene OR GMT).")
    ap.add_argument("--go-mf", default=None, help="GO MF gene sets (CSV/TSV pathway,gene OR GMT).")
    ap.add_argument("--go-cc", default=None, help="GO CC gene sets (CSV/TSV pathway,gene OR GMT).")
    ap.add_argument("--drug-gene-csv", default=None, help="Optional KB with columns: gene,drug[,evidence_tier,source]")
    ap.add_argument("--outdir", default="outputs/gms_pathways_dashboard")
    ap.add_argument("--topn-heatmap", type=int, default=15)
    ap.add_argument("--topn-bars", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tbldir = os.path.join(args.outdir, "tables"); os.makedirs(tbldir, exist_ok=True)
    figdir = os.path.join(args.outdir, "figs"); os.makedirs(figdir, exist_ok=True)

    # Load assignments and genes
    A = read_assignments(args.assignments)
    gms_order = sorted(A["gms"].unique(), key=lambda x: int(re.sub(r"[^0-9]", "", x) or 0))
    sample2genes = scan_variant_genes(args.variants_glob, A["sample_id"].tolist())

    # Per-sample therapy table (curated, plus optional gene→drug matches)
    KB = load_drug_gene_kb(args.drug_gene_csv) if args.drug_gene_csv else pd.DataFrame(columns=["gene","drug","evidence_tier","source"])
    rows = []
    for _, r in A.iterrows():
        sid, gms = r["sample_id"], r["gms"]
        m = GMS_MAP.get(gms, {"pathways":[], "therapy_classes":[], "exemplar_drugs":[]})
        matched_genes, matched_drugs = [], []
        if not KB.empty:
            genes_u = {g.upper() for g in sample2genes.get(sid, set())}
            if genes_u:
                hit = KB[KB["gene"].str.upper().isin(genes_u)]
                if not hit.empty:
                    matched_genes = hit["gene"].tolist()
                    dd = []
                    for _, kr in hit.iterrows():
                        lab = kr["drug"]
                        if isinstance(kr.get("evidence_tier",""), str) and kr["evidence_tier"]:
                            lab = f"{lab} [{kr['evidence_tier']}]"
                        dd.append(lab)
                    matched_drugs = dd
        rows.append({
            "sample_id": sid,
            "gms": gms,
            "pathways_curated": uniq_join(m.get("pathways", [])),
            "therapy_classes": uniq_join(m.get("therapy_classes", [])),
            "exemplar_drugs": uniq_join(m.get("exemplar_drugs", [])),
            "matched_genes": uniq_join(matched_genes),
            "matched_drugs": uniq_join(matched_drugs),
        })
    per_sample = pd.DataFrame(rows)
    out_ps = os.path.join(tbldir, "per_sample_therapy.csv")
    per_sample.to_csv(out_ps, index=False)
    print(f"[OK] per-sample therapy → {out_ps}")

    # Per-GMS summary
    counts = A["gms"].value_counts().rename_axis("gms").reset_index(name="n_samples")
    mm = []
    for g in sorted(GMS_MAP.keys(), key=lambda x: int(x[3:])):
        mm.append({
            "gms": g,
            "pathways_curated": uniq_join(GMS_MAP[g]["pathways"]),
            "therapy_classes": uniq_join(GMS_MAP[g]["therapy_classes"]),
            "exemplar_drugs": uniq_join(GMS_MAP[g]["exemplar_drugs"]),
        })
    summary = pd.DataFrame(mm).merge(counts, on="gms", how="left").fillna({"n_samples":0})
    out_sum = os.path.join(tbldir, "per_gms_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(f"[OK] per-GMS summary → {out_sum}")

    # Build GMS→genes & background
    gms2genes: Dict[str, Set[str]] = {g: set() for g in gms_order}
    for _, r in A.iterrows():
        sid, gms = r["sample_id"], r["gms"]
        gms2genes[gms].update(sample2genes.get(sid, set()))
    background = set().union(*gms2genes.values()) if gms2genes else set()
    print(f"[GENES] Background unique genes: {len(background)}")

    # Load gene sets
    RCTS = load_genesets(args.reactome, "REACTOME")
    KEGG = load_genesets(args.kegg, "KEGG")
    GO_BP = load_genesets(args.go_bp, "GO_BP")
    GO_MF = load_genesets(args.go_mf, "GO_MF")
    GO_CC = load_genesets(args.go_cc, "GO_CC")

    # Enrichment for each collection
    def enrich_collection(col_map: Dict[str, Set[str]], label: str, outname: str) -> pd.DataFrame:
        if not col_map:
            return pd.DataFrame(columns=["gms","collection","set_label","k","K","n","N","p","fdr","overlap_genes"])
        out = []
        for g in gms_order:
            df = enrich_one_set(gms2genes[g], background, col_map, label, g)
            out.append(df)
        RES = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        outfile = os.path.join(tbldir, outname)
        RES.to_csv(outfile, index=False)
        print(f"[OK] enrichment {label} → {outfile} (rows={len(RES)})")
        return RES

    E_R = enrich_collection(RCTS, "Reactome", "enrichment_reactome.csv")
    E_K = enrich_collection(KEGG, "KEGG", "enrichment_kegg.csv")
    E_BP = enrich_collection(GO_BP, "GO_BP", "enrichment_go_bp.csv")
    E_MF = enrich_collection(GO_MF, "GO_MF", "enrichment_go_mf.csv")
    E_CC = enrich_collection(GO_CC, "GO_CC", "enrichment_go_cc.csv")

    # Heatmaps
    if not E_R.empty:
        heatmap_topn(E_R, gms_order, "Reactome enrichment (−log10 FDR)", os.path.join(figdir, "heatmap_reactome_topN.png"), args.topn_heatmap)
    if not E_K.empty:
        heatmap_topn(E_K, gms_order, "KEGG enrichment (−log10 FDR)", os.path.join(figdir, "heatmap_kegg_topN.png"), args.topn_heatmap)
    if not E_BP.empty:
        heatmap_topn(E_BP, gms_order, "GO BP enrichment (−log10 FDR)", os.path.join(figdir, "heatmap_go_bp_topN.png"), args.topn_heatmap)
    if not E_MF.empty:
        heatmap_topn(E_MF, gms_order, "GO MF enrichment (−log10 FDR)", os.path.join(figdir, "heatmap_go_mf_topN.png"), args.topn_heatmap)
    if not E_CC.empty:
        heatmap_topn(E_CC, gms_order, "GO CC enrichment (−log10 FDR)", os.path.join(figdir, "heatmap_go_cc_topN.png"), args.topn_heatmap)

    # Per-GMS barplots (topN)
    for g in gms_order:
        if not E_R.empty: bars_per_gms(E_R, g, "Reactome enrichment", os.path.join(figdir, f"bars_reactome_{g}.png"), args.topn_bars)
        if not E_K.empty: bars_per_gms(E_K, g, "KEGG enrichment", os.path.join(figdir, f"bars_kegg_{g}.png"), args.topn_bars)
        if not E_BP.empty: bars_per_gms(E_BP, g, "GO BP enrichment", os.path.join(figdir, f"bars_go_bp_{g}.png"), args.topn_bars)

    print("[DONE] Dashboard tables and figures generated.")

if __name__ == "__main__":
    main()
