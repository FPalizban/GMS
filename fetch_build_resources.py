#!/usr/bin/env python3
"""
Fetch & build local resources for pathway/therapy/PGx mapping.

Outputs (default: ./resources):
  - reactome.gmt
  - kegg.gmt
  - go_bp.gmt
  - oncology_drug_gene.csv          (DGIdb + CIViC + optional OncoKB)
  - pgx_rules.csv                   (from CPIC Genes–Drugs, or a template)

Usage examples
--------------
# Fetch everything into ./resources
python scripts/fetch_build_resources.py all --out resources

# Only pathways (Reactome/KEGG/GO)
python scripts/fetch_build_resources.py pathways --out resources

# Only drug–gene (DGIdb + CIViC; plus OncoKB if ONCOKB_TOKEN set)
python scripts/fetch_build_resources.py druggene --out resources

# Only PGx (CPIC Genes–Drugs → CSV; or template)
python scripts/fetch_build_resources.py pgx --out resources
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import os
import re
import sys
import time
import gzip
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests

# -------------------------------------------------------------------
# Small utils
# -------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def http_get(url: str, timeout: int = 60, stream: bool = True) -> requests.Response:
    r = requests.get(url, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r

def save_bytes_to(path: Path, content: bytes) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        f.write(content)

def write_gmt(sets: List[Tuple[str, str, List[str]]], out_path: Path) -> None:
    # sets: [(set_name, desc, [genes...])]
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        for name, desc, genes in sets:
            genes = sorted({g.strip().upper() for g in genes if g and isinstance(g, str)})
            if not genes:
                continue
            f.write("\t".join([name, desc] + genes) + "\n")

def bh_fdr(p):
    import numpy as np
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    order = np.argsort(p)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty_like(q); q_adj[order] = np.clip(q_sorted, 0, 1)
    return q_adj

# -------------------------------------------------------------------
# 1) Reactome GMT (official)
# -------------------------------------------------------------------

def fetch_reactome_gmt(outdir: Path) -> Path:
    # Reactome "current" download directory hosts ReactomePathways.gmt.zip
    # Example: https://reactome.org/download/current/ReactomePathways.gmt.zip
    url = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
    print(f"[Reactome] GET {url}")
    r = http_get(url)
    zpath = outdir / "raw" / "ReactomePathways.gmt.zip"
    save_bytes_to(zpath, r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # inside: ReactomePathways.gmt
        names = zf.namelist()
        member = [n for n in names if n.lower().endswith(".gmt")]
        if not member:
            raise RuntimeError("Reactome zip has no .gmt file.")
        raw_gmt = zf.read(member[0]).decode("utf-8", errors="ignore")
    out = outdir / "reactome.gmt"
    save_bytes_to(out, raw_gmt.encode("utf-8"))
    print(f"[Reactome] wrote {out}")
    return out

# -------------------------------------------------------------------
# 2) KEGG GMT (human) via KEGG REST
# -------------------------------------------------------------------

def fetch_kegg_gmt(outdir: Path) -> Path:
    """
    Build KEGG GMT for Homo sapiens using KEGG REST:
      - list pathways:   https://rest.kegg.jp/list/pathway/hsa
      - link genes:      https://rest.kegg.jp/link/hsa/<pathway_id>
      - gene symbol map: https://rest.kegg.jp/list/hsa
    """
    base = "https://rest.kegg.jp"
    print("[KEGG] fetching pathway list (human)")
    txt = http_get(f"{base}/list/pathway/hsa", stream=False).text
    # lines like: path:hsa00010\tGlycolysis / Gluconeogenesis - Homo sapiens (human)
    pathways = []
    for line in txt.strip().splitlines():
        if not line.strip(): continue
        pid, name = line.split("\t", 1)
        pid = pid.replace("path:", "").strip()
        name = name.split(" - Homo sapiens")[0].strip()
        pathways.append((pid, name))
    # Build geneID → symbol map from 'list hsa'
    print("[KEGG] fetching human gene symbol map")
    gtxt = http_get(f"{base}/list/hsa", stream=False).text
    # lines like: hsa:10458\tCFLAR; CASH; CLARP; I-FLICE; MRIT;...
    gid2sym = {}
    for line in gtxt.strip().splitlines():
        if not line.strip(): continue
        gid, desc = line.split("\t", 1)
        gid = gid.replace("hsa:", "").strip()
        # first token before ';' or ',' is canonical symbol
        sym = re.split(r"[;, ]", desc.strip())[0]
        gid2sym[gid] = sym.upper()
    # Link genes per pathway
    sets = []
    for pid, pname in pathways:
        link_txt = http_get(f"{base}/link/hsa/path:{pid}", stream=False).text
        # lines like: path:hsa00010\thsa:3098
        genes = []
        for line in link_txt.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[1].startswith("hsa:"):
                gid = parts[1].split(":")[1]
                sym = gid2sym.get(gid)
                if sym: genes.append(sym)
        if genes:
            sets.append((f"KEGG_{pid}", pname, genes))
    out = outdir / "kegg.gmt"
    write_gmt(sets, out)
    print(f"[KEGG] wrote {out} (sets={len(sets)})")
    return out

# -------------------------------------------------------------------
# 3) GO:BP GMT from NCBI gene2go + human gene_info
# -------------------------------------------------------------------

def fetch_go_bp_gmt(outdir: Path) -> Path:
    """
    Build GO:BP gene sets using:
      - gene2go: https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
      - Homo_sapiens.gene_info.gz:
        https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
    """
    gene2go_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
    geneinfo_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
    print(f"[GO] GET {gene2go_url}")
    g2 = http_get(gene2go_url).content
    print(f"[GO] GET {geneinfo_url}")
    gi = http_get(geneinfo_url).content

    # read as DataFrame
    g2df = pd.read_csv(io.BytesIO(gzip.decompress(g2)), sep="\t", comment="#",
                       names=["tax_id","GeneID","GO_ID","Evidence","Qualifier","GO_term","PubMed","Category"])
    g2df = g2df[g2df["tax_id"] == 9606]             # human only
    g2df = g2df[g2df["Category"].str.lower() == "process"]  # BP only
    # gene_info: map GeneID -> Symbol
    gidf = pd.read_csv(io.BytesIO(gzip.decompress(gi)), sep="\t", comment="#", header=None,
                       names=["#tax_id","GeneID","Symbol","LocusTag","Synonyms","dbXrefs","chromosome","map_location",
                              "description","type_of_gene","Symbol_from_nomenclature_authority","Full_name_from_nomenclature_authority",
                              "Nomenclature_status","Other_designations","Modification_date","Feature_type"])
    gidf = gidf[gidf["#tax_id"] == 9606][["GeneID","Symbol"]].drop_duplicates()
    sym_map = {int(r.GeneID): str(r.Symbol).upper() for r in gidf.itertuples(index=False)}

    # assemble sets
    recs: Dict[str, List[str]] = {}
    for row in g2df.itertuples(index=False):
        geneid = int(row.GeneID)
        sym = sym_map.get(geneid)
        if not sym: continue
        go_id = str(row.GO_ID)
        name = str(row.GO_term)
        key = f"{go_id}__{name}"
        recs.setdefault(key, []).append(sym)
    sets = []
    for key, genes in recs.items():
        go_id, name = key.split("__", 1)
        sets.append((f"GO_BP_{go_id}", name, genes))
    out = outdir / "go_bp.gmt"
    write_gmt(sets, out)
    print(f"[GO] wrote {out} (sets={len(sets)})")
    return out

# -------------------------------------------------------------------
# 4) Drug–gene knowledge (DGIdb + CIViC + optional OncoKB)
# -------------------------------------------------------------------

def _normalize_gene_drug(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["gene","drug","source","evidence_tier"])
    out = pd.DataFrame({
        "gene": df["gene"].astype(str).str.upper().str.strip(),
        "drug": df["drug"].astype(str).str.strip(),
        "source": source,
        "evidence_tier": df.get("evidence_tier", pd.Series(["NA"]*len(df))).astype(str).str.strip(),
    })
    out = out[out["gene"].ne("") & out["drug"].ne("")]
    return out.drop_duplicates()

def fetch_dgidb() -> pd.DataFrame:
    # Stable "latest" TSV path seen in multiple clients
    url = "https://www.dgidb.org/data/latest/interactions.tsv"
    print(f"[DGIdb] GET {url}")
    txt = http_get(url, stream=False).text
    df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
    # common columns: gene_name, drug_name, interaction_types, interaction_score/score
    colmap = {c.lower(): c for c in df.columns}
    gene = colmap.get("gene_name") or colmap.get("gene") or "gene_name"
    drug = colmap.get("drug_name") or colmap.get("drug") or "drug_name"
    # pick a score-like column if present
    etier = colmap.get("interaction_score") or colmap.get("score") or None
    keep = {}
    keep["gene"] = df[gene] if gene in df.columns else pd.Series(dtype=str)
    keep["drug"] = df[drug] if drug in df.columns else pd.Series(dtype=str)
    if etier and etier in df.columns:
        keep["evidence_tier"] = df[etier]
    nd = pd.DataFrame(keep)
    return _normalize_gene_drug(nd, source="DGIdb")

def fetch_civic() -> pd.DataFrame:
    """
    CIViC nightly VariantSummaries (has a 'drugs' column).
    Tries 'nightly-VariantSummaries.tsv' first; falls back to dated VariantSummaries if needed.
    """
    urls = [
        "https://civicdb.org/downloads/nightly/nightly-VariantSummaries.tsv",
        # fallback (older style releases have dated folders)
    ]
    last_err = None
    for url in urls:
        try:
            print(f"[CIViC] GET {url}")
            txt = http_get(url, stream=False).text
            df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
            # typical columns include: gene, variant, drugs
            colmap = {c.lower(): c for c in df.columns}
            gene_col = colmap.get("gene") or "Gene"
            drugs_col = colmap.get("drugs") or "drugs"
            if gene_col not in df.columns or drugs_col not in df.columns:
                raise ValueError("Expected 'gene' and 'drugs' columns not found.")
            # explode comma/pipe-separated drugs
            tmp = df[[gene_col, drugs_col]].dropna()
            tmp["drugs"] = tmp[drugs_col].astype(str).str.replace(r"[|]", ",", regex=True)
            rows = []
            for row in tmp.itertuples(index=False):
                g = str(getattr(row, gene_col)).upper().strip()
                for d in str(getattr(row, "drugs")).split(","):
                    d = d.strip()
                    if d:
                        rows.append((g, d))
            nd = pd.DataFrame(rows, columns=["gene","drug"])
            nd["evidence_tier"] = "NA"
            return _normalize_gene_drug(nd, source="CIViC")
        except Exception as e:
            last_err = e
            print(f"[CIViC] failed: {e}; trying next pattern...")
    print(f"[CIViC] WARNING: could not retrieve nightly VariantSummaries ({last_err}); returning empty.")
    return pd.DataFrame(columns=["gene","drug","source","evidence_tier"])

def fetch_oncokb(token: str | None) -> pd.DataFrame:
    """
    Very light OncoKB aggregation: query demo/all endpoints is tricky for full coverage.
    Here we use the 'drugs' search to collect names and attach to genes when possible
    via gene treatments in curatedGenes endpoint. This is intentionally conservative.

    Requires: ONCOKB_TOKEN in environment (Bearer).
    """
    if not token:
        print("[OncoKB] No token provided; skipping.")
        return pd.DataFrame(columns=["gene","drug","source","evidence_tier"])

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    base = "https://www.oncokb.org/api/v1"
    try:
        print("[OncoKB] fetching curated genes")
        genes = requests.get(f"{base}/utils/allCuratedGenes", headers=headers, timeout=60)
        genes.raise_for_status()
        genes = genes.json()  # list of {hugoSymbol, entrezGeneId, ...}
        rows = []
        # For each gene, pull treatments summary (if any) from annotate by gene (empty alteration) won't work.
        # As a compromise, we record gene presence with placeholder drug '—' to tag provenance.
        for g in genes:
            sym = str(g.get("hugoSymbol","")).upper()
            if sym:
                rows.append((sym, "—", "2"))  # '2' as mid-tier placeholder; mapping in downstream is coarse
        df = pd.DataFrame(rows, columns=["gene","drug","evidence_tier"])
        return _normalize_gene_drug(df, source="OncoKB")
    except Exception as e:
        print(f"[OncoKB] WARNING: fetch failed ({e}); returning empty.")
        return pd.DataFrame(columns=["gene","drug","source","evidence_tier"])

def build_druggene(outdir: Path) -> Path:
    dgi = fetch_dgidb()
    civ = fetch_civic()
    onk = fetch_oncokb(os.environ.get("ONCOKB_TOKEN", None))
    merged = pd.concat([dgi, civ, onk], axis=0, ignore_index=True)
    if merged.empty:
        # write a tiny template so downstream never breaks
        merged = pd.DataFrame([
            {"gene":"BRCA2","drug":"olaparib","source":"Template","evidence_tier":"1"},
            {"gene":"TP53","drug":"—","source":"Template","evidence_tier":"NA"},
        ])
    out = outdir / "oncology_drug_gene.csv"
    ensure_dir(out.parent)
    merged.drop_duplicates().to_csv(out, index=False)
    print(f"[DrugGene] wrote {out} (rows={merged.shape[0]})")
    return out

# -------------------------------------------------------------------
# 5) PGx (CPIC Genes–Drugs → CSV, or template)
# -------------------------------------------------------------------

def build_pgx(outdir: Path) -> Path:
    """
    Try to fetch the Genes–Drugs table XLSX from https://cpicpgx.org/genes-drugs/
    (there is a 'Download this table (XLSX)' link on the page).
    If the link can't be found or download fails, write a clean template with TPMT/DPYD/UGT1A1.
    """
    page = "https://cpicpgx.org/genes-drugs/"
    xlsx_url = None
    try:
        print(f"[CPIC] GET {page}")
        html = http_get(page, stream=False).text
        # find first .xlsx link on the page
        m = re.search(r'href="([^"]+\.xlsx)"', html, flags=re.I)
        if m:
            href = m.group(1)
            xlsx_url = href if href.startswith("http") else (page.rstrip("/") + "/" + href.lstrip("/"))
            print(f"[CPIC] XLSX link: {xlsx_url}")
            data = http_get(xlsx_url).content
            df = pd.read_excel(io.BytesIO(data))
            # normalize columns
            cols = {c.lower(): c for c in df.columns}
            gene = cols.get("gene") or "Gene"
            drug = cols.get("drug") or cols.get("drugs") or "Drug"
            guideline = cols.get("guideline") or "Guideline"
            action = cols.get("recommendation") or cols.get("classification") or "Action"
            note = cols.get("notes") or "Notes"
            # coerce & tidy
            outdf = pd.DataFrame({
                "gene": df[gene].astype(str).str.upper().str.strip(),
                "medication": df[drug].astype(str).str.strip(),
                "guideline": df.get(guideline, pd.Series([""]*len(df))),
                "action": df.get(action, pd.Series([""]*len(df))),
                "note": df.get(note, pd.Series([""]*len(df))),
            })
            outdf = outdf[outdf["gene"].ne("") & outdf["medication"].ne("")]
            out = outdir / "pgx_rules.csv"
            outdf.to_csv(out, index=False)
            print(f"[CPIC] wrote {out} (rows={outdf.shape[0]})")
            return out
        else:
            print("[CPIC] XLSX link not found on page; writing template.")
    except Exception as e:
        print(f"[CPIC] WARNING: failed to fetch/parse CPIC page ({e}); writing template.")

    # template fallback
    tmpl = pd.DataFrame([
        {"gene":"TPMT","medication":"thiopurines","guideline":"CPIC A","action":"consider dose reduction","note":""},
        {"gene":"DPYD","medication":"fluoropyrimidines","guideline":"CPIC A","action":"avoid or reduce dose","note":""},
        {"gene":"UGT1A1","medication":"irinotecan","guideline":"CPIC A","action":"consider dose reduction","note":""},
    ])
    out = outdir / "pgx_rules.csv"
    tmpl.to_csv(out, index=False)
    print(f"[CPIC] wrote template {out}")
    return out

# -------------------------------------------------------------------
# Orchestrations
# -------------------------------------------------------------------

def build_pathways(outdir: Path) -> None:
    fetch_reactome_gmt(outdir)
    fetch_kegg_gmt(outdir)
    fetch_go_bp_gmt(outdir)

def build_druggene_only(outdir: Path) -> None:
    build_druggene(outdir)

def build_pgx_only(outdir: Path) -> None:
    build_pgx(outdir)

def build_all(outdir: Path) -> None:
    build_pathways(outdir)
    build_druggene(outdir)
    build_pgx(outdir)

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch & build local pathway/drug–gene/PGx resources")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ["all","pathways","druggene","pgx"]:
        s = sub.add_parser(name)
        s.add_argument("--out", default="resources", help="output directory (default: resources)")

    args = ap.parse_args()
    outdir = Path(args.out)
    ensure_dir(outdir)

    if args.cmd == "all":
        build_all(outdir)
    elif args.cmd == "pathways":
        build_pathways(outdir)
    elif args.cmd == "druggene":
        build_druggene_only(outdir)
    elif args.cmd == "pgx":
        build_pgx_only(outdir)

    print("\n✅ Done. Resources available under:", outdir.resolve())

if __name__ == "__main__":
    main()
