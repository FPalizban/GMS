#!/usr/bin/env bash
# get_gene_sets_public.sh
# Fetch/build Reactome, KEGG (human), and GO BP/MF/CC gene sets into GMT files.
# Usage: bash get_gene_sets_public.sh [OUTDIR]
# Default OUTDIR: resources/pathways
set -euo pipefail

OUTDIR="${1:-resources/pathways}"
REACT="$OUTDIR/reactome"; KEGG="$OUTDIR/kegg"; GO="$OUTDIR/go"; TMP="$OUTDIR/tmp"
mkdir -p "$REACT" "$KEGG" "$GO" "$TMP"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found. Please install it."; exit 1; }; }
need curl
need python3
need unzip

echo "==> Output dir: $OUTDIR"

############################################
# 1) Reactome (ready-to-use GMT, public)
############################################
echo "[1/5] Reactome GMT (download)"
REACT_ZIP="$REACT/ReactomePathways.gmt.zip"
REACT_GMT="$REACT/ReactomePathways.gmt"
curl -L -C - -o "$REACT_ZIP" "https://reactome.org/download/current/ReactomePathways.gmt.zip"
unzip -o "$REACT_ZIP" -d "$REACT" >/dev/null
# normalize name
if [ ! -f "$REACT_GMT" ]; then
  found="$(ls "$REACT"/*.gmt | head -n1 || true)"
  if [ -n "$found" ]; then mv "$found" "$REACT_GMT"; fi
fi
test -f "$REACT_GMT" || { echo "ERROR: Reactome GMT not found after unzip."; exit 1; }
echo "    → $REACT_GMT"

############################################
# 2) KEGG (build GMT via REST; organism: human 'hsa')
############################################
echo "[2/5] KEGG GMT (build via REST)"
cat > "$TMP/build_kegg.py" <<'PY'
import os, sys, urllib.request, time
outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)

def fetch(url, tries=6):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            err = e; time.sleep(1+2*i)
    raise SystemExit(f"Failed to fetch {url}: {err}")

print("[KEGG] listing human pathways…")
pw_txt = fetch("https://rest.kegg.jp/list/pathway/hsa")
pw = {}
for line in pw_txt.strip().splitlines():
    if "\t" not in line: continue
    pid, name = line.split("\t", 1)  # pid like 'path:hsa04110'
    if not pid.startswith("path:hsa"): continue
    short = name.split(" - Homo sapiens")[0]
    pw[pid] = short

print(f"[KEGG] pathways: {len(pw)}")

print("[KEGG] linking genes↔pathways…")
link_txt = fetch("https://rest.kegg.jp/link/pathway/hsa")
p2g = {k:set() for k in pw}
for line in link_txt.strip().splitlines():
    if "\t" not in line: continue
    g, p = line.split("\t")
    if p in p2g:
        p2g[p].add(g)  # g like 'hsa:891'

print("[KEGG] mapping hsa:ID → symbols…")
hsa_txt = fetch("https://rest.kegg.jp/list/hsa")
id2sym = {}
for line in hsa_txt.strip().splitlines():
    if "\t" not in line: continue
    i, nm = line.split("\t",1)      # i like 'hsa:891'
    sym = nm.split(";")[0].split(",")[0].strip()
    if sym: id2sym[i] = sym

gmt = os.path.join(outdir, "kegg_hsa.gmt")
w = 0
with open(gmt, "w") as fh:
    for pid, genes in sorted(p2g.items()):
        if not genes: continue
        syms = sorted({id2sym.get(g,"") for g in genes if id2sym.get(g,"")})
        if not syms: continue
        label = f"KEGG_{pid.split(':')[1]}: {pw[pid]}"
        fh.write(label + "\tKEGG\t" + "\t".join(syms) + "\n")
        w += 1
print(f"[OK] KEGG GMT written: {gmt} (sets={w})")
PY
python3 "$TMP/build_kegg.py" "$KEGG"
KEGG_GMT="$KEGG/kegg_hsa.gmt"
test -f "$KEGG_GMT" || { echo "ERROR: KEGG GMT failed."; exit 1; }
echo "    → $KEGG_GMT"

############################################
# 3) GO ontology + GOA human annotations
############################################
echo "[3/5] GO ontology + GOA human"
curl -L -C - -o "$GO/go.obo" "http://purl.obolibrary.org/obo/go.obo"
curl -L -C - -o "$GO/goa_human.gaf.gz" "http://current.geneontology.org/annotations/goa_human.gaf.gz"

############################################
# 4) Build GO BP/MF/CC GMTs (exclude ND; exclude NOT)
############################################
echo "[4/5] Build GO BP/MF/CC GMTs"
cat > "$TMP/build_go_gmts.py" <<'PY'
import os, gzip
from collections import defaultdict

outdir = os.environ["GO_OUTDIR"]
obo_path = os.path.join(outdir, "go.obo")
gaf_gz   = os.path.join(outdir, "goa_human.gaf.gz")

# Parse GO id → name
go_name = {}
with open(obo_path, "r", encoding="utf-8", errors="ignore") as fh:
    cur = None
    for line in fh:
        line=line.rstrip("\n")
        if line=="[Term]":
            cur=None
        elif line.startswith("id: GO:"):
            cur = line.split("id: ",1)[1].strip()
        elif cur and line.startswith("name: "):
            go_name[cur]=line.split("name: ",1)[1].strip()

# Build sets by aspect from GOA (exclude ND evidence and NOT qualifiers)
sets = {"P": defaultdict(set), "F": defaultdict(set), "C": defaultdict(set)}
with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if not line or line.startswith("!"): continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 15: continue
        symbol = parts[2].strip()
        go_id  = parts[4].strip()
        aspect = parts[8].strip()  # P/F/C
        ev     = parts[6].strip()
        qual   = parts[3].strip()
        if not symbol or not go_id or aspect not in sets: continue
        if "NOT" in qual:  # exclude negations
            continue
        if ev == "ND":     # exclude 'No biological Data'
            continue
        sets[aspect][go_id].add(symbol)

def write_gmt(aspect, fname):
    path = os.path.join(outdir, fname)
    with open(path, "w") as fh:
        for go_id, genes in sorted(sets[aspect].items()):
            if not genes: continue
            name = go_name.get(go_id, "")
            label = f"{go_id} {name}".strip()
            fh.write(label + "\tGO\t" + "\t".join(sorted(genes)) + "\n")
    print(f"[OK] {aspect} → {path} (terms={len(sets[aspect])})")

write_gmt("P", "go_bp.gmt")
write_gmt("F", "go_mf.gmt")
write_gmt("C", "go_cc.gmt")
PY
GO_OUTDIR="$GO" python3 "$TMP/build_go_gmts.py"

############################################
# 5) Summary
############################################
echo "[5/5] Done. Gene set files:"
ls -lh "$REACT/ReactomePathways.gmt" \
      "$KEGG/kegg_hsa.gmt" \
      "$GO/go_bp.gmt" "$GO/go_mf.gmt" "$GO/go_cc.gmt" 2>/dev/null || true

echo
echo "Use these paths with your dashboard script:"
echo "  --reactome $REACT/ReactomePathways.gmt"
echo "  --kegg     $KEGG/kegg_hsa.gmt"
echo "  --go-bp    $GO/go_bp.gmt"
echo "  --go-mf    $GO/go_mf.gmt"
echo "  --go-cc    $GO/go_cc.gmt"
