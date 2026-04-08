#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-resources/hg38}"
mkdir -p "$OUTDIR"/{fasta,gtf,regulatory,cpg,constraint,spliceai,tmp}

echo "[1/6] Reference genome (GRCh38 primary assembly) + index"
# GENCODE provides hg38 FASTA aligned with their GTFs.
# Pick a specific release to stay reproducible; v46 shown here.
curl -L -o "$OUTDIR/fasta/GRCh38.primary_assembly.genome.fa.gz" \
  "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/GRCh38.primary_assembly.genome.fa.gz"
gunzip -c "$OUTDIR/fasta/GRCh38.primary_assembly.genome.fa.gz" > "$OUTDIR/fasta/GRCh38.primary_assembly.genome.fa"
samtools faidx "$OUTDIR/fasta/GRCh38.primary_assembly.genome.fa"

echo "[2/6] GENCODE GTF (basic) for GRCh38 (v46)"
curl -L -o "$OUTDIR/gtf/gencode.v46.basic.annotation.gtf.gz" \
  "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz"
gunzip -c "$OUTDIR/gtf/gencode.v46.basic.annotation.gtf.gz" > "$OUTDIR/gtf/gencode.v46.basic.annotation.gtf"

echo "[3/6] ENCODE cCREs (hg38, all classes; BED)"
# This is the canonical GRCh38 cCREs v2 bed (downloadable without login).
curl -L -o "$OUTDIR/regulatory/ENCFF924IMH.bed.gz" \
  "https://www.encodeproject.org/files/ENCFF924IMH/@@download/ENCFF924IMH.bed.gz"
gunzip -c "$OUTDIR/regulatory/ENCFF924IMH.bed.gz" > "$OUTDIR/regulatory/ccres.hg38.bed"
# Split into crude promoter vs enhancer sets by cCRE class (PLS, pELS/dELS).
awk 'BEGIN{FS=OFS="\t"} $4 ~ /PLS/ {print}' "$OUTDIR/regulatory/ccres.hg38.bed" > "$OUTDIR/regulatory/promoters.ccres.hg38.bed"
awk 'BEGIN{FS=OFS="\t"} $4 ~ /ELS/ {print}' "$OUTDIR/regulatory/ccres.hg38.bed" > "$OUTDIR/regulatory/enhancers.ccres.hg38.bed"

echo "[4/6] CpG islands (UCSC cpgIslandExt -> BED)"
curl -L -o "$OUTDIR/cpg/cpgIslandExt.txt.gz" \
  "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
gunzip -c "$OUTDIR/cpg/cpgIslandExt.txt.gz" \
  | awk 'BEGIN{FS=OFS="\t"} {print $2,$3,$4,$5}' > "$OUTDIR/cpg/cpg_islands.hg38.bed"
# Columns become: chrom, chromStart, chromEnd, name

echo "[5/6] Gene constraint (two options)"

# (A) UCSC hg38 table with gnomAD constraint (gene-level; v2.1.1 basis)
# Requires UCSC public MySQL client; produces a simple CSV with gene symbol + scores.
mysql --host=genome-mysql.soe.ucsc.edu -u genome -A -e \
  "SELECT gene, pLI, lof_z, mis_z, syn_z, oeu_lof AS oe_lof, loeuf FROM pliByGene" hg38 \
  | sed '1s/oeu_lof/oe_lof/' > "$OUTDIR/constraint/gnomad_constraint_v2like_hg38.csv" || true

# (B) Newer gnomAD v4/v4.1 transcript-level constraint (beta) via UCSC
# You can also pull pliByTranscriptV4_1 similarly (optional):
mysql --host=genome-mysql.soe.ucsc.edu -u genome -A -e \
  "SELECT name AS transcript, geneSymbol, oe_lof, loeuf, mis_z, syn_z FROM pliByTranscriptV4_1" hg38 \
  > "$OUTDIR/constraint/gnomad_constraint_v4_transcripts_hg38.csv" || true

echo "[6/6] SpliceAI precomputed (SNVs; hg38)"
# Option 1 (easy, SNV-only): Ensembl-provided SNV VCF for MANE transcripts
# (available in the 'variation_plugins' directory; comes tabix-indexed).
curl -L -o "$OUTDIR/spliceai/spliceai_scores.raw.snv.ensembl_mane.grch38.110.vcf.gz" \
  "https://ftp.ensembl.org/pub/data_files/homo_sapiens/GRCh38/variation_plugins/spliceai_scores.raw.snv.ensembl_mane.grch38.110.vcf.gz"
# Index (should already have .tbi; this re-creates if needed)
tabix -p vcf "$OUTDIR/spliceai/spliceai_scores.raw.snv.ensembl_mane.grch38.110.vcf.gz" || true

# Option 2 (full Illumina precomputed, SNVs/indels; very large; needs BaseSpace login)
# See notes in the docs below to fetch: spliceai_scores.raw.snv.hg38.vcf.gz (+ indels)
# and then:
# tabix -p vcf spliceai_scores.raw.snv.hg38.vcf.gz
# tabix -p vcf spliceai_scores.raw.indel.hg38.vcf.gz

echo "[OK] All done → $OUTDIR"
