# Germline Mutation Signatures (GMS) in Pediatric Cancer
This repository contains the source code and analytical pipeline for the manuscript: "Deep learning-driven discovery of germline mutation signatures in pediatric cancers." This project introduces a novel, unsupervised machine learning framework designed to decode the latent genomic architecture of pediatric cancer predisposition. By integrating high-dimensional whole-exome/whole-genome (WES/WGS) annotations with deep representation learning, we define 13 distinct Germline Mutation Signatures (GMSs) that bridge the gap between single-gene Mendelian risk and pathway-centric translational oncology.
# Project Overview
Historically, pediatric germline analysis has relied on gene-centric pathogenicity models. This framework shifts to a cohort-level, systems-wide approach. The pipeline processes an augmented feature set of germline variants—moving beyond standard mutational burden to include noncoding, regulatory, and structural metrics—compressing them into a latent space to reveal shared inherited vulnerabilities across diverse pediatric cancer types.

Key Capabilities:

Feature Integration: Processes complex variant annotations including constraint-weighted missense load, compositional CLR spectra, and ENCODE cCRE overlaps.

Deep Representation Learning: Utilizes an unsupervised Autoencoder to compress high-dimensional genomic features while preserving non-linear biological relationships.

Probabilistic Clustering: Applies Gaussian Mixture Modeling (GMM) to identify robust, reproducible patient clusters (GMS 1–13).

Translational Mapping: Links discovered signatures to ACMG/AMP clinical classifications, pathway enrichments, and targeted therapeutics.

# Tools & Dependencies
The pipeline is primarily built in Python and relies on standard bioinformatics and machine learning libraries.

Core Machine Learning & Data Processing:

scikit-learn: For Gaussian Mixture Modeling, data scaling, and internal validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).

PyTorch / TensorFlow (choose the one you used): For building and training the deep autoencoder.

pandas & numpy: For high-dimensional genomic matrix manipulation.

scipy: For compositional data analysis (CLR transformations).
