import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Feature Extraction Engine (Case-Optimized)
# ---------------------------------------------------------
def extract_case_features(folder_path):
    # Matches the standard svpGDC_..._MSC.txt files [4, 5]
    files = glob.glob(os.path.join(folder_path, "svpGDC_*_MSC.txt"))
    if not files:
        # Fallback for general .txt files if naming differs
        files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not files:
        print(f"[ERROR] No files found in {folder_path}.")
        return pd.DataFrame()

    patient_data = []
    print(f"Processing {len(files)} case files...")

    for f in files:
        # Derive sample ID safely to avoid AttributeError [3]
        sample_id = os.path.basename(f).replace("svpGDC_", "").replace("_MSC.txt", "")
        
        try:
            df = pd.read_csv(f, sep='\t', low_memory=False)
            if df.empty: continue
            total_vars = len(df)
            
            # --- BASE Burden Features ---
            # Columns taken directly from source MSC files [4, 5]
            snv_frac = len(df[df['Variant_Type'].str.upper() == 'SNV']) / total_vars
            indel_frac = len(df[df['Variant_Type'].str.upper().isin(['INSERTION', 'DELETION', 'INDEL'])]) / total_vars
            high_frac = len(df[df['VEP_IMPACT'].str.upper() == 'HIGH']) / total_vars
            
            # --- AUGMENTED Features (Reviewer requested ablation) [6, 7] ---
            # CWML: Pathogenicity weighted by gene constraint (gnomAD oe_lof)
            path_scores = pd.to_numeric(df['Pathogenicity_score'], errors='coerce').fillna(0)
            oe_lof = pd.to_numeric(df['gnomAD_oe_lof'], errors='coerce').fillna(1)
            cwml = path_scores.sum() / max(1, oe_lof.mean())
            
            # AAR: Ancestry-Aware Rareness (median -log10 frequency)
            af = pd.to_numeric(df['gnomAD_AF_ALL'], errors='coerce').fillna(0.01).replace(0, 1e-6)
            aar = -np.log10(af).median()
            
            # Splice Burden: Variants with SpliceAI Delta >= 0.2
            splice_scores = pd.to_numeric(df['Splicing_score'], errors='coerce').fillna(0)
            splice_burden = (splice_scores >= 0.2).sum()

            patient_data.append({
                'sample_id': sample_id,
                'SNV_Frac': snv_frac, 'Indel_Frac': indel_frac, 'HIGH_Frac': high_frac,
                'CWML': cwml, 'AAR': aar, 'Splice_Burden': splice_burden
            })
        except Exception:
            continue

    if not patient_data: return pd.DataFrame()
    return pd.DataFrame(patient_data).set_index('sample_id')

# ---------------------------------------------------------
# 2. Symmetric Deep Autoencoder Architecture (Methods 2.5) [8, 9]
# ---------------------------------------------------------
class GMSAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Architecture: N -> 256 -> 128 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x):
        return self.encoder(x)

# ---------------------------------------------------------
# 3. Execution Engine
# ---------------------------------------------------------
def run_ablation_study(folder):
    df = extract_case_features(folder)
    if df.empty: return

    # Standardize data
    X_scaled = StandardScaler().fit_transform(df)
    
    # --- Framework 1: Primary (AE + GMM) ---
    print("Running Primary Model (AE+GMM)...")
    torch.manual_seed(42)
    # FIX: Using index [1] for column count
    ae = GMSAutoencoder(X_scaled.shape[1]) 
    latent_ae = ae(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy()
    labels_ae = GaussianMixture(n_components=13, random_state=42).fit_predict(latent_ae)
    
    # --- Framework 2: Linear Baseline (PCA + GMM) ---
    print("Running Linear Baseline (PCA+GMM)...")
    latent_pca = PCA(n_components=min(32, X_scaled.shape[1])).fit_transform(X_scaled)
    labels_pca = GaussianMixture(n_components=13, random_state=42).fit_predict(latent_pca)

    # --- Framework 3: Base-Only (GMM on Burden Features) ---
    print("Running Base-Only Sensitivity...")
    base_cols = ['SNV_Frac', 'Indel_Frac', 'HIGH_Frac']
    X_base = StandardScaler().fit_transform(df[base_cols])
    labels_base = GaussianMixture(n_components=13, random_state=42).fit_predict(X_base)

    # Compile Scores for Section 3.7 [1, 10]
    results = {
        'Framework': ['Primary (AE+GMM)', 'Linear (PCA+GMM)', 'GMM (Base Burden Only)'],
        'Silhouette': [
            silhouette_score(latent_ae, labels_ae),
            silhouette_score(latent_pca, labels_pca),
            silhouette_score(X_base, labels_base)
        ],
        'Davies-Bouldin': [
            davies_bouldin_score(latent_ae, labels_ae),
            davies_bouldin_score(latent_pca, labels_pca),
            davies_bouldin_score(X_base, labels_base)
        ]
    }
    
    res_df = pd.DataFrame(results)
    print("\n--- ABLATION RESULTS FOR MANUSCRIPT ---")
    print(res_df.to_string(index=False))

    # Save visualization for Supplementary Materials [11]
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Framework', y='Silhouette', data=res_df, palette='viridis')
    plt.title('Ablation Analysis: Clustering Quality (Silhouette Score)')
    plt.savefig('/Users/palizbanf/Documents/cancer/projects/GMS/MDPI_Biomedicines/revision/ablation_metrics_plot.png', dpi=300)
    print("\n[SUCCESS] Use these scores to complete Section 3.7.")

if __name__ == "__main__":
    CASE_FOLDER = "/Users/palizbanf/Documents/cancer/GDCross"
    run_ablation_study(CASE_FOLDER)