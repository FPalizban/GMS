import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Feature Extraction (Direct from Raw Cases)
# ---------------------------------------------------------
def extract_cohort_features(folder_path):
    files = glob.glob(os.path.join(folder_path, "svpGDC_*_MSC.txt"))
    patient_data = []
    print(f"Reading {len(files)} cohort files for XAI attribution...")

    for f in files:
        sample_id = os.path.basename(f).replace("svpGDC_", "").replace("_MSC.txt", "")
        try:
            df = pd.read_csv(f, sep='\t', low_memory=False)
            total = len(df)
            if total == 0: continue

            # BASE Burden Features
            snv_f = len(df[df['Variant_Type'] == 'SNV']) / total
            indel_f = len(df[df['Variant_Type'].isin(['insertion', 'deletion'])]) / total
            high_f = len(df[df['VEP_IMPACT'] == 'HIGH']) / total

            # AUGMENTED Features (As defined in Methods 2.4)
            path_score = pd.to_numeric(df['Pathogenicity_score'], errors='coerce').fillna(0).sum()
            oe_lof = pd.to_numeric(df['gnomAD_oe_lof'], errors='coerce').mean()
            cwml = path_score / max(1, oe_lof) if not np.isnan(oe_lof) else path_score
            af = pd.to_numeric(df['gnomAD_AF_ALL'], errors='coerce').replace(0, 1e-6)
            aar = -np.log10(af).median()
            
            splice_col = 'Splicing_score' if 'Splicing_score' in df.columns else None
            splice_b = (pd.to_numeric(df[splice_col], errors='coerce') >= 0.2).sum() if splice_col else 0

            patient_data.append({
                'sample_id': sample_id, 'SNV_Frac': snv_f, 'Indel_Frac': indel_f,
                'HIGH_Frac': high_f, 'CWML': cwml, 'AAR': aar, 'Splice_Burden': splice_b
            })
        except: continue
    return pd.DataFrame(patient_data).set_index('sample_id')

# ---------------------------------------------------------
# 2. XAI Attribution & Visualization Engine
# ---------------------------------------------------------
def run_xai_pipeline(df, assignments, n_boots=200, out_dir="/Users/palizbanf/Documents/cancer/projects/GMS/MDPI_Biomedicines/revision/outputs/xai"):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    performance_metrics = []
    
    features = df.columns
    unique_gms = sorted(assignments.unique())

    for gms in unique_gms:
        print(f"Attributing {gms}...")
        y_true = (assignments == gms).astype(int)
        
        # A. Feature Attribution (AUROC)
        gms_aucs = []
        for col in features:
            scores = df[col].values
            auc_val = roc_auc_score(y_true, scores)
            
            # Find Youden Cutoff
            fpr, tpr, thr = roc_curve(y_true, scores)
            j_stat = tpr - fpr
            best_thr = thr[np.argmax(j_stat)]
            
            # 95% Bootstrap CIs (B=200 iterations)
            boot_stats = []
            for _ in range(n_boots):
                y_b, s_b = resample(y_true, scores)
                if len(np.unique(y_b)) > 1: 
                    boot_stats.append(roc_auc_score(y_b, s_b))
            ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])

            gms_aucs.append({'Feature': col, 'AUROC': auc_val, 'Cutoff': best_thr, 'CI_Low': ci_low, 'CI_High': ci_high})
        
        attr_df = pd.DataFrame(gms_aucs).sort_values('AUROC', ascending=False)
        attr_df['GMS'] = gms
        results.append(attr_df)

        # B. FIX: Visualization 1 - Top Feature ROC Curve
        # Changed .iloc['Feature'] to .iloc['Feature'] for integer indexing
        top_feat = attr_df.iloc['Feature'] 
        top_auc = attr_df.iloc['AUROC']
        fpr, tpr, _ = roc_curve(y_true, df[top_feat])
        
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {top_auc:.3f}")
        plt.plot([1], color='navy', lw=2, linestyle='--') # Fixed coordinates
        plt.title(f"{gms}: {top_feat} Attribution")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(f"{out_dir}/roc_{gms}.png", dpi=300)
        plt.close()

        # C. Visualization 2: Feature Importance Bar Plot
        plt.figure(figsize=(8, 4))
        sns.barplot(x='AUROC', y='Feature', data=attr_df, palette='magma')
        plt.axvline(0.5, color='red', linestyle='--')
        plt.title(f"Feature Attribution AUROC: {gms}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/importance_{gms}.png", dpi=300)
        plt.close()

        # D. Depth-2 Surrogate Decision Tree (Interpretability Surrogate)
        clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
        clf.fit(df, y_true)
        y_pred = clf.predict(df)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        rules = export_text(clf, feature_names=list(features))
        performance_metrics.append({'GMS': gms, 'Accuracy': acc, 'F1': f1, 'Rules': rules})

    # E. Final Performance Summary Plot
    perf_df = pd.DataFrame(performance_metrics)
    plt.figure(figsize=(12, 5))
    perf_melted = perf_df.melt(id_vars='GMS', value_vars=['Accuracy', 'F1'])
    sns.barplot(x='GMS', y='value', hue='variable', data=perf_melted, palette='muted')
    plt.title("Surrogate Decision Tree Performance (Interpretable Rules Check)")
    plt.ylabel("Metric Score")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/surrogate_performance_summary.png", dpi=300)
    plt.close()

    return pd.concat(results), perf_df

# ---------------------------------------------------------
# 3. Main Execution Engine
# ---------------------------------------------------------
if __name__ == "__main__":
    # Update to your actual data folder
    CASE_FOLDER = "/Users/palizbanf/Documents/cancer/GDCross"
    
    # 1. Feature Aggregation
    df_feat = extract_cohort_features(CASE_FOLDER)
    if df_feat.empty:
        print("[CRITICAL] Data extraction failed.")
    else:
        # 2. Simulate Clustering Assignments (K=13)
        X = StandardScaler().fit_transform(df_feat)
        gmm = GaussianMixture(n_components=13, random_state=42).fit(X)
        labels = gmm.predict(X)
        gms_labels = pd.Series([f"GMS{l+1}" for l in labels], index=df_feat.index)

        # 3. Run XAI Pipeline and Save
        final_attr, final_perf = run_xai_pipeline(df_feat, gms_labels)
        
        final_attr.to_csv("/Users/palizbanf/Documents/cancer/projects/GMS/MDPI_Biomedicines/revision/xai_feature_thresholds.csv", index=False)
        final_perf.to_csv("/Users/palizbanf/Documents/cancer/projects/GMS/MDPI_Biomedicines/revision/xai_surrogate_metrics.csv", index=False)
        
        print("\n[SUCCESS] Pipeline complete.")
        print(f"Tables saved: xai_feature_thresholds.csv, xai_surrogate_metrics.csv")
        print(f"Visualizations saved in: outputs/xai/")