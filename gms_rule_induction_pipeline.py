import os
import glob
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Feature Extraction Engine (Direct from Raw TXT)
# ---------------------------------------------------------
def extract_features_robust(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not files:
        files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
    
    patient_data = []
    print(f"Aggregating features from {len(files)} files...")

    # Column Sniffer Candidates
    TYPE_CAND = ['Variant_Type', 'VariantType', 'variant_type', 'type', 'Type']
    IMPACT_CAND = ['VEP_IMPACT', 'IMPACT', 'Impact', 'vep_impact']

    for f in files:
        basename = os.path.basename(f)
        sample_id = basename.split("__")[5] if "__" in basename else basename.replace("svpGDC_", "").replace("_MSC.txt", "")
        
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                first_line = fh.readline()
                sep = '\t' if first_line.count('\t') >= first_line.count(',') else ','
            df = pd.read_csv(f, sep=sep, low_memory=False, on_bad_lines='skip')
            if df.empty: continue

            # Fuzzy Match Columns
            type_col = next((c for c in df.columns if any(k in c.upper() for k in ['TYPE', 'VARIANT_TYPE'])), None)
            impact_col = next((c for c in df.columns if 'IMPACT' in c.upper()), None)
            path_col = next((c for c in df.columns if 'PATHOGENICITY' in c.upper() or 'SCORE' in c.upper()), None)
            af_col = next((c for c in df.columns if 'AF_ALL' in c.upper() or 'GNOMAD_AF' in c.upper()), None)
            
            if not type_col: continue
            
            total_vars = len(df)
            snv_f = len(df[df[type_col].astype(str).str.upper() == 'SNV']) / total_vars
            indel_f = len(df[df[type_col].astype(str).str.upper().isin(['INSERTION', 'DELETION', 'INDEL'])]) / total_vars
            high_f = (len(df[df[impact_col].astype(str).str.upper() == 'HIGH']) / total_vars) if impact_col else 0
            cwml = pd.to_numeric(df[path_col], errors='coerce').sum() if path_col else 0
            aar = -np.log10(pd.to_numeric(df[af_col], errors='coerce').replace(0, 1e-6)).median() if af_col else 0
            
            patient_data.append({
                'sample_id': sample_id, 'SNV_Frac': snv_f, 'Indel_Frac': indel_f, 
                'HIGH_Frac': high_f, 'CWML': cwml, 'AAR': aar
            })
        except: continue
        
    return pd.DataFrame(patient_data).set_index('sample_id')

# ---------------------------------------------------------
# 2. XAI Core: Youden Cutoffs & Decision Tree Rules
# ---------------------------------------------------------
def perform_rule_induction(features_df, assignments, n_boots=200):
    results = []
    tree_rules = {}
    
    unique_gms = sorted(assignments.unique())
    
    for gms in unique_gms:
        print(f"Inducing rules for {gms}...")
        y_true = (assignments == gms).astype(int)
        
        # 1. Feature Attribution (One-vs-Rest AUROC)
        gms_stats = []
        for col in features_df.columns:
            scores = features_df[col].values
            
            # AUROC and Youden Cutoff
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            j_stat = tpr - fpr
            best_idx = np.argmax(j_stat)
            cutoff = thresholds[best_idx]
            auc_val = roc_auc_score(y_true, scores)
            
            # 95% Bootstrap CI
            boot_aucs = []
            for _ in range(n_boots):
                y_b, s_b = resample(y_true, scores)
                if len(np.unique(y_b)) > 1:
                    boot_aucs.append(roc_auc_score(y_b, s_b))
            ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
            
            gms_stats.append({
                'GMS': gms, 'Feature': col, 'AUROC': auc_val, 
                'Youden_Cutoff': cutoff, 'CI_95_Lower': ci_low, 'CI_95_Upper': ci_high
            })
        
        results.extend(gms_stats)

        # 2. Depth-2 Surrogate Decision Tree
        clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
        clf.fit(features_df, y_true)
        y_pred = clf.predict(features_df)
        
        # Tree Metrics
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Rule Extraction
        rules_text = export_text(clf, feature_names=list(features_df.columns))
        tree_rules[gms] = {
            'rules': rules_text, 'Accuracy': acc, 'F1': f1, 'Precision': p
        }

    return pd.DataFrame(results), tree_rules

# ---------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    CASE_PATH = "/Users/palizbanf/Documents/cancer/GDCross"
    
    # Extract data and simulate/load clustering for attribution
    df = extract_features_robust(CASE_PATH)
    X = StandardScaler().fit_transform(df)
    
    # Perform Clustering (K=13 as per BIC elbow)
    gmm = GaussianMixture(n_components=13, random_state=42)
    labels = gmm.fit_predict(X)
    df['GMS_Label'] = [f"GMS{l+1}" for l in labels]
    
    # Run XAI Pipeline
    threshold_df, tree_manifest = perform_rule_induction(df.drop(columns='GMS_Label'), df['GMS_Label'])
    
    # Save Outputs
    threshold_df.to_csv("/Users/palizbanf/Documents/cancer/projects/GMS/MDPI_Biomedicines/revision/xai_thresholds_per_signature.csv", index=False)
    with open("/Users/palizbanf/Documents/cancer/projects/GMS/MDPI_Biomedicines/revision/xai_tree_rules.txt", "w") as f:
        for gms, data in tree_manifest.items():
            f.write(f"--- {gms} Rules (F1: {data['F1']:.3f}, Acc: {data['Accuracy']:.3f}) ---\n")
            f.write(data['rules'] + "\n\n")
            
    print("\n[SUCCESS] XAI analysis complete.")
    print("Files saved: xai_thresholds_per_signature.csv and xai_tree_rules.txt")