import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit

def expected_calibration_error(y_true, prob, bins=10):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(prob, bin_edges) - 1
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        mask = (bin_ids == i)
        if np.any(mask):
            bin_size = np.sum(mask)
            acc = np.mean(y_true[mask])
            conf = np.mean(prob[mask])
            ece += (bin_size / n) * abs(acc - conf)
    return float(ece)

def bootstrap_ci_and_delta(y_true, probs_dict, n_boot=1000, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    # Store individual metrics
    metrics = {arm: {'auroc': [], 'auprc': [], 'brier': []} for arm in probs_dict}
    # Store deltas vs M_vital
    deltas = {'M_lab_vs_vital': {'auroc': [], 'auprc': []},
              'M_integrated_vs_vital': {'auroc': [], 'auprc': []}}
    
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        
        cur = {}
        for arm, p in probs_dict.items():
            pb = p[idx]
            cur[arm] = {
                'auroc': roc_auc_score(y_b, pb),
                'auprc': average_precision_score(y_b, pb),
                'brier': brier_score_loss(y_b, pb)
            }
            metrics[arm]['auroc'].append(cur[arm]['auroc'])
            metrics[arm]['auprc'].append(cur[arm]['auprc'])
            metrics[arm]['brier'].append(cur[arm]['brier'])
            
        deltas['M_lab_vs_vital']['auroc'].append(cur['M_lab']['auroc'] - cur['M_vital']['auroc'])
        deltas['M_lab_vs_vital']['auprc'].append(cur['M_lab']['auprc'] - cur['M_vital']['auprc'])
        deltas['M_integrated_vs_vital']['auroc'].append(cur['M_integrated']['auroc'] - cur['M_vital']['auroc'])
        deltas['M_integrated_vs_vital']['auprc'].append(cur['M_integrated']['auprc'] - cur['M_vital']['auprc'])
        
    return metrics, deltas

def resolve_file(candidate_names):
    for name in candidate_names:
        if os.path.exists(name):
            return name
    raise FileNotFoundError(f"None of {candidate_names} found in current directory: {os.getcwd()}")

def main():
    mimic_file = resolve_file(['mimic_landmark.csv', 'mimic_landmark-(4).csv'])
    eicu_file = resolve_file(['eicu_landmark.csv', 'eicu_landmark-(4).csv'])
    
    print(f"Loading MIMIC from: {mimic_file}")
    print(f"Loading eICU from:   {eicu_file}")
    
    mimic = pd.read_csv(mimic_file)
    eicu = pd.read_csv(eicu_file)
    
    features_vitals = ["hr", "rr", "spo2", "map", "temp"]
    features_labs = ["creatinine", "sodium", "potassium", "bun", "glucose", "lactate"]
    
    arms = {
        "M_vital": features_vitals,
        "M_lab": features_labs,
        "M_integrated": features_vitals + features_labs
    }
    
    common_num = ['age'] + (['age_over89'] if 'age_over89' in mimic else [])
    common_cat = ['sex'] if 'sex' in mimic else []
    
    train_mask = mimic.y_24h.notna() & mimic.group_id.notna()
    mim = mimic.loc[train_mask].copy()
    y_mim = mim.y_24h.astype(int).to_numpy()
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2026)
    tr, te = next(splitter.split(mim, y_mim, groups=mim.group_id))
    
    y_te = y_mim[te]
    e_mask = eicu.y_24h.notna() & eicu.group_id.notna()
    e_sub = eicu.loc[e_mask].copy()
    y_e = e_sub.y_24h.astype(int).to_numpy()
    
    probs_mimic = {}
    probs_eicu = {}
    
    print("\nTraining models and predicting...")
    for arm, base_feats in arms.items():
        feat_cols = [f"{name}_{stat}" for name in base_feats for stat in ('mean', 'min', 'max', 'last', 'n')]
        num = common_num + feat_cols
        cat = common_cat
        
        numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())])
        parts = [('numeric', numeric, num)]
        if cat:
            parts.append(('categorical', OneHotEncoder(handle_unknown='ignore'), cat))
        
        pre = ColumnTransformer(parts, remainder='drop')
        model = Pipeline([('preprocess', pre), ('classifier', LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=2026))])
        
        model.fit(mim.iloc[tr], y_mim[tr])
        probs_mimic[arm] = model.predict_proba(mim.iloc[te])[:, 1]
        probs_eicu[arm] = model.predict_proba(e_sub)[:, 1]

    print("\nCalculating Bootstrap 95% CIs and Paired Deltas...")
    m_metrics, m_deltas = bootstrap_ci_and_delta(y_te, probs_mimic)
    e_metrics, e_deltas = bootstrap_ci_and_delta(y_e, probs_eicu)
    
    rows = []
    for cohort_name, y_true, probs, metrics, deltas in [
        ("MIMIC-IV (Internal)", y_te, probs_mimic, m_metrics, m_deltas),
        ("eICU-CRD (External)", y_e, probs_eicu, e_metrics, e_deltas)
    ]:
        for arm in arms:
            auroc_pt = roc_auc_score(y_true, probs[arm])
            auprc_pt = average_precision_score(y_true, probs[arm])
            brier_pt = brier_score_loss(y_true, probs[arm])
            ece_pt = expected_calibration_error(y_true, probs[arm])
            
            auroc_ci = (np.percentile(metrics[arm]['auroc'], 2.5), np.percentile(metrics[arm]['auroc'], 97.5))
            auprc_ci = (np.percentile(metrics[arm]['auprc'], 2.5), np.percentile(metrics[arm]['auprc'], 97.5))
            brier_ci = (np.percentile(metrics[arm]['brier'], 2.5), np.percentile(metrics[arm]['brier'], 97.5))
            
            rows.append({
                "Cohort": cohort_name,
                "Arm": arm,
                "AUROC": f"{auroc_pt:.3f} ({auroc_ci[0]:.3f}-{auroc_ci[1]:.3f})",
                "AUPRC": f"{auprc_pt:.3f} ({auprc_ci[0]:.3f}-{auprc_ci[1]:.3f})",
                "Brier": f"{brier_pt:.4f} ({brier_ci[0]:.4f}-{brier_ci[1]:.4f})",
                "ECE": f"{ece_pt:.4f}"
            })
            
    df_perf = pd.DataFrame(rows)
    print("\n================ MODEL PERFORMANCE & CALIBRATION (UNCERTAINTY-AWARE) ================")
    print(df_perf.to_string(index=False))
    df_perf.to_csv("uncertainty_aware_metrics.csv", index=False)
    
    delta_rows = []
    for cohort_name, deltas in [("MIMIC-IV (Internal)", m_deltas), ("eICU-CRD (External)", e_deltas)]:
        for comp in ['M_lab_vs_vital', 'M_integrated_vs_vital']:
            d_auroc = deltas[comp]['auroc']
            d_auprc = deltas[comp]['auprc']
            delta_rows.append({
                "Cohort": cohort_name,
                "Comparison": comp,
                "Delta_AUROC": f"{np.mean(d_auroc):+.3f} ({np.percentile(d_auroc, 2.5):+.3f} to {np.percentile(d_auroc, 97.5):+.3f})",
                "Delta_AUPRC": f"{np.mean(d_auprc):+.3f} ({np.percentile(d_auprc, 2.5):+.3f} to {np.percentile(d_auprc, 97.5):+.3f})",
                "P_value_AUROC": f"{(np.sum(np.array(d_auroc) <= 0) / len(d_auroc)):.4f}" if np.mean(d_auroc) > 0 else f"{(np.sum(np.array(d_auroc) >= 0) / len(d_auroc)):.4f}"
            })
            
    df_delta = pd.DataFrame(delta_rows)
    print("\n================ INCREMENTAL VALUE ANALYSIS (PAIRED COMPARISON) ================")
    print(df_delta.to_string(index=False))
    df_delta.to_csv("incremental_value_deltas.csv", index=False)
    print("\nResults successfully saved to 'uncertainty_aware_metrics.csv' and 'incremental_value_deltas.csv'.")

if __name__ == '__main__':
    main()
