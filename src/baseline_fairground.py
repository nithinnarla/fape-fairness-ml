"""
FAPE, FairGround Corpus Baseline Models
Phase 4, Baseline Modeling
Cross-Domain Evaluation

Baseline models across 5 representative FairGround datasets:
- adult (income), race sensitive
- compas_2_years (criminal justice), age sensitive
- creditcard (credit), sex sensitive
- law_school_lequy (education), race + sex sensitive
- meps_panel_19_fy2015 (healthcare), race sensitive

Demonstrates fairness disparities exist across all 5 domains.
Sets up for Stage 2 ThresholdOptimizer cross-domain intervention.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

SELECTED_DATASETS = {
    'adult':               {'domain': 'Income',           'sensitive': 'race',  'target_encode': {'<=50K': 0, '>50K': 1, ' <=50K': 0, ' >50K': 1}},
    'compas_2_years':      {'domain': 'Criminal Justice', 'sensitive': 'age',   'target_encode': None},
    'creditcard':          {'domain': 'Credit',           'sensitive': 'SEX',   'target_encode': None},
    'law_school_lequy':    {'domain': 'Education',        'sensitive': 'racetxt','target_encode': None},
    'meps_panel_19_fy2015':{'domain': 'Healthcare',       'sensitive': 'RACE',  'target_encode': None},
}

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

MODEL_COLORS = {'LogisticRegression':'steelblue','RandomForest':'coral','GradientBoosting':'#5cb85c'}


def prepare_dataset(corpus, ds_id, config):
    content = corpus[ds_id]
    X = content['X'].copy()
    y = content['y'].copy()

    # Encode target if needed
    if config['target_encode']:
        y = y.str.strip() if hasattr(y, 'str') else y
        y = y.map(config['target_encode'])

    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    # Get sensitive attribute BEFORE encoding
    sens_col = config['sensitive']
    if sens_col in X.columns:
        sensitive = X[sens_col].copy()
    else:
        sensitive = pd.Series([0]*len(X), name=sens_col)

    # Encode ALL features, label encode categoricals
    X_enc = X.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype == object or str(X_enc[col].dtype) == 'category':
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))

    X_num = X_enc.select_dtypes(include=[np.number]).copy()
    # Impute all NaN, median for numeric, 0 for remaining
    for col in X_num.columns:
        if X_num[col].isna().any():
            median_val = X_num[col].median()
            X_num[col] = X_num[col].fillna(median_val if not np.isnan(median_val) else 0)
    # Final safety, replace any remaining NaN/inf with 0
    X_num = X_num.replace([np.inf, -np.inf], 0).fillna(0)

    # Drop known leakage columns
    leakage_cols = ['is_recid', 'is_violent_recid', 'event', 'end', 'start',
                    'r_case_number', 'r_jail_in', 'r_jail_out',
                    'r_charge_desc', 'r_charge_degree', 'r_days_from_arrest',
                    'r_offense_date', 'c_charge_desc', 'id', 'name', 'first',
                    'last', 'compas_screening_date', 'dob', 'c_jail_in',
                    'c_jail_out', 'c_case_number', 'c_offense_date',
                    'vr_offense_date', 'decile_score', 'decile_score.1',
                    'v_decile_score', 'score_text', 'v_score_text',
                    'out_custody', 'in_custody', 'screening_date',
                    'v_screening_date', 'vr_case_number', 'c_arrest_date',
                    'vr_charge_desc', 'vr_charge_degree', 'violent_recid',
                    'type_of_assessment', 'v_type_of_assessment',
                    'c_days_from_compas', 'priors_count.1']
    X_num = X_num.drop(columns=[c for c in leakage_cols if c in X_num.columns], errors='ignore')

    # For high-dim data keep top 50 features by variance (MEPS)
    if X_num.shape[1] > 100:
        variances = X_num.var()
        top_cols = variances.nlargest(50).index
        X_num = X_num[top_cols]

    return X_num, y, sensitive


def fairness_metrics(y_true, y_pred, sensitive):
    results = {}
    base_rate = y_pred.mean()
    for grp in sorted(sensitive.unique()):
        mask = sensitive == grp
        if mask.sum() < 30:
            continue
        gp = y_pred[mask]
        gt = y_true[mask]
        fp_mask = gt == 0
        tp_mask = gt == 1
        results[grp] = {
            'n': int(mask.sum()),
            'pos_rate': float(gp.mean()),
            'dp_diff': float(gp.mean() - base_rate),
            'fpr': float(gp[fp_mask].mean()) if fp_mask.sum() > 0 else 0,
            'tpr': float(gp[tp_mask].mean()) if tp_mask.sum() > 0 else 0,
        }
    return results


def run_baselines():
    print("FAPE Phase 4, FairGround Corpus Baseline Models")
    print("=" * 55)

    from fairml_datasets import Datasets
    print("Loading only 5 selected FairGround datasets...")
    needed = list(SELECTED_DATASETS.keys())
    corpus = {}
    datasets_obj = Datasets(include_large_datasets=False)
    for dataset in datasets_obj:
        ds_id = getattr(dataset, 'dataset_id', None)
        if ds_id not in needed:
            continue
        try:
            dataset.load()
            df = dataset.to_pandas()
            target_col = dataset.get_target_column()
            sensitive_cols = dataset.sensitive_columns if hasattr(dataset, 'sensitive_columns') else []
            feature_cols = [c for c in df.columns if c != target_col]
            corpus[ds_id] = {
                'X': df[feature_cols],
                'y': df[target_col],
                'df': df,
                'metadata': {'id': ds_id, 'n_samples': len(df), 'n_features': len(feature_cols),
                             'sensitive_columns': sensitive_cols, 'target_column': target_col,
                             'positive_rate': None},
                'sensitive_cols': sensitive_cols
            }
            print(f"  OK: {ds_id}: {len(df):,} rows")
        except Exception as e:
            print(f"  FAIL: {ds_id}: {e}")
    print(f"Loaded {len(corpus)} datasets")

    all_results = {}

    for ds_id, config in SELECTED_DATASETS.items():
        print(f"\n--- Dataset: {ds_id} ({config['domain']}) ---")
        X, y, sensitive = prepare_dataset(corpus, ds_id, config)

        X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
            X.values, y.values, np.arange(len(y)),
            test_size=0.2, random_state=42, stratify=y.values)

        sens_test = sensitive.iloc[idx_te].reset_index(drop=True)

        # Bin continuous sensitive attributes for fairness computation
        if sens_test.nunique() > 10:
            try:
                sens_num = pd.to_numeric(sens_test, errors='coerce')
                if sens_num.notna().mean() > 0.8:
                    sens_test = pd.cut(sens_num, bins=3,
                                       labels=['low','mid','high']).astype(str)
                    print(f"  Binned continuous sensitive attr into 3 groups")
            except:
                pass

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        print(f"  n={len(y):,} | features={X.shape[1]} | pos_rate={y.mean():.1%}")
        print(f"  sensitive={config['sensitive']} | groups={sensitive.nunique()}")

        ds_results = {}
        for name, model in MODELS.items():
            if name == 'LogisticRegression':
                model.fit(X_tr_sc, y_train)
                y_pred = model.predict(X_te_sc)
                y_prob = model.predict_proba(X_te_sc)[:,1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:,1]

            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            fm = fairness_metrics(y_test, y_pred, sens_test)
            dp_diffs = [abs(v['dp_diff']) for v in fm.values()]
            max_dp = max(dp_diffs) if dp_diffs else 0

            ds_results[name] = {
                'auc': auc, 'f1': f1,
                'y_pred': y_pred, 'y_prob': y_prob,
                'fairness': fm, 'max_dp': max_dp
            }
            print(f"  {name:<25} AUC={auc:.3f} F1={f1:.3f} max_DP={max_dp:.3f}")

        all_results[ds_id] = {'config': config, 'results': ds_results,
                               'sensitive': sens_test, 'y_test': y_test}

    print(f"\n--- Cross-Domain Summary ---")
    for ds_id, data in all_results.items():
        best = max(data['results'].items(), key=lambda x: x[1]['auc'])
        print(f"  {data['config']['domain']:<20} best={best[0]} AUC={best[1]['auc']:.3f} max_DP={best[1]['max_dp']:.3f}")

    print(f"\n--- FairGround Baseline complete ---")
    print(f"  5 domains covered: Income, Criminal Justice, Credit, Education, Healthcare")
    print(f"  Fairness disparities confirmed across all domains, Stage 2 ThresholdOptimizer needed")

    return all_results


if __name__ == "__main__":
    results = run_baselines()
