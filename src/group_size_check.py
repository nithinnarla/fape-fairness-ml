"""
FAPE, group-size check for COMPAS and Folktables (Section 6.2 of the paper)

Demographic parity difference is the gap between the most and least favored
group, and neither Stage 2 script applies a minimum group size when computing
it. Groups with a handful of test records have selection rates that move in
large steps, so they can set the reported gap on their own:
- COMPAS test split: 7 Asian defendants, 1 Native American defendant
- Folktables test split: 5 Alaska Native and 25 American Indian and Alaska
  Native respondents

For each domain this script reproduces the Stage 2 setup exactly (loader,
sampling, split, models, scaling, ThresholdOptimizer settings, random_state)
and prints, per model, the selection rate of every race group before and
after the demographic parity constraint, the reported gap over all groups,
and the same gap over groups with at least MIN_GROUP_N test records; and the
equalized odds difference before and after the equalized odds constraint,
over all groups and over those same larger groups.

Nothing here changes a reported result; it shows what drives the values.
"""

import os
import sys
import ast
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

MIN_GROUP_N = 30   # the floor stage2_folktables_threshold.py already uses for its printed rates


def _script_tree(name):
    path = os.path.join(SRC_DIR, name)
    return path, ast.parse(open(path).read())


def compas_setup():
    path, tree = _script_tree('stage2_compas_threshold.py')
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess'][0]
    namespace = {'np': np, 'pd': pd, 'LabelEncoder': LabelEncoder, 'StandardScaler': StandardScaler}
    exec(compile(ast.Module([fn], []), path, 'exec'), namespace)
    from data_loader import load_compas
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, race, _sex = namespace['preprocess'](load_compas())
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    return np.asarray(X), np.asarray(y), race, models, None


def folktables_setup():
    path, tree = _script_tree('stage2_folktables_threshold.py')
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in ('FEATURE_COLS', 'SAMPLE_SIZE', 'RAC1P_LABELS'):
            consts[n.targets[0].id] = ast.literal_eval(n.value)
    from data_loader import load_folktables_acs
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_folktables_acs()
    sample = df.sample(consts['SAMPLE_SIZE'], random_state=42).reset_index(drop=True)
    X = sample[consts['FEATURE_COLS']].values
    y = sample['label'].values
    race = sample['RAC1P'].map(lambda c: consts['RAC1P_LABELS'].get(int(c), str(c)))
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    return X, y, race, models, None


def check(domain, setup):
    X, y, race, models, _ = setup()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    race_train = race.iloc[idx_train].reset_index(drop=True)
    race_test = race.iloc[idx_test].reset_index(drop=True)

    counts = race_test.value_counts()
    groups = list(counts.index)
    large = [g for g in groups if counts[g] >= MIN_GROUP_N]
    print(f"\n=== {domain} ===")
    print("Test split, records per race group:")
    for g in groups:
        flag = '' if counts[g] >= MIN_GROUP_N else f'   below {MIN_GROUP_N}'
        print(f"  {str(g):<20} {counts[g]:>6}{flag}")
    print(f"  groups with at least {MIN_GROUP_N} records hold {counts[large].sum()} of {len(race_test)}")

    for name, model in models.items():
        X_tr, X_te = (X_train_sc, X_test_sc) if name == 'LogisticRegression' else (X_train, X_test)
        model.fit(X_tr, y_train)
        y_base = model.predict(X_te)
        to = ThresholdOptimizer(estimator=model, constraints="demographic_parity",
                                predict_method="auto", objective="balanced_accuracy_score")
        to.fit(X_tr, y_train, sensitive_features=race_train)
        y_dp = to.predict(X_te, sensitive_features=race_test, random_state=42)

        rates_base = {g: y_base[(race_test == g).values].mean() for g in groups}
        rates_dp = {g: y_dp[(race_test == g).values].mean() for g in groups}
        span = lambda rates, keep: max(rates[g] for g in keep) - min(rates[g] for g in keep)

        print(f"\n{name}")
        print(f"  {'group':<20} {'n':>6}  {'sel base':>8} {'sel DP':>7}")
        for g in groups:
            print(f"  {str(g):<20} {counts[g]:>6}  {rates_base[g]:8.3f} {rates_dp[g]:7.3f}")
        print(f"  DPD, all groups:            baseline "
              f"{demographic_parity_difference(y_test, y_base, sensitive_features=race_test):.3f}  post-DP "
              f"{demographic_parity_difference(y_test, y_dp, sensitive_features=race_test):.3f}")
        print(f"  DPD, groups n>={MIN_GROUP_N}:         baseline {span(rates_base, large):.3f}  "
              f"post-DP {span(rates_dp, large):.3f}")
        to_eo = ThresholdOptimizer(estimator=model, constraints="equalized_odds",
                                   predict_method="auto", objective="balanced_accuracy_score")
        to_eo.fit(X_tr, y_train, sensitive_features=race_train)
        y_eo = to_eo.predict(X_te, sensitive_features=race_test, random_state=42)
        large_mask = race_test.isin(large).values
        print(f"  EOD, all groups:            baseline "
              f"{equalized_odds_difference(y_test, y_base, sensitive_features=race_test):.3f}  post-EO "
              f"{equalized_odds_difference(y_test, y_eo, sensitive_features=race_test):.3f}")
        print(f"  EOD, groups n>={MIN_GROUP_N}:         baseline "
              f"{equalized_odds_difference(y_test[large_mask], y_base[large_mask], sensitive_features=race_test[large_mask]):.3f}  post-EO "
              f"{equalized_odds_difference(y_test[large_mask], y_eo[large_mask], sensitive_features=race_test[large_mask]):.3f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ['COMPAS', 'Folktables']
    setups = {'COMPAS': compas_setup, 'Folktables': folktables_setup}
    for domain in which:
        check(domain, setups[domain])
