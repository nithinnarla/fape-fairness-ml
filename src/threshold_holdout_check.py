"""
FAPE, threshold-fitting check for random forest (Section 6.4 of the paper)

Every intervention script hands ThresholdOptimizer an estimator and the training
split, so it refits the model and chooses its group thresholds on the same
records the model was trained on. Logistic regression and gradient boosting
score their training records much as they score new ones. A random forest
fits its training records almost perfectly, so the scores the thresholds are
chosen from look nothing like the scores it gives the test split.

This script runs two designs side by side for Student (Math) and MEPS, the two
evaluations where random forest worsens under a constraint and small groups
do not explain it:
(a) the design the intervention scripts use: thresholds chosen on the training split
(b) held out: the model trains on 75% of the training split and the
    thresholds are chosen (prefit=True) on the other 25%

Both designs score the same test split. The intervention results are not changed; (a)
reproduces them. Each model's accuracy on its own training records is printed
first, since that is what separates random forest from the other two.
"""

import os
import sys
import ast
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}
CONSTRAINTS = {
    'demographic_parity': ('DPD', demographic_parity_difference),
    'equalized_odds': ('EOD', equalized_odds_difference),
}
HOLDOUT_SHARE = 0.25


def student_math():
    """The intervention script's Student setup: its own prepare_student, sex as the attribute."""
    path = os.path.join(SRC_DIR, 'stage2_student_threshold.py')
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_student'][0]
    namespace = {'pd': pd, 'np': np, 'LabelEncoder': LabelEncoder}
    exec(compile(ast.Module([fn], []), path, 'exec'), namespace)
    from student_loader import load_student_performance
    with contextlib.redirect_stdout(io.StringIO()):
        subjects = load_student_performance()
    math_key = [k for k in subjects if k.lower().startswith('mat')][0]
    X, y, sex = namespace['prepare_student'](subjects[math_key], 'sex')
    return np.asarray(X), np.asarray(y), sex.reset_index(drop=True)


def meps():
    """The intervention script's FairGround setup for MEPS: documented features, RACE dropped from X."""
    from fairml_datasets import Dataset
    from fairground_loader import documented_feature_columns
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dataset = Dataset.from_id('meps_panel_19_fy2015')
        df = dataset.load()
    target = dataset.get_target_column()
    cols = [c for c in documented_feature_columns('meps_panel_19_fy2015', df, target) if c != 'RACE']
    X = df[cols].copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int).values
    return X.fillna(0).values, y, df['RACE'].astype(str)


def check(domain, setup):
    X, y, groups = setup()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    g_train = groups.iloc[idx_train].reset_index(drop=True)
    g_test = groups.iloc[idx_test].reset_index(drop=True)
    fit_idx, hold_idx = train_test_split(np.arange(len(y_train)), test_size=HOLDOUT_SHARE,
                                         random_state=42, stratify=y_train)
    scaler = StandardScaler().fit(X_train)

    print(f"\n=== {domain} (test split {len(y_test)}, threshold hold-out {len(hold_idx)}) ===")
    for name, estimator in MODELS.items():
        tr, te = (scaler.transform(X_train), scaler.transform(X_test)) if name == 'LogisticRegression' \
            else (X_train, X_test)
        full = clone(estimator).fit(tr, y_train)
        part = clone(estimator).fit(tr[fit_idx], y_train[fit_idx])
        print(f"  {name:<19} training accuracy {full.score(tr, y_train):.3f}")
        for constraint, (label, metric) in CONSTRAINTS.items():
            stage2 = ThresholdOptimizer(estimator=clone(estimator), constraints=constraint,
                                        predict_method='auto', objective='balanced_accuracy_score')
            stage2.fit(tr, y_train, sensitive_features=g_train)
            held = ThresholdOptimizer(estimator=part, constraints=constraint, prefit=True,
                                      predict_method='auto', objective='balanced_accuracy_score')
            held.fit(tr[hold_idx], y_train[hold_idx], sensitive_features=g_train.iloc[hold_idx])
            a0 = metric(y_test, full.predict(te), sensitive_features=g_test)
            a1 = metric(y_test, stage2.predict(te, sensitive_features=g_test, random_state=42),
                        sensitive_features=g_test)
            b0 = metric(y_test, part.predict(te), sensitive_features=g_test)
            b1 = metric(y_test, held.predict(te, sensitive_features=g_test, random_state=42),
                        sensitive_features=g_test)
            print(f"  {name:<19} {label}  (a) training-split thresholds {a0:.3f} -> {a1:.3f}"
                  f"   (b) held-out thresholds {b0:.3f} -> {b1:.3f}")


if __name__ == '__main__':
    check('Student (Math), sex', student_math)
    check('MEPS, race', meps)
