"""Train-only chronological baseline comparison. Run from any directory."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from features import augment_features, MOMENTUM_FEATURES_TARGET_1, OPTIMIZED_FEATURES_TARGET_1

ROOT = Path(__file__).resolve().parent
TARGETS = ["Target_1", "Target_2"]


def make_model(name, features, numerical, categorical, parameters=None):
    parameters = parameters or {}
    if name == "rf":
        preprocess = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        model = make_pipeline(preprocess, RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=4))
    elif name == "catboost":
        settings = dict(iterations=300, depth=4, learning_rate=0.03,
            loss_function="Logloss", random_seed=42, thread_count=4,
            verbose=False, allow_writing_files=False)
        settings.update(parameters)
        model = CatBoostClassifier(cat_features=categorical, **settings)
    elif name == "catboost_opt":
        settings = dict(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=10,
            loss_function="Logloss", random_seed=42, thread_count=4,
            verbose=False, allow_writing_files=False)
        settings.update(parameters)
        model = CatBoostClassifier(cat_features=categorical, **settings)
    elif name == "momentum":
        preprocess = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  RobustScaler()), [c for c in features if c in numerical]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             [c for c in features if c in categorical]),
        ])
        settings = dict(C=0.03, max_iter=1000, random_state=42)
        settings.update(parameters)
        model = make_pipeline(preprocess, LogisticRegression(**settings))
    elif name == "optimized":
        preprocess = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  RobustScaler()), [c for c in features if c in numerical]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             [c for c in features if c in categorical]),
        ])
        settings = dict(C=0.02, max_iter=1000, random_state=42)
        settings.update(parameters)
        model = make_pipeline(preprocess, LogisticRegression(**settings))
    else:
        preprocess = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler()), [c for c in features if c in numerical]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             [c for c in features if c in categorical]),
        ])
        settings = dict(C=0.1, max_iter=1000, random_state=42)
        settings.update(parameters)
        model = make_pipeline(preprocess, LogisticRegression(**settings))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rf", "catboost", "conservative", "momentum", "optimized"], default="rf")
    args = parser.parse_args()
    start = time.monotonic()
    train_path = ROOT / "train.csv"
    df = pd.read_csv(train_path, encoding="cp949").sort_values("Time_step")
    assert len(df) == 924 and df.Time_step.is_unique
    assert df.Time_step.diff().dropna().eq(1).all()
    assert all(set(df[t].unique()) == {0, 1} for t in TARGETS)
    df = augment_features(df)
    X = df.drop(columns=["Time_step", *TARGETS]).copy()
    # Weekday is a category even when stored as an integer.
    X["Weekday"] = X["Weekday"].astype(str)
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical = [c for c in X if c not in categorical]
    for c in categorical:
        X[c] = X[c].fillna("Unknown").astype(str)
    X[numerical] = X[numerical].replace([np.inf, -np.inf], np.nan)
    feature_sets = {t: list(X.columns) for t in TARGETS}
    if args.model == "conservative":
        feature_sets = {
            "Target_1": [c for c in X if c.startswith("FI_") and not "_d" in c or c == "Weekday"],
            "Target_2": ["Weekday"],
        }
    elif args.model == "momentum":
        feature_sets = {
            "Target_1": MOMENTUM_FEATURES_TARGET_1,
            "Target_2": [c for c in X if not "_d" in c and not "Spread_" in c and not "Ratio_" in c],
        }
    elif args.model == "optimized":
        feature_sets = {
            "Target_1": OPTIMIZED_FEATURES_TARGET_1,
            "Target_2": [c for c in X if not "_d" in c and not "Spread_" in c and not "Ratio_" in c],
        }
    scores = []
    # Provisional assumption: D+3 means three rows. Confirm with organizers.
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=3, test_size=150, gap=3).split(X), 1):
        assert tr[-1] + 3 < va[0]
        result = {"fold": fold, "train_end": int(df.iloc[tr[-1]].Time_step),
                  "valid_start": int(df.iloc[va[0]].Time_step),
                  "valid_end": int(df.iloc[va[-1]].Time_step)}
        for target in TARGETS:
            assert df.iloc[tr][target].nunique() == df.iloc[va][target].nunique() == 2
            features = feature_sets[target]
            factory_name = args.model
            if args.model == "momentum":
                factory_name = "momentum" if target == "Target_1" else "catboost"
            elif args.model == "optimized":
                factory_name = "optimized" if target == "Target_1" else "catboost_opt"
            model = make_model(factory_name, features, numerical, categorical)
            model.fit(X.iloc[tr][features], df.iloc[tr][target])
            positive = list(model.classes_).index(1)
            probability = model.predict_proba(X.iloc[va][features])[:, positive]
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            result[target] = float(roc_auc_score(df.iloc[va][target], probability))
            train_probability = model.predict_proba(X.iloc[tr][features])[:, positive]
            result[f"{target}_train_auc"] = float(roc_auc_score(df.iloc[tr][target], train_probability))
        result["mean_auc"] = float(np.mean([result[t] for t in TARGETS]))
        scores.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"model": args.model, "seed": 42, "gap": 3, "validation_size": 150,
              "features": feature_sets, "folds": scores,
              "conservative_parameters": {"C": 0.1, "max_iter": 1000}
                  if args.model == "conservative" else None,
              "momentum_parameters": {"C": 0.03, "max_iter": 1000, "scaler": "RobustScaler"}
                  if args.model == "momentum" else None,
              "optimized_parameters": {"C": 0.02, "max_iter": 1000, "scaler": "RobustScaler"}
                  if args.model == "optimized" else None,
              "mean_auc": float(np.mean([s["mean_auc"] for s in scores])),
              "elapsed_seconds": round(time.monotonic() - start, 2),
              "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ["pandas", "scikit-learn", "catboost", "optuna"]}}
    output = ROOT / "experiments" / f"{args.model}_baseline.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Mean AUC: {report['mean_auc']:.4f}; saved to {output}")


if __name__ == "__main__":
    main()
