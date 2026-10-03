"""Core logic: data loading, model training/saving (joblib), input validation and prediction helpers."""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "breast_cancer_wisconsin.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")

FEATURES = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
            "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
            "mean_fractal_dimension"]

FEATURE_INFO = {
    "mean_radius": "Mean distance from the centre to points on the nucleus boundary",
    "mean_texture": "Standard deviation of gray-scale values (texture)",
    "mean_perimeter": "Mean size of the nucleus perimeter",
    "mean_area": "Mean area of the nucleus",
    "mean_smoothness": "Local variation in radius lengths",
    "mean_compactness": "Perimeter² / area − 1.0",
    "mean_concavity": "Severity of concave portions of the contour",
    "mean_concave_points": "Number of concave portions of the contour",
    "mean_symmetry": "Symmetry of the nucleus",
    "mean_fractal_dimension": "'Coastline approximation' − 1",
}


def pretty(name):
    return name.replace("_", " ").capitalize()


# ----------------------------------------------------------------------------- data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df[FEATURES], (df["diagnosis"] == "M").astype(int)      # 1 = malignant, 0 = benign


def area_ratio(radius, area):
    """area / (pi * r^2): close to 1 for a roughly circular nucleus; used for a consistency check."""
    return area / (math.pi * radius ** 2)


# ----------------------------------------------------------------------------- training
def train_and_save(save=True):
    """Split first, compare 3 models with cross-validation on the training set only, fit the best, save it."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    def pipe(clf):                       # scaler lives inside the pipeline -> no leakage
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    candidates = {
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for name, p in candidates.items():
        s = cross_validate(p, X_train, y_train, cv=cv, scoring=["roc_auc", "accuracy"])
        comparison[name] = {"cv_roc_auc": float(s["test_roc_auc"].mean()),
                            "cv_accuracy": float(s["test_accuracy"].mean())}
    best_name = max(comparison, key=lambda k: comparison[k]["cv_roc_auc"])
    # Parsimony rule: if the interpretable linear model is within 0.005 ROC-AUC of the best, prefer it
    if comparison["Logistic Regression"]["cv_roc_auc"] >= comparison[best_name]["cv_roc_auc"] - 0.005:
        best_name = "Logistic Regression"
    best = candidates[best_name].fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ratio = area_ratio(X_train["mean_radius"], X_train["mean_area"])

    meta = {
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "comparison": comparison,
        "test_metrics": {"accuracy": float(accuracy_score(y_test, pred)),
                         "precision": float(precision_score(y_test, pred)),
                         "recall": float(recall_score(y_test, pred)),
                         "f1": float(f1_score(y_test, pred)),
                         "roc_auc": float(roc_auc_score(y_test, proba))},
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),     # rows: actual [B, M]
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "ranges": {f: {"min": float(X_train[f].min()), "max": float(X_train[f].max()),
                       "median": float(X_train[f].median())} for f in FEATURES},
        "class_medians": {"benign": X_train[y_train == 0].median().to_dict(),
                          "malignant": X_train[y_train == 1].median().to_dict()},
        "area_ratio_bounds": [float(ratio.min() * 0.9), float(ratio.max() * 1.1)],
    }
    if save:
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(best, MODEL_PATH, compress=3)
            with open(META_PATH, "w") as f:
                json.dump(meta, f, indent=2)
        except OSError:
            pass                                    # read-only disk: keep the in-memory model
    return best, meta


def load_artifacts():
    """Load the saved model with joblib. If it is missing/unreadable/built with another scikit-learn
    version, retrain automatically so the app never crashes. Returns (model, meta, source)."""
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        if meta.get("sklearn_version") == sklearn.__version__:
            return model, meta, "loaded from model/model.joblib"
    except Exception:
        pass
    model, meta = train_and_save(save=True)
    return model, meta, "re-trained at start-up (saved model not usable)"


# ----------------------------------------------------------------------------- validation
def validate_values(values, meta):
    """Validate one set of inputs. Returns (errors, warnings)."""
    errors, warnings = [], []
    for f in FEATURES:
        v, r = values.get(f), meta["ranges"][f]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            errors.append(f"{pretty(f)} is missing.")
        elif v <= 0:
            errors.append(f"{pretty(f)} must be greater than 0.")
        elif v > 2.5 * r["max"]:
            errors.append(f"{pretty(f)} = {v:g} is implausibly large (training maximum is {r['max']:.4g}).")
        elif v < r["min"] or v > r["max"]:
            warnings.append(f"{pretty(f)} = {v:g} is outside the training range "
                            f"({r['min']:.4g} to {r['max']:.4g}); the prediction may be unreliable.")
    if not errors:
        lo, hi = meta["area_ratio_bounds"]
        ratio = area_ratio(values["mean_radius"], values["mean_area"])
        if not lo <= ratio <= hi:
            warnings.append("Mean radius and mean area look inconsistent with each other "
                            f"(area / (π·r²) = {ratio:.2f}, usually between {lo:.2f} and {hi:.2f}).")
    return errors, warnings


def prepare_batch(df, meta):
    """Validate an uploaded table. Returns (valid_df, problems) where problems is a list of messages."""
    problems = []
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        return None, [f"Missing required column(s): {', '.join(missing_cols)}."]
    data = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = data.isna().any(axis=1) | (data <= 0).any(axis=1)
    for idx in data.index[bad][:10]:
        problems.append(f"Row {idx + 2}: contains an empty, non-numeric or non-positive value (skipped).")
    if bad.sum() > 10:
        problems.append(f"... and {int(bad.sum()) - 10} more invalid rows (skipped).")
    valid = data[~bad]
    if valid.empty:
        problems.append("No valid rows were found.")
        return None, problems
    return valid, problems


# ----------------------------------------------------------------------------- prediction
def predict_proba_malignant(model, X):
    return model.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]


def explain(model, values):
    """Per-feature contribution (log-odds) for linear models, otherwise global importances."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        scaled = model.named_steps["scaler"].transform(pd.DataFrame([values], columns=FEATURES))[0]
        return pd.Series(clf.coef_[0] * scaled, index=FEATURES), "local"
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=FEATURES), "global"
    return None, None
