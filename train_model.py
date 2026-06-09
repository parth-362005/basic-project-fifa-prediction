"""
train_model.py
==============
Trains Logistic Regression, Random Forest, and XGBoost classifiers on
preprocessed match data, evaluates them, and saves the best model to disk.

Usage:
    python train_model.py
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier

# Run preprocessing first if the processed file doesn't exist
from preprocess import run_pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROCESSED_PATH = "data/processed_matches.csv"
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH      = os.path.join(MODEL_DIR, "model_meta.pkl")

# Features used for training
FEATURES = ["rank_diff", "points_diff", "is_neutral"]
TARGET   = "outcome"

# Outcome labels
LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_processed_data():
    """Load (or create) the preprocessed match dataset."""
    if not os.path.exists(PROCESSED_PATH):
        print("[INFO] Processed data not found. Running preprocessing pipeline…")
        run_pipeline()
    df = pd.read_csv(PROCESSED_PATH)
    print(f"[INFO] Loaded processed data: {len(df):,} rows.")
    return df


# ─────────────────────────────────────────────
# 2. PREPARE FEATURES / LABELS
# ─────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    """Return feature matrix X and label vector y."""
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(int)
    return X, y


# ─────────────────────────────────────────────
# 3. BUILD MODELS
# ─────────────────────────────────────────────

def build_models():
    """
    Returns a dict of {model_name: model_instance}.
    All models use fixed random_state=42 for reproducibility.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        ),
    }
    return models


# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train all models, print evaluation metrics, and return:
      - best_model_name (str)
      - best_model (fitted estimator)
      - results dict {name: accuracy}
    """
    models  = build_models()
    results = {}

    print("\n" + "═" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("═" * 60)

    for name, model in models.items():
        print(f"\n▶  Training: {name}")

        # 5-fold CV on training set
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring="accuracy")
        print(f"   CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        test_acc  = accuracy_score(y_test, y_pred)
        print(f"   Test Accuracy: {test_acc:.4f}")

        print("\n   Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[LABELS[i] for i in sorted(LABELS)],
            zero_division=0
        ))

        results[name] = test_acc

    # ── Pick the best model ──
    best_name  = max(results, key=results.get)
    best_model = models[best_name]

    print("═" * 60)
    print(f"\n🏆  Best model : {best_name}  (Test Acc = {results[best_name]:.4f})")
    print("═" * 60 + "\n")

    return best_name, best_model, results


# ─────────────────────────────────────────────
# 5. SAVE ARTEFACTS
# ─────────────────────────────────────────────

def save_artefacts(model, scaler, best_name: str, results: dict):
    """Persist model, scaler, and metadata to the models/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "best_model_name": best_name,
        "features":        FEATURES,
        "labels":          LABELS,
        "results":         results,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"[INFO] Model saved  → {MODEL_PATH}")
    print(f"[INFO] Scaler saved → {SCALER_PATH}")
    print(f"[INFO] Meta  saved  → {META_PATH}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    # Load data
    df = load_processed_data()
    X, y = prepare_xy(df)

    # Train / test split (80 / 20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # Scale features (important for Logistic Regression)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train & evaluate
    best_name, best_model, results = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Save
    save_artefacts(best_model, scaler, best_name, results)
    print("\n✅  Training complete! Run `streamlit run app.py` to launch the app.")


if __name__ == "__main__":
    main()
