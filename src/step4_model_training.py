"""
Step 4: Model Training
======================
Trains the Multi-Model Framework matching the paper's architecture:

Local Model Framework (LMF) — 5 independent models:
  - SPM  (Success Probability)      → GradientBoostingRegressor
  - CFSM (Course Fit Score)         → GradientBoostingRegressor
  - PFM  (Prerequisite Fulfillment) → LGBMClassifier
  - GPM  (Graduation Priority)      → LGBMClassifier
  - RLM  (Recommended Load)         → GradientBoostingRegressor

Global Model Framework (GMF):
  - Meta-model (LightGBM) synthesizes all 5 model outputs → final score
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_splits():
    """Load train and validation splits."""
    train = pd.read_pickle(os.path.join(DATA_DIR, "train.pkl"))
    val = pd.read_pickle(os.path.join(DATA_DIR, "val.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.joblib"))
    print(f"Train: {train.shape}, Val: {val.shape}")
    print(f"Features: {feature_cols}")
    return train, val, feature_cols


def train_spm(X_train, y_train, X_val, y_val):
    """Train Success Probability Model (GradientBoostingRegressor)."""
    print("\n  ── Training SPM (Success Probability Model) ──")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"     Time: {elapsed:.1f}s | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}")
    return model


def train_cfsm(X_train, y_train, X_val, y_val):
    """Train Course Fit Score Model (GradientBoostingRegressor)."""
    print("\n  ── Training CFSM (Course Fit Score Model) ──")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"     Time: {elapsed:.1f}s | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}")
    return model


def train_pfm(X_train, y_train, X_val, y_val):
    """Train Prerequisite Fulfillment Model (LGBMClassifier)."""
    print("\n  ── Training PFM (Prerequisite Fulfillment Model) ──")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    print(f"     Time: {elapsed:.1f}s | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    return model


def train_gpm(X_train, y_train, X_val, y_val):
    """Train Graduation Priority Model (LGBMClassifier)."""
    print("\n  ── Training GPM (Graduation Priority Model) ──")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    print(f"     Time: {elapsed:.1f}s | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    return model


def train_rlm(X_train, y_train, X_val, y_val):
    """Train Recommended Load Model (GradientBoostingRegressor)."""
    print("\n  ── Training RLM (Recommended Load Model) ──")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"     Time: {elapsed:.1f}s | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}")
    return model


def generate_meta_features(models, X, feature_cols):
    """Generate predictions from all 5 local models as meta-features."""
    meta = pd.DataFrame()
    meta["spm_pred"] = models["spm"].predict(X[feature_cols])
    meta["cfsm_pred"] = models["cfsm"].predict(X[feature_cols])
    meta["pfm_pred"] = models["pfm"].predict_proba(X[feature_cols])[:, 1]
    meta["gpm_pred"] = models["gpm"].predict_proba(X[feature_cols])[:, 1]
    meta["rlm_pred"] = models["rlm"].predict(X[feature_cols])
    return meta


def train_global_model(meta_train, y_train, meta_val, y_val):
    """
    Train Global Model Framework (GMF).
    LightGBM meta-model that synthesizes all 5 local model outputs.
    Final recommendation score = weighted combination.
    """
    print("\n" + "=" * 60)
    print("  Training Global Model Framework (GMF)")
    print("=" * 60)

    # Create a composite recommendation score as the global target
    # Higher success prob + higher fit + prerequisite met + reasonable load → better recommendation
    y_global_train = (
        0.30 * y_train["success_probability"].values +
        0.25 * y_train["course_fit_score"].values +
        0.20 * y_train["prerequisite_met"].values.astype(float) +
        0.15 * y_train["graduation_priority"].values.astype(float) +
        0.10 * (y_train["recommended_load"].values / 7.0)  # normalized
    )

    y_global_val = (
        0.30 * y_val["success_probability"].values +
        0.25 * y_val["course_fit_score"].values +
        0.20 * y_val["prerequisite_met"].values.astype(float) +
        0.15 * y_val["graduation_priority"].values.astype(float) +
        0.10 * (y_val["recommended_load"].values / 7.0)
    )

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )

    t0 = time.time()
    model.fit(
        meta_train, y_global_train,
        eval_set=[(meta_val, y_global_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    elapsed = time.time() - t0

    train_pred = model.predict(meta_train)
    val_pred = model.predict(meta_val)
    train_rmse = np.sqrt(mean_squared_error(y_global_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_global_val, val_pred))

    print(f"  Time: {elapsed:.1f}s")
    print(f"  GMF Train RMSE: {train_rmse:.5f}")
    print(f"  GMF Val RMSE:   {val_rmse:.5f}")
    print(f"  Best iteration: {model.best_iteration_}")

    return model, y_global_train, y_global_val


def main():
    print("\n" + "=" * 60)
    print("  STEP 4: MODEL TRAINING")
    print("  Multi-Model ML Framework (LMF + GMF)")
    print("=" * 60 + "\n")

    # ── Load data ────────────────────────────────────────────────
    train, val, feature_cols = load_splits()

    X_train = train[feature_cols]
    X_val = val[feature_cols]

    target_cols = {
        "spm": "success_probability",
        "cfsm": "course_fit_score",
        "pfm": "prerequisite_met",
        "gpm": "graduation_priority",
        "rlm": "recommended_load"
    }

    # ── Train Local Model Framework (LMF) ────────────────────────
    print("=" * 60)
    print("  LOCAL MODEL FRAMEWORK (LMF) — 5 Models")
    print("=" * 60)

    models = {}
    total_start = time.time()

    # SPM
    models["spm"] = train_spm(
        X_train, train[target_cols["spm"]],
        X_val, val[target_cols["spm"]]
    )

    # CFSM
    models["cfsm"] = train_cfsm(
        X_train, train[target_cols["cfsm"]],
        X_val, val[target_cols["cfsm"]]
    )

    # PFM
    models["pfm"] = train_pfm(
        X_train, train[target_cols["pfm"]],
        X_val, val[target_cols["pfm"]]
    )

    # GPM
    models["gpm"] = train_gpm(
        X_train, train[target_cols["gpm"]],
        X_val, val[target_cols["gpm"]]
    )

    # RLM
    models["rlm"] = train_rlm(
        X_train, train[target_cols["rlm"]],
        X_val, val[target_cols["rlm"]]
    )

    lmf_time = time.time() - total_start
    print(f"\n  LMF Total Training Time: {lmf_time:.1f}s")

    # ── Generate meta-features ───────────────────────────────────
    print("\nGenerating meta-features from LMF outputs...")
    meta_train = generate_meta_features(models, train, feature_cols)
    meta_val = generate_meta_features(models, val, feature_cols)
    print(f"  Meta-feature columns: {list(meta_train.columns)}")

    # ── Train Global Model Framework (GMF) ───────────────────────
    y_targets_train = train[list(target_cols.values())]
    y_targets_val = val[list(target_cols.values())]

    gmf_model, y_global_train, y_global_val = train_global_model(
        meta_train, y_targets_train,
        meta_val, y_targets_val
    )

    # ── Save all models ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving models...")
    print("=" * 60)

    for name, model in models.items():
        path = os.path.join(MODEL_DIR, f"{name}_model.joblib")
        joblib.dump(model, path)
        print(f"  Saved: {path}")

    gmf_path = os.path.join(MODEL_DIR, "gmf_model.joblib")
    joblib.dump(gmf_model, gmf_path)
    print(f"  Saved: {gmf_path}")

    # Save meta-features and global targets for evaluation
    joblib.dump(meta_train, os.path.join(ARTIFACT_DIR, "meta_train.joblib"))
    joblib.dump(meta_val, os.path.join(ARTIFACT_DIR, "meta_val.joblib"))
    joblib.dump(y_global_train, os.path.join(ARTIFACT_DIR, "y_global_train.joblib"))
    joblib.dump(y_global_val, os.path.join(ARTIFACT_DIR, "y_global_val.joblib"))
    joblib.dump(target_cols, os.path.join(ARTIFACT_DIR, "target_cols.joblib"))

    # ── Feature Importance ───────────────────────────────────────
    print("\n── GMF Feature Importance ──")
    importance = pd.Series(
        gmf_model.feature_importances_,
        index=meta_train.columns
    ).sort_values(ascending=False)
    for feat, imp in importance.items():
        print(f"  {feat:15s}: {imp:>6d}")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  STEP 4 COMPLETE | Total time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
