"""
Step 5: Model Evaluation (Validation Set)
==========================================
Evaluates all 5 local models + global model on the 10% validation set.
Test set remains sealed until Step 6.

Metrics:
  - Regression (SPM, CFSM, RLM): RMSE, MAE, R²
  - Classification (PFM, GPM): Accuracy, Precision, Recall, F1, AUC-ROC
  - Global Model: RMSE, MAE, R²
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Plot styling
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGSIZE = (10, 6)


def evaluate_regression(name, y_true, y_pred):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    print(f"\n  ── {name} (Regression) ──")
    print(f"     RMSE:  {rmse:.6f}")
    print(f"     MAE:   {mae:.6f}")
    print(f"     R²:    {r2:.6f}")
    print(f"     MAPE:  {mape:.2f}%")

    return {"model": name, "RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}


def evaluate_classification(name, y_true, y_pred, y_prob):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    print(f"\n  ── {name} (Classification) ──")
    print(f"     Accuracy:  {acc:.4f}")
    print(f"     Precision: {prec:.4f}")
    print(f"     Recall:    {rec:.4f}")
    print(f"     F1:        {f1:.4f}")
    print(f"     AUC-ROC:   {auc:.4f}")

    return {"model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC-ROC": auc}


def plot_residuals(name, y_true, y_pred, prefix="val"):
    """Plot residual analysis for regression models."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{name} — Residual Analysis (Validation)", fontsize=14, fontweight="bold")

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#4C72B0")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # Residual distribution
    axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    # Predicted vs Actual
    axes[2].scatter(y_true, y_pred, alpha=0.3, s=8, color="#C44E52")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    axes[2].set_xlabel("Actual")
    axes[2].set_ylabel("Predicted")
    axes[2].set_title("Predicted vs Actual")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{prefix}_{name.lower()}_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Plot saved: {path}")


def plot_confusion_matrices(results_clf, val, models, feature_cols):
    """Plot confusion matrices for classification models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices (Validation)", fontsize=14, fontweight="bold")

    targets = {"PFM": "prerequisite_met", "GPM": "graduation_priority"}

    for idx, (name, target) in enumerate(targets.items()):
        y_true = val[target]
        y_pred = models[name.lower()].predict(val[feature_cols])

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        axes[idx].set_title(f"{name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "val_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion matrices saved: {path}")


def plot_metrics_summary(results_reg, results_clf):
    """Plot summary bar charts of all metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Validation Metrics Summary", fontsize=14, fontweight="bold")

    # Regression metrics
    reg_df = pd.DataFrame(results_reg)
    reg_melted = reg_df.melt(id_vars="model", var_name="Metric", value_name="Value")
    reg_melted = reg_melted[reg_melted["Metric"] != "MAPE"]
    sns.barplot(data=reg_melted, x="model", y="Value", hue="Metric", ax=axes[0])
    axes[0].set_title("Regression Models")
    axes[0].set_xlabel("")
    axes[0].legend(loc="upper right")

    # Classification metrics
    clf_df = pd.DataFrame(results_clf)
    clf_melted = clf_df.melt(id_vars="model", var_name="Metric", value_name="Value")
    sns.barplot(data=clf_melted, x="model", y="Value", hue="Metric", ax=axes[1])
    axes[1].set_title("Classification Models")
    axes[1].set_xlabel("")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "val_metrics_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Metrics summary saved: {path}")


def main():
    print("\n" + "=" * 60)
    print("  STEP 5: VALIDATION EVALUATION")
    print("  ⛔ Test set still sealed!")
    print("=" * 60 + "\n")

    # ── Load models and data ─────────────────────────────────────
    val = pd.read_pickle(os.path.join(DATA_DIR, "val.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.joblib"))
    target_cols = joblib.load(os.path.join(ARTIFACT_DIR, "target_cols.joblib"))

    models = {}
    for name in ["spm", "cfsm", "pfm", "gpm", "rlm"]:
        models[name] = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.joblib"))
    gmf_model = joblib.load(os.path.join(MODEL_DIR, "gmf_model.joblib"))

    X_val = val[feature_cols]

    # ── Evaluate Local Models ────────────────────────────────────
    print("=" * 60)
    print("  LOCAL MODEL EVALUATION")
    print("=" * 60)

    results_reg = []
    results_clf = []

    # SPM (regression)
    y_pred = models["spm"].predict(X_val)
    y_true = val[target_cols["spm"]]
    res = evaluate_regression("SPM", y_true.values, y_pred)
    results_reg.append(res)
    plot_residuals("SPM", y_true.values, y_pred)

    # CFSM (regression)
    y_pred = models["cfsm"].predict(X_val)
    y_true = val[target_cols["cfsm"]]
    res = evaluate_regression("CFSM", y_true.values, y_pred)
    results_reg.append(res)
    plot_residuals("CFSM", y_true.values, y_pred)

    # PFM (classification)
    y_pred = models["pfm"].predict(X_val)
    y_prob = models["pfm"].predict_proba(X_val)[:, 1]
    y_true = val[target_cols["pfm"]]
    res = evaluate_classification("PFM", y_true.values, y_pred, y_prob)
    results_clf.append(res)

    # GPM (classification)
    y_pred = models["gpm"].predict(X_val)
    y_prob = models["gpm"].predict_proba(X_val)[:, 1]
    y_true = val[target_cols["gpm"]]
    res = evaluate_classification("GPM", y_true.values, y_pred, y_prob)
    results_clf.append(res)

    # RLM (regression)
    y_pred = models["rlm"].predict(X_val)
    y_true = val[target_cols["rlm"]]
    res = evaluate_regression("RLM", y_true.values, y_pred)
    results_reg.append(res)
    plot_residuals("RLM", y_true.values, y_pred)

    # ── Evaluate Global Model ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GLOBAL MODEL (GMF) EVALUATION")
    print("=" * 60)

    meta_val = joblib.load(os.path.join(ARTIFACT_DIR, "meta_val.joblib"))
    y_global_val = joblib.load(os.path.join(ARTIFACT_DIR, "y_global_val.joblib"))

    gmf_pred = gmf_model.predict(meta_val)
    res_gmf = evaluate_regression("GMF", y_global_val, gmf_pred)
    results_reg.append(res_gmf)
    plot_residuals("GMF", y_global_val, gmf_pred)

    # ── Plots ────────────────────────────────────────────────────
    plot_confusion_matrices(results_clf, val, models, feature_cols)
    plot_metrics_summary(results_reg, results_clf)

    # ── Results Table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS SUMMARY")
    print("=" * 60)

    print("\n── Regression Models ──")
    reg_df = pd.DataFrame(results_reg)
    print(reg_df.to_string(index=False))

    print("\n── Classification Models ──")
    clf_df = pd.DataFrame(results_clf)
    print(clf_df.to_string(index=False))

    # Save results
    joblib.dump({"regression": results_reg, "classification": results_clf},
                os.path.join(ARTIFACT_DIR, "val_metrics.joblib"))

    print("\n" + "=" * 60)
    print("  STEP 5 COMPLETE")
    print("  ⛔ Test set still sealed — proceed to Step 6!")
    print("=" * 60)


if __name__ == "__main__":
    main()
