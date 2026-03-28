"""
Step 6: Final Test Evaluation
=============================
Unseals the test set for the FIRST TIME and evaluates all models.

Metrics:
  - Regression: RMSE, MAE, R²
  - Classification: Accuracy, Precision, Recall, F1, AUC-ROC
  - Ranking: Precision@k, Recall@k, NDCG@k (per-student, k=5,10,20)
  - Residual analysis (homoscedasticity check per paper)
  - Comparison: Paper results vs Our results
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
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix
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

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ── Ranking Metrics ──────────────────────────────────────────────────────────

def dcg_at_k(scores, k):
    """Compute DCG@k."""
    scores = np.asarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))


def ndcg_at_k(y_true, y_pred, k):
    """Compute NDCG@k."""
    order = np.argsort(-y_pred)
    y_true_sorted = np.asarray(y_true)[order]

    dcg = dcg_at_k(y_true_sorted, k)
    ideal_order = np.argsort(-np.asarray(y_true))
    ideal_sorted = np.asarray(y_true)[ideal_order]
    idcg = dcg_at_k(ideal_sorted, k)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true_binary, y_pred, k):
    """Compute Precision@k."""
    top_k_idx = np.argsort(-y_pred)[:k]
    return np.mean(y_true_binary[top_k_idx]) if k > 0 else 0.0


def recall_at_k(y_true_binary, y_pred, k):
    """Compute Recall@k."""
    top_k_idx = np.argsort(-y_pred)[:k]
    n_relevant = y_true_binary.sum()
    if n_relevant == 0:
        return 0.0
    return y_true_binary[top_k_idx].sum() / n_relevant


def compute_ranking_metrics(test, gmf_pred, threshold_percentile=75):
    """Compute per-student ranking metrics."""
    print("\n  Computing per-student ranking metrics...")

    # Create relevance labels using threshold
    threshold = np.percentile(test["success_probability"], threshold_percentile)
    test = test.copy()
    test["relevant"] = (test["success_probability"] >= threshold).astype(int)
    test["gmf_score"] = gmf_pred

    student_col = "student_id"
    if student_col not in test.columns:
        print("  Warning: student_id not found, computing global ranking")
        return compute_global_ranking(test["relevant"].values, gmf_pred)

    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in [5, 10, 20]}

    for sid, group in test.groupby(student_col):
        if len(group) < 5:
            continue

        y_true = group["relevant"].values
        y_score = group["gmf_score"].values
        y_success = group["success_probability"].values

        for k in [5, 10, 20]:
            if len(group) >= k:
                results[k]["precision"].append(precision_at_k(y_true, y_score, k))
                results[k]["recall"].append(recall_at_k(y_true, y_score, k))
                results[k]["ndcg"].append(ndcg_at_k(y_success, y_score, k))

    ranking_results = {}
    for k in [5, 10, 20]:
        ranking_results[k] = {
            "Precision@k": np.mean(results[k]["precision"]) if results[k]["precision"] else 0,
            "Recall@k": np.mean(results[k]["recall"]) if results[k]["recall"] else 0,
            "NDCG@k": np.mean(results[k]["ndcg"]) if results[k]["ndcg"] else 0,
        }

    return ranking_results


def compute_global_ranking(y_relevant, y_pred):
    """Fallback: compute global ranking if no student grouping."""
    results = {}
    for k in [5, 10, 20]:
        results[k] = {
            "Precision@k": precision_at_k(y_relevant, y_pred, k),
            "Recall@k": recall_at_k(y_relevant, y_pred, k),
            "NDCG@k": ndcg_at_k(y_relevant.astype(float), y_pred, k),
        }
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_roc_curves(test, models, feature_cols):
    """Plot ROC curves for PFM and GPM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC Curves (Test Set)", fontsize=14, fontweight="bold")

    targets = {"PFM": "prerequisite_met", "GPM": "graduation_priority"}
    X_test = test[feature_cols]

    for idx, (name, target) in enumerate(targets.items()):
        y_true = test[target]
        y_prob = models[name.lower()].predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        axes[idx].plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        axes[idx].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[idx].set_xlabel("False Positive Rate")
        axes[idx].set_ylabel("True Positive Rate")
        axes[idx].set_title(f"{name} ROC Curve")
        axes[idx].legend(loc="lower right", fontsize=12)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "test_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved: {path}")


def plot_pr_curves(test, models, feature_cols):
    """Plot Precision-Recall curves for classification models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Precision-Recall Curves (Test Set)", fontsize=14, fontweight="bold")

    targets = {"PFM": "prerequisite_met", "GPM": "graduation_priority"}
    X_test = test[feature_cols]

    for idx, (name, target) in enumerate(targets.items()):
        y_true = test[target]
        y_prob = models[name.lower()].predict_proba(X_test)[:, 1]

        prec, rec, _ = precision_recall_curve(y_true, y_prob)

        axes[idx].plot(rec, prec, linewidth=2, color="#C44E52")
        axes[idx].set_xlabel("Recall")
        axes[idx].set_ylabel("Precision")
        axes[idx].set_title(f"{name} Precision-Recall Curve")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "test_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PR curves saved: {path}")


def plot_ranking_metrics(ranking_results):
    """Plot ranking metrics bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ranking Metrics (Test Set, Per-Student Averaged)", fontsize=14, fontweight="bold")

    metrics = ["Precision@k", "Recall@k", "NDCG@k"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for idx, metric in enumerate(metrics):
        k_vals = [5, 10, 20]
        values = [ranking_results[k][metric] for k in k_vals]

        bars = axes[idx].bar([f"k={k}" for k in k_vals], values, color=colors[idx], alpha=0.8)
        axes[idx].set_title(metric)
        axes[idx].set_ylim(0, 1)

        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f"{val:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "test_ranking_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ranking metrics saved: {path}")


def plot_test_residuals(y_true, y_pred, name):
    """Plot residual analysis for test set (homoscedasticity check per paper)."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{name} — Test Set Residual Analysis", fontsize=14, fontweight="bold")

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#4C72B0")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted\n(Homoscedasticity Check)")

    axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    axes[2].scatter(y_true, y_pred, alpha=0.3, s=8, color="#C44E52")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[2].plot([mn, mx], [mn, mx], "k--", linewidth=1)
    axes[2].set_xlabel("Actual")
    axes[2].set_ylabel("Predicted")
    axes[2].set_title("Predicted vs Actual")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"test_{name.lower()}_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Test residual plot saved: {path}")


def main():
    print("\n" + "=" * 60)
    print("  STEP 6: FINAL TEST EVALUATION")
    print("  ✅ Unsealing test set for the FIRST TIME!")
    print("=" * 60 + "\n")

    # ── Load models and test data ────────────────────────────────
    test = pd.read_pickle(os.path.join(DATA_DIR, "test.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.joblib"))
    target_cols = joblib.load(os.path.join(ARTIFACT_DIR, "target_cols.joblib"))

    models = {}
    for name in ["spm", "cfsm", "pfm", "gpm", "rlm"]:
        models[name] = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.joblib"))
    gmf_model = joblib.load(os.path.join(MODEL_DIR, "gmf_model.joblib"))

    X_test = test[feature_cols]
    print(f"Test set: {test.shape}")

    # ── Evaluate Local Models ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LOCAL MODEL RESULTS (TEST SET)")
    print("=" * 60)

    results_reg = []
    results_clf = []

    # SPM
    y_pred = models["spm"].predict(X_test)
    y_true = test[target_cols["spm"]].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results_reg.append({"Model": "SPM", "RMSE": rmse, "MAE": mae, "R²": r2})
    print(f"\n  SPM:  RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
    plot_test_residuals(y_true, y_pred, "SPM")

    # CFSM
    y_pred = models["cfsm"].predict(X_test)
    y_true = test[target_cols["cfsm"]].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results_reg.append({"Model": "CFSM", "RMSE": rmse, "MAE": mae, "R²": r2})
    print(f"  CFSM: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
    plot_test_residuals(y_true, y_pred, "CFSM")

    # PFM
    y_pred = models["pfm"].predict(X_test)
    y_prob = models["pfm"].predict_proba(X_test)[:, 1]
    y_true = test[target_cols["pfm"]].values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    results_clf.append({"Model": "PFM", "Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1": f1, "AUC-ROC": auc})
    print(f"  PFM:  Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # GPM
    y_pred = models["gpm"].predict(X_test)
    y_prob = models["gpm"].predict_proba(X_test)[:, 1]
    y_true = test[target_cols["gpm"]].values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    results_clf.append({"Model": "GPM", "Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1": f1, "AUC-ROC": auc})
    print(f"  GPM:  Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # RLM
    y_pred = models["rlm"].predict(X_test)
    y_true = test[target_cols["rlm"]].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results_reg.append({"Model": "RLM", "RMSE": rmse, "MAE": mae, "R²": r2})
    print(f"  RLM:  RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
    plot_test_residuals(y_true, y_pred, "RLM")

    # ── Global Model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GLOBAL MODEL (GMF) RESULTS (TEST SET)")
    print("=" * 60)

    # Generate meta features for test
    meta_test = pd.DataFrame()
    meta_test["spm_pred"] = models["spm"].predict(X_test)
    meta_test["cfsm_pred"] = models["cfsm"].predict(X_test)
    meta_test["pfm_pred"] = models["pfm"].predict_proba(X_test)[:, 1]
    meta_test["gpm_pred"] = models["gpm"].predict_proba(X_test)[:, 1]
    meta_test["rlm_pred"] = models["rlm"].predict(X_test)

    y_global_test = (
        0.30 * test[target_cols["spm"]].values +
        0.25 * test[target_cols["cfsm"]].values +
        0.20 * test[target_cols["pfm"]].values.astype(float) +
        0.15 * test[target_cols["gpm"]].values.astype(float) +
        0.10 * (test[target_cols["rlm"]].values / 7.0)
    )

    gmf_pred = gmf_model.predict(meta_test)
    rmse = np.sqrt(mean_squared_error(y_global_test, gmf_pred))
    mae = mean_absolute_error(y_global_test, gmf_pred)
    r2 = r2_score(y_global_test, gmf_pred)
    results_reg.append({"Model": "GMF", "RMSE": rmse, "MAE": mae, "R²": r2})
    print(f"\n  GMF:  RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
    plot_test_residuals(y_global_test, gmf_pred, "GMF")

    # ── Ranking Metrics ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RANKING METRICS (TEST SET)")
    print("=" * 60)

    ranking_results = compute_ranking_metrics(test, gmf_pred)
    for k in [5, 10, 20]:
        print(f"\n  k={k}:")
        for metric, value in ranking_results[k].items():
            print(f"    {metric}: {value:.4f}")

    # ── Plots ────────────────────────────────────────────────────
    print("\nGenerating test plots...")
    plot_roc_curves(test, models, feature_cols)
    plot_pr_curves(test, models, feature_cols)
    plot_ranking_metrics(ranking_results)

    # ── Comparison: Paper vs Ours ────────────────────────────────
    print("\n" + "=" * 60)
    print("  PAPER vs OUR RESULTS")
    print("=" * 60)

    comparison = pd.DataFrame([
        {"Model": "SPM", "Paper RMSE": 0.00956, "Our RMSE": results_reg[0]["RMSE"]},
        {"Model": "CFSM", "Paper RMSE": 0.01171, "Our RMSE": results_reg[1]["RMSE"]},
        {"Model": "PFM", "Paper Acc": ">99%", "Our Acc": f"{results_clf[0]['Accuracy']:.4f}"},
        {"Model": "GPM", "Paper Acc": ">99%", "Our Acc": f"{results_clf[1]['Accuracy']:.4f}"},
        {"Model": "RLM", "Paper RMSE": 0.00541, "Our RMSE": results_reg[2]["RMSE"]},
    ])
    print(comparison.to_string(index=False))

    # ── Save results ─────────────────────────────────────────────
    all_results = {
        "regression": results_reg,
        "classification": results_clf,
        "ranking": ranking_results,
    }
    joblib.dump(all_results, os.path.join(ARTIFACT_DIR, "test_metrics.joblib"))

    print("\n" + "=" * 60)
    print("  STEP 6 COMPLETE — Test evaluation finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
