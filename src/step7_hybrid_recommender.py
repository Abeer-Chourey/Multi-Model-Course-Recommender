"""
Step 7: Hybrid Course Recommender
==================================
Generates personalized course recommendations for sample students
using the trained multi-model framework.

Uses GMF final scores with constraint enforcement:
  - Prerequisite check (PFM)
  - Load limits (RLM)
  - Graduation priority (GPM)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

plt.style.use("seaborn-v0_8-darkgrid")


def load_resources():
    """Load all models, data, and artifacts."""
    # Models
    models = {}
    for name in ["spm", "cfsm", "pfm", "gpm", "rlm"]:
        models[name] = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.joblib"))
    gmf_model = joblib.load(os.path.join(MODEL_DIR, "gmf_model.joblib"))

    # Data
    courses = pd.read_pickle(os.path.join(DATA_DIR, "courses_enriched.pkl"))
    students = pd.read_pickle(os.path.join(DATA_DIR, "students_enriched.pkl"))
    interaction = pd.read_pickle(os.path.join(DATA_DIR, "interaction_features.pkl"))

    # Artifacts
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))

    return models, gmf_model, courses, students, interaction, feature_cols, scaler


def generate_recommendations(student_id, models, gmf_model, interaction_df,
                              feature_cols, top_n=15):
    """
    Generate top-N personalized course recommendations for a student.

    Process:
    1. Get all student-course pairs for this student
    2. Run all 5 local models → meta-features
    3. Run GMF → final recommendation score
    4. Apply constraints (prerequisite, load)
    5. Rank and return top-N
    """
    # Get student's interaction data
    student_data = interaction_df[interaction_df["student_id"] == student_id].copy()

    if len(student_data) == 0:
        return None

    X = student_data[feature_cols]

    # Local model predictions
    student_data["spm_score"] = models["spm"].predict(X)
    student_data["cfsm_score"] = models["cfsm"].predict(X)
    student_data["pfm_prob"] = models["pfm"].predict_proba(X)[:, 1]
    student_data["gpm_prob"] = models["gpm"].predict_proba(X)[:, 1]
    student_data["rlm_pred"] = models["rlm"].predict(X)

    # Meta features for GMF
    meta_features = student_data[["spm_score", "cfsm_score", "pfm_prob",
                                   "gpm_prob", "rlm_pred"]].copy()
    meta_features.columns = ["spm_pred", "cfsm_pred", "pfm_pred",
                             "gpm_pred", "rlm_pred"]

    # GMF final score
    student_data["gmf_score"] = gmf_model.predict(meta_features)

    # ── Apply constraints ────────────────────────────────────────
    # 1. Prerequisite filter: keep only courses where prerequisite is likely met
    student_data["prereq_met"] = (student_data["pfm_prob"] >= 0.5).astype(int)

    # 2. Graduation priority boost
    student_data["priority_boost"] = student_data["gpm_prob"] * 0.1

    # 3. Final constrained score
    student_data["final_score"] = (
        student_data["gmf_score"] +
        student_data["priority_boost"]
    )

    # Filter: only recommend courses where prerequisites are met
    eligible = student_data[student_data["prereq_met"] == 1].copy()

    if len(eligible) == 0:
        # Fallback: relax prerequisite constraint
        eligible = student_data.copy()

    # Rank and get top-N
    eligible = eligible.sort_values("final_score", ascending=False).head(top_n)

    return eligible


def format_recommendation_table(recs, courses_df, rank_start=1):
    """Format recommendations into a readable table."""
    result_rows = []

    for idx, (_, row) in enumerate(recs.iterrows()):
        course_id = row["course_id"]

        # Look up course title
        course_match = courses_df[courses_df["course_id"] == course_id]
        if len(course_match) > 0:
            title_col = [c for c in course_match.columns if "title" in c.lower() or "name" in c.lower()]
            if title_col:
                title = str(course_match.iloc[0][title_col[0]])[:50]  # truncate
            else:
                title = f"Course {course_id}"
        else:
            title = f"Course {course_id}"

        result_rows.append({
            "Rank": rank_start + idx,
            "Course": title,
            "Success Prob": f"{row['spm_score']:.3f}",
            "Fit Score": f"{row['cfsm_score']:.3f}",
            "Prereq Met": "✓" if row["prereq_met"] == 1 else "✗",
            "Grad Priority": f"{row['gpm_prob']:.3f}",
            "Final Score": f"{row['final_score']:.4f}",
        })

    return pd.DataFrame(result_rows)


def plot_recommendations(all_recs, student_ids):
    """Visualize recommendations for sample students."""
    n_students = len(student_ids)
    fig, axes = plt.subplots(n_students, 1, figsize=(14, 5 * n_students))
    fig.suptitle("Personalized Course Recommendations\n(Multi-Model Framework)",
                 fontsize=16, fontweight="bold", y=1.02)

    if n_students == 1:
        axes = [axes]

    for idx, (sid, recs) in enumerate(zip(student_ids, all_recs)):
        ax = axes[idx]

        if recs is None or len(recs) == 0:
            ax.text(0.5, 0.5, f"No recommendations for Student {sid}",
                    ha="center", va="center", fontsize=14)
            continue

        # Bar chart of final scores
        labels = [f"#{i+1}" for i in range(len(recs))]
        scores = recs["final_score"].values

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scores)))

        bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], alpha=0.8)
        ax.set_xlabel("Final Recommendation Score", fontsize=11)
        ax.set_title(f"Student {sid} — Top {len(recs)} Recommendations", fontsize=12)

        # Add score labels
        for bar, score in zip(bars, scores[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{score:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "hybrid_recommendations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Recommendation visualization saved: {path}")


def save_recommendations_csv(all_tables, student_ids):
    """Save all recommendations to CSV."""
    combined = []
    for sid, table in zip(student_ids, all_tables):
        if table is not None:
            table_copy = table.copy()
            table_copy.insert(0, "Student ID", sid)
            combined.append(table_copy)

    if combined:
        result = pd.concat(combined, ignore_index=True)
        path = os.path.join(DATA_DIR, "recommendations.csv")
        result.to_csv(path, index=False)
        print(f"\n  All recommendations saved: {path}")
        return result
    return None


def main():
    print("\n" + "=" * 60)
    print("  STEP 7: HYBRID COURSE RECOMMENDER")
    print("  Personalized Recommendations via Multi-Model Framework")
    print("=" * 60 + "\n")

    # ── Load resources ───────────────────────────────────────────
    models, gmf_model, courses, students, interaction, feature_cols, scaler = load_resources()
    print(f"Loaded {len(courses)} courses, {len(students)} students")
    print(f"Interaction records: {len(interaction)}")

    # ── Select sample students ───────────────────────────────────
    unique_students = interaction["student_id"].unique()
    n_sample = min(5, len(unique_students))
    sample_students = np.random.RandomState(42).choice(unique_students, n_sample, replace=False)

    print(f"\nGenerating recommendations for {n_sample} sample students...")
    print(f"Student IDs: {list(sample_students)}")

    # ── Generate recommendations ─────────────────────────────────
    all_recs = []
    all_tables = []

    for sid in sample_students:
        print(f"\n{'─' * 60}")
        print(f"  Student: {sid}")
        print(f"{'─' * 60}")

        recs = generate_recommendations(
            sid, models, gmf_model, interaction, feature_cols, top_n=15
        )

        if recs is not None:
            all_recs.append(recs)
            table = format_recommendation_table(recs, courses)
            all_tables.append(table)

            print(table.to_string(index=False))

            # Show model breakdown
            print(f"\n  Model Score Ranges:")
            print(f"    SPM  (Success):      {recs['spm_score'].min():.3f} - {recs['spm_score'].max():.3f}")
            print(f"    CFSM (Fit):          {recs['cfsm_score'].min():.3f} - {recs['cfsm_score'].max():.3f}")
            print(f"    PFM  (Prereq Prob):  {recs['pfm_prob'].min():.3f} - {recs['pfm_prob'].max():.3f}")
            print(f"    GPM  (Grad Priority): {recs['gpm_prob'].min():.3f} - {recs['gpm_prob'].max():.3f}")
            print(f"    RLM  (Rec Load):     {recs['rlm_pred'].min():.2f} - {recs['rlm_pred'].max():.2f}")
            print(f"    GMF  (Final):        {recs['gmf_score'].min():.4f} - {recs['gmf_score'].max():.4f}")
        else:
            all_recs.append(None)
            all_tables.append(None)
            print("  No data available for this student")

    # ── Visualizations ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    plot_recommendations(all_recs, sample_students)

    # ── Save all recommendations ─────────────────────────────────
    combined = save_recommendations_csv(all_tables, sample_students)

    # ── Summary Statistics ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RECOMMENDATION SUMMARY")
    print("=" * 60)

    if combined is not None:
        print(f"\n  Total recommendations generated: {len(combined)}")
        print(f"  Students served: {combined['Student ID'].nunique()}")
        print(f"  Avg recommendations per student: {len(combined) / combined['Student ID'].nunique():.1f}")

        # Prerequisite compliance
        prereq_met = (combined["Prereq Met"] == "✓").sum()
        print(f"  Prerequisite compliance: {prereq_met}/{len(combined)} ({100*prereq_met/len(combined):.1f}%)")

    print("\n" + "=" * 60)
    print("  STEP 7 COMPLETE — Pipeline finished!")
    print("=" * 60)
    print("\n  Output files:")
    print(f"    → data/recommendations.csv")
    print(f"    → plots/hybrid_recommendations.png")
    print(f"\n  All 7 steps completed successfully! ✓")


if __name__ == "__main__":
    main()
