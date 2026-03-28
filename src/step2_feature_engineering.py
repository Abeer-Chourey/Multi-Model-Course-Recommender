"""
Step 2: Feature Engineering
===========================
Creates student-course interaction dataset and engineers features
for all 5 local models (SPM, CFSM, PFM, GPM, RLM).

Applies z-score normalization (matching paper methodology).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data():
    """Load cleaned datasets from Step 1."""
    courses = pd.read_pickle(os.path.join(DATA_DIR, "courses.pkl"))
    students = pd.read_pickle(os.path.join(DATA_DIR, "students.pkl"))
    print(f"Loaded courses: {courses.shape}")
    print(f"Loaded students: {students.shape}")
    return courses, students


def encode_course_difficulty(courses):
    """Map course level to numeric difficulty (1-4 scale)."""
    level_map = {
        "Beginner Level": 1, "Beginner": 1,
        "Intermediate Level": 2, "Intermediate": 2,
        "Expert Level": 3, "Expert": 3, "Advanced": 3,
        "All Levels": 2, "All": 2  # mid-range default
    }

    level_col = None
    for col in courses.columns:
        if "level" in col.lower():
            level_col = col
            break

    if level_col:
        courses["course_difficulty"] = courses[level_col].map(level_map).fillna(2)
    else:
        courses["course_difficulty"] = 2  # default mid

    print(f"  Course difficulty distribution:\n{courses['course_difficulty'].value_counts().sort_index()}")
    return courses


def compute_course_features(courses):
    """Engineer numeric features for courses."""
    print("\nEngineering course features...")

    # Popularity score (normalized subscribers + reviews)
    sub_col = [c for c in courses.columns if "subscrib" in c.lower()]
    rev_col = [c for c in courses.columns if "review" in c.lower() and "num" in c.lower()]

    if sub_col:
        courses["norm_subscribers"] = np.log1p(courses[sub_col[0]])
    else:
        courses["norm_subscribers"] = 0

    if rev_col:
        courses["norm_reviews"] = np.log1p(courses[rev_col[0]])
    else:
        courses["norm_reviews"] = 0

    # Popularity composite
    courses["course_popularity"] = (
        0.6 * courses["norm_subscribers"] + 0.4 * courses["norm_reviews"]
    )

    # Cost (normalized price)
    price_col = [c for c in courses.columns if "price" in c.lower()]
    if price_col:
        courses["course_cost"] = pd.to_numeric(courses[price_col[0]], errors="coerce").fillna(0)
    else:
        courses["course_cost"] = 0

    # Content richness (lectures × duration if available)
    lec_col = [c for c in courses.columns if "lecture" in c.lower()]
    dur_col = [c for c in courses.columns if "duration" in c.lower()]
    if lec_col and dur_col:
        courses["content_richness"] = (
            np.log1p(courses[lec_col[0]].fillna(0)) +
            np.log1p(courses[dur_col[0]].fillna(0))
        )
    elif lec_col:
        courses["content_richness"] = np.log1p(courses[lec_col[0]].fillna(0))
    else:
        courses["content_richness"] = 0

    # Subject encoding
    subj_col = [c for c in courses.columns if "subject" in c.lower() or "category" in c.lower()]
    if subj_col:
        le = LabelEncoder()
        courses["subject_encoded"] = le.fit_transform(courses[subj_col[0]].astype(str))
        joblib.dump(le, os.path.join(ARTIFACT_DIR, "subject_encoder.joblib"))
        print(f"  Subjects: {list(le.classes_)}")
    else:
        courses["subject_encoded"] = 0

    # Assign a unique course_id if not present
    if "course_id" not in courses.columns:
        id_col = courses.columns[0]
        courses["course_id"] = courses[id_col]

    print(f"  Course features engineered: {courses.shape}")
    return courses


def compute_student_features(students):
    """Engineer numeric features for students."""
    print("\nEngineering student features...")

    # Student capability (composite of GPA + academic performance)
    gpa_col = [c for c in students.columns if "gpa" in c.lower()]
    grade_col = [c for c in students.columns if "grade" in c.lower() and "avg" in c.lower()]

    if gpa_col:
        max_gpa = students[gpa_col[0]].max()
        students["student_capability"] = students[gpa_col[0]] / max_gpa  # normalize 0-1
    elif grade_col:
        max_grade = students[grade_col[0]].max()
        students["student_capability"] = students[grade_col[0]] / max_grade
    else:
        students["student_capability"] = 0.5  # default

    # Engagement score
    engagement_cols = []
    for prefix in ["attendance", "lms_login", "assignment", "forum", "video"]:
        matching = [c for c in students.columns if prefix in c.lower()]
        engagement_cols.extend(matching)

    if engagement_cols:
        # Normalize each engagement column to 0-1 and average
        eng_data = students[engagement_cols].copy()
        for col in eng_data.columns:
            col_min = eng_data[col].min()
            col_max = eng_data[col].max()
            if col_max > col_min:
                eng_data[col] = (eng_data[col] - col_min) / (col_max - col_min)
            else:
                eng_data[col] = 0.5
        students["engagement_score"] = eng_data.mean(axis=1)
    else:
        students["engagement_score"] = 0.5

    # Course load
    load_col = [c for c in students.columns if "load" in c.lower() or "course_load" in c.lower()]
    if load_col:
        students["current_load"] = students[load_col[0]]
    else:
        students["current_load"] = 5  # default

    # Enrollment status encoding
    status_col = [c for c in students.columns if "status" in c.lower() or "enrollment" in c.lower()]
    if status_col:
        le_status = LabelEncoder()
        students["enrollment_encoded"] = le_status.fit_transform(students[status_col[0]].astype(str))
        joblib.dump(le_status, os.path.join(ARTIFACT_DIR, "status_encoder.joblib"))
        print(f"  Enrollment statuses: {list(le_status.classes_)}")
    else:
        students["enrollment_encoded"] = 0

    # Risk level encoding (for later use)
    risk_col = [c for c in students.columns if "risk" in c.lower()]
    if risk_col:
        risk_map = {"Low": 0, "Medium": 1, "High": 2}
        students["risk_encoded"] = students[risk_col[0]].map(risk_map).fillna(1)
    else:
        students["risk_encoded"] = 0

    # Assign student_id if not present
    if "student_id" not in students.columns:
        id_col = students.columns[0]
        students["student_id"] = students[id_col]

    print(f"  Student features engineered: {students.shape}")
    return students


def create_interaction_dataset(courses, students, max_interactions=100000):
    """
    Create student-course interaction dataset.
    Simulates realistic interactions between students and courses.
    Paper used 101,330 records; we target ~100K.
    """
    print("\n" + "=" * 60)
    print("Creating student-course interaction dataset...")
    print("=" * 60)

    np.random.seed(42)

    n_students = len(students)
    n_courses = len(courses)
    avg_courses_per_student = max(1, max_interactions // n_students)

    print(f"  Students: {n_students}, Courses: {n_courses}")
    print(f"  Target interactions: ~{max_interactions:,}")
    print(f"  Avg courses/student: {avg_courses_per_student}")

    interactions = []

    for _, student in students.iterrows():
        # Each student interacts with a random subset of courses
        n_sample = min(n_courses, np.random.poisson(avg_courses_per_student) + 1)
        n_sample = max(1, min(n_sample, n_courses))

        sampled_courses = courses.sample(n=n_sample, replace=False)

        for _, course in sampled_courses.iterrows():
            interactions.append({
                "student_id": student["student_id"],
                "course_id": course["course_id"],
                # Student features
                "student_capability": student["student_capability"],
                "engagement_score": student["engagement_score"],
                "current_load": student["current_load"],
                "enrollment_encoded": student["enrollment_encoded"],
                "risk_encoded": student["risk_encoded"],
                # Course features
                "course_difficulty": course["course_difficulty"],
                "course_popularity": course["course_popularity"],
                "course_cost": course["course_cost"],
                "content_richness": course["content_richness"],
                "subject_encoded": course["subject_encoded"],
            })

    interaction_df = pd.DataFrame(interactions)

    # Trim to target size if needed
    if len(interaction_df) > max_interactions:
        interaction_df = interaction_df.sample(n=max_interactions, random_state=42).reset_index(drop=True)

    print(f"  Created {len(interaction_df):,} interaction records")
    return interaction_df


def engineer_target_variables(df):
    """
    Engineer target variables for all 5 local models.

    SPM  (Success Probability):   regression  → [0, 1]
    CFSM (Course Fit Score):      regression  → [0, 1]
    PFM  (Prerequisite Fulfilled): binary classification → {0, 1}
    GPM  (Graduation Priority):   binary classification → {0, 1}
    RLM  (Recommended Load):      regression  → continuous
    """
    print("\nEngineering target variables for 5 local models...")
    np.random.seed(42)

    # ── SPM: Success Probability ─────────────────────────────────
    # Higher capability + higher engagement + lower difficulty → higher success
    difficulty_gap = np.abs(df["student_capability"] - (df["course_difficulty"] / 4.0))
    base_success = (
        0.40 * df["student_capability"] +
        0.30 * df["engagement_score"] +
        0.20 * (1 - difficulty_gap) +
        0.10 * (1 - df["risk_encoded"] / 2.0)
    )
    # Add realistic noise
    noise = np.random.normal(0, 0.05, len(df))
    df["success_probability"] = np.clip(base_success + noise, 0, 1)
    print(f"  SPM target - mean: {df['success_probability'].mean():.4f}, std: {df['success_probability'].std():.4f}")

    # ── CFSM: Course Fit Score ───────────────────────────────────
    # How well the course matches the student's academic profile
    fit_base = (
        0.35 * (1 - difficulty_gap) +  # difficulty alignment
        0.25 * df["engagement_score"] +  # engaged students fit better
        0.20 * df["course_popularity"] / df["course_popularity"].max() +  # popular courses fit more
        0.20 * df["content_richness"] / max(df["content_richness"].max(), 1)
    )
    noise_fit = np.random.normal(0, 0.04, len(df))
    df["course_fit_score"] = np.clip(fit_base + noise_fit, 0, 1)
    print(f"  CFSM target - mean: {df['course_fit_score'].mean():.4f}, std: {df['course_fit_score'].std():.4f}")

    # ── PFM: Prerequisite Fulfilled ──────────────────────────────
    # Binary: student has sufficient capability for the course
    prereq_threshold = df["course_difficulty"] / 4.0 - 0.15  # slightly lenient
    prereq_prob = 1 / (1 + np.exp(-10 * (df["student_capability"] - prereq_threshold)))
    df["prerequisite_met"] = (np.random.random(len(df)) < prereq_prob).astype(int)
    print(f"  PFM target - distribution: {df['prerequisite_met'].value_counts().to_dict()}")

    # ── GPM: Graduation Priority ─────────────────────────────────
    # Binary: is this course a graduation priority?
    # Higher load + advanced courses + active enrollment → priority
    priority_score = (
        0.35 * (df["course_difficulty"] / 4.0) +
        0.30 * (df["current_load"] / max(df["current_load"].max(), 1)) +
        0.20 * (1 - df["enrollment_encoded"] / max(df["enrollment_encoded"].max(), 1)) +
        0.15 * df["student_capability"]
    )
    priority_threshold = priority_score.quantile(0.5)
    noise_gpm = np.random.normal(0, 0.03, len(df))
    df["graduation_priority"] = ((priority_score + noise_gpm) >= priority_threshold).astype(int)
    print(f"  GPM target - distribution: {df['graduation_priority'].value_counts().to_dict()}")

    # ── RLM: Recommended Load ────────────────────────────────────
    # Optimal course load based on student profile (1-7 courses)
    base_load = (
        2.5 * df["student_capability"] +
        1.5 * df["engagement_score"] +
        0.8 * (1 - df["risk_encoded"] / 2.0) +
        0.2 * df["current_load"] / max(df["current_load"].max(), 1)
    )
    noise_load = np.random.normal(0, 0.3, len(df))
    df["recommended_load"] = np.clip(base_load + noise_load, 1, 7).round(1)
    print(f"  RLM target - mean: {df['recommended_load'].mean():.2f}, std: {df['recommended_load'].std():.2f}")

    return df


def apply_normalization(df):
    """Apply z-score normalization to features (matching paper's methodology)."""
    print("\nApplying z-score normalization...")

    feature_cols = [
        "student_capability", "engagement_score", "current_load",
        "enrollment_encoded", "risk_encoded",
        "course_difficulty", "course_popularity", "course_cost",
        "content_richness", "subject_encoded"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler and feature config
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(ARTIFACT_DIR, "feature_cols.joblib"))

    print(f"  Normalized {len(feature_cols)} features: {feature_cols}")
    print(f"  Scaler saved to artifacts/")

    return df, feature_cols


def main():
    print("\n" + "=" * 60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 60 + "\n")

    # ── Load data ────────────────────────────────────────────────
    courses, students = load_data()

    # ── Engineer features ────────────────────────────────────────
    courses = encode_course_difficulty(courses)
    courses = compute_course_features(courses)
    students = compute_student_features(students)

    # ── Create interaction dataset ───────────────────────────────
    interaction_df = create_interaction_dataset(courses, students, max_interactions=100000)

    # ── Engineer targets for 5 models ────────────────────────────
    interaction_df = engineer_target_variables(interaction_df)

    # ── Normalize ────────────────────────────────────────────────
    interaction_df, feature_cols = apply_normalization(interaction_df)

    # ── Save ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving engineered dataset...")
    print("=" * 60)

    out_path = os.path.join(DATA_DIR, "interaction_features.pkl")
    interaction_df.to_pickle(out_path)

    # Also save course and student feature-enriched datasets
    courses.to_pickle(os.path.join(DATA_DIR, "courses_enriched.pkl"))
    students.to_pickle(os.path.join(DATA_DIR, "students_enriched.pkl"))

    print(f"  Interaction dataset: {out_path}")
    print(f"  Shape: {interaction_df.shape}")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Target columns: success_probability, course_fit_score, prerequisite_met, graduation_priority, recommended_load")

    # ── Summary statistics ───────────────────────────────────────
    print("\n── Feature Statistics ──")
    print(interaction_df[feature_cols].describe().round(4).to_string())
    print("\n── Target Statistics ──")
    target_cols = ["success_probability", "course_fit_score", "prerequisite_met",
                   "graduation_priority", "recommended_load"]
    print(interaction_df[target_cols].describe().round(4).to_string())

    print("\n" + "=" * 60)
    print("  STEP 2 COMPLETE")
    print("=" * 60)

    return interaction_df


if __name__ == "__main__":
    main()
