"""
Step 1: Data Loading
====================
Loads the two datasets from local CSV paths entered by the user:
  - Udemy Course Recommendation  (use udemy_course_data.csv)
  - College Student Management

Saves cleaned DataFrames as .pkl files in data/
"""

import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_udemy():
    """Load the Udemy Course Recommendation dataset from a local CSV path."""
    print("=" * 60)
    print("Loading Udemy Course Recommendation dataset...")
    print("=" * 60)

    default_path = "E:/SEM-6/Recommendation/Mini_Project/Dataset/udemy_course_data.csv"
    csv_path = input(f"  Enter the full path to udemy_course_data.csv [{default_path}]: ").strip().strip('"').strip("'")
    if not csv_path:
        csv_path = default_path

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}\n")
    return df


def load_student():
    """Load the College Student Management dataset from a local CSV path."""
    print("=" * 60)
    print("Loading College Student Management dataset...")
    print("=" * 60)

    default_path = "E:/SEM-6/Recommendation/Mini_Project/Dataset/college_student_management_data.csv"
    csv_path = input(f"  Enter the full path to the student management CSV [{default_path}]: ").strip().strip('"').strip("'")
    if not csv_path:
        csv_path = default_path

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}\n")
    return df


def clean_udemy(df):
    """Clean the Udemy course dataset."""
    print("Cleaning Udemy dataset...")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure key columns exist and handle missing values
    expected_cols = [
        "course_id", "course_title", "is_paid", "price",
        "num_subscribers", "num_reviews", "num_lectures",
        "level", "content_duration", "subject"
    ]
    available = [c for c in expected_cols if c in df.columns]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"  Warning: Missing expected columns: {missing}")

    # Convert numeric columns
    for col in ["price", "num_subscribers", "num_reviews", "num_lectures", "content_duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert is_paid to boolean
    if "is_paid" in df.columns:
        df["is_paid"] = df["is_paid"].astype(bool)

    # Drop rows where course_id is null (if exists)
    id_col = "course_id" if "course_id" in df.columns else df.columns[0]
    initial = len(df)
    df = df.dropna(subset=[id_col])
    print(f"  Dropped {initial - len(df)} rows with null IDs")

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {col} NaNs with median={median_val:.2f}")

    print(f"  Final shape: {df.shape}")
    return df


def clean_student(df):
    """Clean the Student Management dataset."""
    print("Cleaning Student Management dataset...")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert numeric columns
    numeric_candidates = [
        "age", "gpa", "course_load", "avg_course_grade",
        "attendance_rate", "lms_logins_past_month",
        "avg_session_duration_minutes", "assignment_submission_rate",
        "forum_participation_count", "video_completion_rate"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with null student_id
    id_col = "student_id" if "student_id" in df.columns else df.columns[0]
    initial = len(df)
    df = df.dropna(subset=[id_col])
    print(f"  Dropped {initial - len(df)} rows with null IDs")

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {col} NaNs with median={median_val:.2f}")

    print(f"  Final shape: {df.shape}")
    return df


def main():
    print("\n" + "=" * 60)
    print("  STEP 1: DATA LOADING")
    print("=" * 60 + "\n")

    # ── Load datasets ────────────────────────────────────────────
    course_df = load_udemy()
    student_df = load_student()

    # ── Clean datasets ───────────────────────────────────────────
    print()
    course_df = clean_udemy(course_df)
    print()
    student_df = clean_student(student_df)

    # ── Save as pickle ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving cleaned datasets...")
    print("=" * 60)

    course_path = os.path.join(DATA_DIR, "courses.pkl")
    student_path = os.path.join(DATA_DIR, "students.pkl")

    course_df.to_pickle(course_path)
    student_df.to_pickle(student_path)

    print(f"  Courses saved to: {course_path}")
    print(f"  Students saved to: {student_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 COMPLETE")
    print("=" * 60)
    print(f"\n  Courses:  {course_df.shape[0]:,} rows × {course_df.shape[1]} cols")
    print(f"  Students: {student_df.shape[0]:,} rows × {student_df.shape[1]} cols")
    print(f"\n  Course columns:  {list(course_df.columns)}")
    print(f"  Student columns: {list(student_df.columns)}")

    # Preview
    print("\n── Course Data Sample ──")
    print(course_df.head(3).to_string())
    print("\n── Student Data Sample ──")
    print(student_df.head(3).to_string())

    return course_df, student_df


if __name__ == "__main__":
    main()
