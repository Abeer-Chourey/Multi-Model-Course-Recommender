"""
Step 3: Data Splitting
======================
Splits the interaction dataset into 70% train / 10% val / 20% test.
Verifies zero data leakage between splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def main():
    print("\n" + "=" * 60)
    print("  STEP 3: DATA SPLITTING (70/10/20)")
    print("=" * 60 + "\n")

    # ── Load interaction dataset ─────────────────────────────────
    df = pd.read_pickle(os.path.join(DATA_DIR, "interaction_features.pkl"))
    print(f"Loaded interaction dataset: {df.shape}")

    # ── Split: 80/20 first, then 87.5/12.5 → 70/10/20 ──────────
    print("\nSplitting data...")

    # First split: 80% (train+val) vs 20% (test)
    train_val, test = train_test_split(df, test_size=0.20, random_state=42)

    # Second split: 87.5/12.5 of 80% → 70/10 of total
    train, val = train_test_split(train_val, test_size=0.125, random_state=42)

    print(f"  Train: {len(train):>8,} ({100 * len(train) / len(df):.1f}%)")
    print(f"  Val:   {len(val):>8,} ({100 * len(val) / len(df):.1f}%)")
    print(f"  Test:  {len(test):>8,} ({100 * len(test) / len(df):.1f}%)")
    print(f"  Total: {len(train) + len(val) + len(test):>8,}")

    # ── Verify zero data leakage ─────────────────────────────────
    print("\nVerifying zero data leakage...")

    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    overlap_tv = train_idx & val_idx
    overlap_tt = train_idx & test_idx
    overlap_vt = val_idx & test_idx

    assert len(overlap_tv) == 0, f"Train-Val overlap: {len(overlap_tv)}"
    assert len(overlap_tt) == 0, f"Train-Test overlap: {len(overlap_tt)}"
    assert len(overlap_vt) == 0, f"Val-Test overlap: {len(overlap_vt)}"

    print(f"  Train ∩ Val:  {len(overlap_tv)} (OK)")
    print(f"  Train ∩ Test: {len(overlap_tt)} (OK)")
    print(f"  Val ∩ Test:   {len(overlap_vt)} (OK)")
    print("  ✓ Zero data leakage confirmed!")

    # ── Verify target distributions ──────────────────────────────
    print("\nTarget distribution across splits:")
    targets = ["success_probability", "course_fit_score", "prerequisite_met",
               "graduation_priority", "recommended_load"]

    for target in targets:
        if target in df.columns:
            if target in ["prerequisite_met", "graduation_priority"]:
                # Classification targets - show class balance
                t_pct = train[target].mean() * 100
                v_pct = val[target].mean() * 100
                te_pct = test[target].mean() * 100
                print(f"  {target:25s} → Train:{t_pct:5.1f}%  Val:{v_pct:5.1f}%  Test:{te_pct:5.1f}%  (class=1 %)")
            else:
                # Regression targets - show mean
                t_mean = train[target].mean()
                v_mean = val[target].mean()
                te_mean = test[target].mean()
                print(f"  {target:25s} → Train:{t_mean:7.4f}  Val:{v_mean:7.4f}  Test:{te_mean:7.4f}  (mean)")

    # ── Save splits ──────────────────────────────────────────────
    print("\nSaving splits...")

    train.to_pickle(os.path.join(DATA_DIR, "train.pkl"))
    val.to_pickle(os.path.join(DATA_DIR, "val.pkl"))
    test.to_pickle(os.path.join(DATA_DIR, "test.pkl"))

    print(f"  Saved: data/train.pkl ({len(train):,} rows)")
    print(f"  Saved: data/val.pkl   ({len(val):,} rows)")
    print(f"  Saved: data/test.pkl  ({len(test):,} rows)")

    # ── ⛔ Test set sealed ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ⛔ TEST SET SEALED — Do NOT use until Step 6!")
    print("=" * 60)
    print("  STEP 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
