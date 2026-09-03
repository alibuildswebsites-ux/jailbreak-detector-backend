#!/usr/bin/env python3
"""
Data Preparation Agent - Phase 1/2
Jailbreak Prompt Detection (Iqra University DS course)

Loads jailbreak_train.csv / jailbreak_test.csv, profiles them, cleans the
prompt text (lowercase, collapse whitespace, drop empty prompts, drop exact
duplicates), and writes cleaned_train.csv / cleaned_test.csv.
"""
import os
import re
import sys

import pandas as pd

WORKDIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IN = f"{WORKDIR}/jailbreak_train.csv"
TEST_IN = f"{WORKDIR}/jailbreak_test.csv"
TRAIN_OUT = f"{WORKDIR}/cleaned_train.csv"
TEST_OUT = f"{WORKDIR}/cleaned_test.csv"


def clean_prompt(text: str) -> str:
    """Lowercase and collapse all whitespace (preserve punctuation/phrasing)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def profile(df: pd.DataFrame, name: str) -> None:
    """Print a structural profile for a dataframe."""
    print(f"\n{'=' * 70}")
    print(f"PROFILE: {name}")
    print(f"{'=' * 70}")
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {list(df.columns)}")
    print(f"Dtypes         :")
    print(df.dtypes.to_string())
    print(f"\nNulls per column:")
    print(df.isnull().sum().to_string())
    if "type" in df.columns:
        print(f"\nClass balance ('type'):")
        print(df["type"].value_counts(dropna=False).to_string())
        print(f"Unique labels: {sorted(df['type'].dropna().unique())}")


def process(path: str, out_path: str, name: str) -> None:
    print(f"\n\n########## Processing {name} ##########")
    df = pd.read_csv(path, encoding="utf-8")
    n_before = len(df)
    profile(df, f"{name} (RAW)")

    # --- Drop exact duplicates at the RAW level (same text + same label) ---
    # Kept as separate accounting so the breakdown below is non-overlapping.
    df_raw_dedup = df.drop_duplicates(subset=["prompt", "type"])
    n_raw_dupes = len(df) - len(df_raw_dedup)
    df = df_raw_dedup

    # --- Clean the prompt text ---------------------------------------------
    df["prompt"] = df["prompt"].apply(clean_prompt)

    # --- Drop empty prompts (after cleaning) -------------------------------
    n_empty = int((df["prompt"].str.strip() == "").sum())
    df = df[df["prompt"].str.strip() != ""].copy()

    # --- Drop exact duplicates that emerge AFTER normalization -------------
    # e.g. rows that differed only by case or whitespace before cleaning.
    df_clean_dedup = df.drop_duplicates(subset=["prompt", "type"])
    n_norm_dupes = len(df) - len(df_clean_dedup)
    df = df_clean_dedup

    # --- Report label conflicts (same prompt, different labels) ------------
    prompt_counts = df["prompt"].value_counts()
    n_conflicts = int((prompt_counts > 1).sum())
    if n_conflicts:
        print(
            f"\nNOTE: {n_conflicts} cleaned prompt(s) appear with BOTH labels "
            "(kept as-is; flagged for the modeling agent)."
        )

    # --- Reset index and save ----------------------------------------------
    df = df.reset_index(drop=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    n_after = len(df)
    dropped = n_before - n_after

    print(f"\n{'=' * 70}")
    print(f"CLEAN SUMMARY: {name}")
    print(f"{'=' * 70}")
    print(f"Rows before cleaning : {n_before}")
    print(f"Rows after cleaning  : {n_after}")
    print(f"Rows dropped         : {dropped}")
    print(f"  - exact duplicates removed (raw)   : {n_raw_dupes}")
    print(f"  - duplicates after normalization   : {n_norm_dupes}")
    print(f"  - empty prompts removed            : {n_empty}")
    print(f"\nClass balance AFTER cleaning:")
    print(df["type"].value_counts().to_string())
    print(f"Saved -> {out_path}")
    print(f"\nSample cleaned prompts:")
    print(df["prompt"].head(3).to_string())
    return n_before, n_after


def main() -> None:
    print(f"Working directory: {WORKDIR}")
    print(f"pandas version   : {pd.__version__}")

    train_before, train_after = process(TRAIN_IN, TRAIN_OUT, "TRAIN")
    test_before, test_after = process(TEST_IN, TEST_OUT, "TEST")

    print(f"\n\n{'#' * 70}")
    print("FINAL SUMMARY")
    print(f"{'#' * 70}")
    print(f"TRAIN: {train_before} -> {train_after} rows "
          f"({train_before - train_after} dropped)")
    print(f"TEST : {test_before} -> {test_after} rows "
          f"({test_before - test_after} dropped)")
    print(f"\nOutput files:")
    print(f"  {TRAIN_OUT}")
    print(f"  {TEST_OUT}")


if __name__ == "__main__":
    sys.exit(main())
