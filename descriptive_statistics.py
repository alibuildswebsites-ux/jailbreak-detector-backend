#!/usr/bin/env python3
"""Lecture 6 descriptive-statistics analysis for the jailbreak dataset.

The analysis is intentionally separate from the model-training pipeline. It
uses the cleaned training split as the primary descriptive population and
reports overall and class-wise statistics. No test information is used to fit
or select a model.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "cleaned_train.csv"
OUT = BASE / "output" / "descriptive_statistics"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["word_count", "char_count"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["word_count"] = out["prompt"].fillna("").str.split().str.len()
    out["char_count"] = out["prompt"].fillna("").str.len()
    return out


def trimmed_mean(s: pd.Series, proportion: float) -> float:
    x = np.sort(s.dropna().to_numpy(dtype=float))
    k = int(np.floor(proportion * len(x)))
    if 2 * k >= len(x):
        return float("nan")
    return float(x[k:len(x) - k].mean())


def descriptive_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", df)] + [(str(c), g) for c, g in df.groupby("type", sort=True)]
    for group_name, g in groups:
        for feature in FEATURES:
            s = g[feature].astype(float)
            q1, q3 = s.quantile(.25), s.quantile(.75)
            mean = s.mean()
            rows.append({
                "group": group_name,
                "feature": feature,
                "count": int(s.count()),
                "mean": mean,
                "median": s.median(),
                "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
                "trimmed_mean_10pct": trimmed_mean(s, .10),
                "trimmed_mean_20pct": trimmed_mean(s, .20),
                "min": s.min(),
                "max": s.max(),
                "range": s.max() - s.min(),
                "variance_sample": s.var(ddof=1),
                "std_sample": s.std(ddof=1),
                "coefficient_variation_pct": (s.std(ddof=1) / mean * 100) if mean else np.nan,
                "q1_25pct": q1,
                "q2_50pct": s.quantile(.50),
                "q3_75pct": q3,
                "iqr": q3 - q1,
                "p10": s.quantile(.10),
                "p90": s.quantile(.90),
                "p95": s.quantile(.95),
                "skewness": s.skew(),
                "kurtosis": s.kurt(),
            })
    return pd.DataFrame(rows)


def zscore_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        s = df[feature].astype(float)
        z = zscore(s, ddof=0)
        rows.append({
            "feature": feature,
            "mean_z": float(np.mean(z)),
            "std_z": float(np.std(z, ddof=0)),
            "min_z": float(np.min(z)),
            "max_z": float(np.max(z)),
            "abs_z_gt_2": int(np.sum(np.abs(z) > 2)),
            "abs_z_gt_3": int(np.sum(np.abs(z) > 3)),
            "abs_z_gt_3_rate": float(np.mean(np.abs(z) > 3)),
        })
    return pd.DataFrame(rows)


def estimated_kde_modality(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate visible KDE peak counts; report as exploratory, not definitive."""
    rows = []
    for feature in FEATURES:
        for group_name, g in [("overall", df)] + [(str(c), x) for c, x in df.groupby("type", sort=True)]:
            x = g[feature].astype(float).to_numpy()
            # KDE needs variation; fall back to one peak for a constant series.
            if len(np.unique(x)) < 2:
                peaks = 1
            else:
                kde = gaussian_kde(x)
                grid = np.linspace(np.percentile(x, .5), np.percentile(x, 99.5), 500)
                density = kde(grid)
                prominence = max(np.max(density) * .02, 1e-12)
                peaks = int(len(find_peaks(density, prominence=prominence)[0]))
                peaks = max(peaks, 1)
            label = {1: "approximately unimodal", 2: "approximately bimodal"}.get(peaks, "approximately multimodal")
            rows.append({"group": group_name, "feature": feature, "estimated_kde_peaks": peaks, "interpretation": label})
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    df = add_features(pd.read_csv(TRAIN))

    # Lecture 6: pandas describe() — numeric engineered variables.
    describe = df[FEATURES].describe().T.reset_index().rename(columns={"index": "feature"})
    describe.to_csv(OUT / "pandas_describe.csv", index=False)

    stats = descriptive_rows(df)
    stats.to_csv(OUT / "descriptive_statistics.csv", index=False)

    percentiles = stats[["group", "feature", "p10", "q1_25pct", "q2_50pct", "q3_75pct", "p90", "p95", "iqr"]].copy()
    percentiles.to_csv(OUT / "percentiles_quartiles_iqr.csv", index=False)

    zscores = zscore_summary(df)
    zscores.to_csv(OUT / "zscore_summary.csv", index=False)

    modality = estimated_kde_modality(df)
    modality.to_csv(OUT / "modality_kde_summary.csv", index=False)

    # Frequency distributions / summary tables from Lecture 6.
    length_bins = pd.cut(df["word_count"], bins=[-np.inf, 50, 250, np.inf], labels=["Short (≤50)", "Medium (51–250)", "Long (>250)"])
    freq = pd.crosstab(length_bins, df["type"], margins=True)
    freq.to_csv(OUT / "frequency_table_length_bins.csv")

    class_freq = df["type"].value_counts().rename_axis("type").reset_index(name="frequency")
    class_freq["percentage"] = class_freq["frequency"] / len(df) * 100
    class_freq.to_csv(OUT / "frequency_table_class.csv", index=False)

    # A compact human-readable interpretation, explicitly tied to the lecture concepts.
    overall_wc = stats[(stats.group == "overall") & (stats.feature == "word_count")].iloc[0]
    overall_cc = stats[(stats.group == "overall") & (stats.feature == "char_count")].iloc[0]
    lines = [
        "# Lecture 6 — Descriptive Statistics",
        "",
        f"Source: `cleaned_train.csv` ({len(df):,} training prompts). The official test split is not used in these descriptive calculations.",
        "",
        "## Central tendency",
        f"- Word count: mean={overall_wc['mean']:.2f}, median={overall_wc['median']:.2f}, mode={overall_wc['mode']:.0f}.",
        f"- Character count: mean={overall_cc['mean']:.2f}, median={overall_cc['median']:.2f}, mode={overall_cc['mode']:.0f}.",
        f"- The mean is well above the median for both features, consistent with strong right-skew and a long upper tail.",
        f"- 10% trimmed means: word_count={overall_wc['trimmed_mean_10pct']:.2f}; char_count={overall_cc['trimmed_mean_10pct']:.2f}.",
        f"- 20% trimmed means: word_count={overall_wc['trimmed_mean_20pct']:.2f}; char_count={overall_cc['trimmed_mean_20pct']:.2f}.",
        "",
        "## Dispersion",
        f"- Word count: range={overall_wc['range']:.0f}, sample variance={overall_wc['variance_sample']:.2f}, sample SD={overall_wc['std_sample']:.2f}, CV={overall_wc['coefficient_variation_pct']:.2f}%, IQR={overall_wc['iqr']:.2f}.",
        f"- Character count: range={overall_cc['range']:.0f}, sample variance={overall_cc['variance_sample']:.2f}, sample SD={overall_cc['std_sample']:.2f}, CV={overall_cc['coefficient_variation_pct']:.2f}%, IQR={overall_cc['iqr']:.2f}.",
        "",
        "## Position and z-scores",
        "- Q1, Q2/median, Q3, P10, P90 and P95 are provided in `percentiles_quartiles_iqr.csv`.",
        "- `zscore_summary.csv` reports the number and rate of observations beyond |Z|>2 and |Z|>3. The ±3 threshold is the lecture's typical outlier flag.",
        "",
        "## Shape and distribution",
        f"- Word-count skewness={overall_wc['skewness']:.2f}, kurtosis={overall_wc['kurtosis']:.2f}.",
        f"- Character-count skewness={overall_cc['skewness']:.2f}, kurtosis={overall_cc['kurtosis']:.2f}.",
        "- Both variables are strongly right-skewed and heavy-tailed rather than approximately normal.",
        "- Modality is reported using exploratory KDE peak counting in `modality_kde_summary.csv`; it should be interpreted as visual/EDA evidence rather than a formal hypothesis test.",
        "",
        "## Applicability decisions",
        "- A line plot is not used for the dataset because it has no time/order variable for a meaningful trend; the lecture describes line plots as generally useful for trends over time.",
        "- Mean/variance/SD use sample formulas (`ddof=1`) because the project treats the supplied training observations as a sample for statistical estimation. `pandas describe()` is also provided for reference.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Lecture 6 descriptive-statistics artifacts written to {OUT}")


if __name__ == "__main__":
    main()
