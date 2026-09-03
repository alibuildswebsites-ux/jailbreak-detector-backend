#!/usr/bin/env python3
"""Lecture 7 inferential-statistics analysis for the jailbreak dataset.

All inferential analyses use the cleaned training split as the sample. The
official test split is not used for statistical inference. The script follows
the lecture topics: interval estimation, bootstrapping, hypothesis testing,
correlation, chi-square tests, and regression. Analyses that are not
scientifically applicable to independent prompt observations are documented
rather than fabricated.
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    mannwhitneyu,
    norm,
    shapiro,
    levene,
    pearsonr,
    spearmanr,
    t,
    ttest_ind,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "cleaned_train.csv"
OUT = BASE / "output" / "inferential_statistics"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BOOTSTRAP_RESAMPLES = 5000
ALPHA = 0.05
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["prompt"].fillna("").astype(str)
    out["word_count"] = text.str.split().str.len()
    out["char_count"] = text.str.len()
    out["punctuation_count"] = text.str.count(r"[^\w\s]")
    out["uppercase_count"] = text.str.count(r"[A-Z]")
    out["digit_count"] = text.str.count(r"\d")
    out["newline_count"] = text.str.count("\n")
    out["exclamation_count"] = text.str.count("!")
    out["question_count"] = text.str.count(r"\?")
    out["uppercase_ratio"] = out["uppercase_count"] / out["char_count"].replace(0, np.nan)
    out["uppercase_ratio"] = out["uppercase_ratio"].fillna(0.0)
    out["punctuation_ratio"] = out["punctuation_count"] / out["word_count"].replace(0, np.nan)
    out["punctuation_ratio"] = out["punctuation_ratio"].fillna(0.0)
    out["length_bin"] = pd.cut(
        out["word_count"],
        bins=[-np.inf, 50, 250, np.inf],
        labels=["Short (≤50)", "Medium (51–250)", "Long (>250)"],
    )
    keywords = ["dan", "ignore", "previous instructions", "no restrictions", "roleplay", "unfiltered"]
    lowered = text.str.lower()
    out["any_keyword"] = lowered.apply(lambda s: int(any(k in s for k in keywords)))
    return out


def ci_mean(s: pd.Series, confidence: float, method: str):
    x = s.dropna().astype(float).to_numpy()
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    if method == "t":
        critical = float(t.ppf((1 + confidence) / 2, n - 1))
    elif method == "z_plugin":
        # Lecture's known-population-sigma formula, demonstrated with the
        # sample SD as a planning plug-in. This is NOT a known-sigma case.
        critical = float(norm.ppf((1 + confidence) / 2))
    else:
        raise ValueError(method)
    margin = critical * se
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "confidence": confidence,
        "critical_value": critical,
        "margin_of_error": margin,
        "lower": mean - margin,
        "upper": mean + margin,
        "method": method,
    }


def proportion_ci(successes: int, n: int, confidence: float):
    p_hat = successes / n
    zc = float(norm.ppf((1 + confidence) / 2))
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margin = zc * se
    return {
        "successes": successes,
        "n": n,
        "sample_proportion": p_hat,
        "confidence": confidence,
        "z_critical": zc,
        "standard_error": se,
        "margin_of_error": margin,
        "lower": p_hat - margin,
        "upper": p_hat + margin,
        "method": "normal approximation",
    }


def sample_size_mean(sigma: float, margin: float, confidence: float):
    zc = float(norm.ppf((1 + confidence) / 2))
    raw = (zc * sigma / margin) ** 2
    return {"parameter": "mean", "sigma_planning": sigma, "margin_of_error": margin,
            "confidence": confidence, "z_critical": zc, "raw_n": raw,
            "minimum_integer_n": int(np.ceil(raw))}


def sample_size_proportion(p: float, margin: float, confidence: float):
    zc = float(norm.ppf((1 + confidence) / 2))
    raw = p * (1 - p) * (zc / margin) ** 2
    return {"parameter": "proportion", "p_planning": p, "margin_of_error": margin,
            "confidence": confidence, "z_critical": zc, "raw_n": raw,
            "minimum_integer_n": int(np.ceil(raw))}


def bootstrap_mean(s: pd.Series, group: str, rng: np.random.Generator):
    x = s.dropna().astype(float).to_numpy()
    # Vectorized resampling keeps the lecture's repeated sampling-with-
    # replacement procedure exact while making 5,000 resamples inexpensive.
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_RESAMPLES, len(x)))
    means = x[indices].mean(axis=1)
    rows = []
    for confidence in (0.90, 0.95):
        lower_q = (1 - confidence) / 2 * 100
        upper_q = (1 + confidence) / 2 * 100
        rows.append({
            "group": group,
            "statistic": "mean",
            "n": len(x),
            "resamples": BOOTSTRAP_RESAMPLES,
            "sample_mean": float(np.mean(x)),
            "bootstrap_mean": float(np.mean(means)),
            "bootstrap_sd": float(np.std(means, ddof=1)),
            "confidence": confidence,
            "lower": float(np.percentile(means, lower_q)),
            "upper": float(np.percentile(means, upper_q)),
            "method": "percentile bootstrap",
        })
    return rows, means


def add_test(rows, name, family, null, alternative, statistic, p_value, decision, conclusion):
    rows.append({
        "test": name,
        "family": family,
        "null_hypothesis": null,
        "alternative_hypothesis": alternative,
        "alpha": ALPHA,
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "decision": decision,
        "conclusion": conclusion,
    })


def decision(p):
    return "Reject H0" if p <= ALPHA else "Fail to reject H0"


def ols_fit(y, X, names):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    fitted = X @ beta
    residuals = y - fitted
    df_resid = n - k
    sse = float(residuals @ residuals)
    mse = sse / df_resid
    cov_beta = mse * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 0))
    t_stats = beta / se
    p_values = 2 * t.sf(np.abs(t_stats), df_resid)
    critical = float(t.ppf(0.975, df_resid))
    lower = beta - critical * se
    upper = beta + critical * se
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = 1 - sse / sst
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / df_resid
    return {
        "beta": beta, "fitted": fitted, "residuals": residuals, "se": se,
        "t_stats": t_stats, "p_values": p_values, "lower": lower, "upper": upper,
        "df_resid": df_resid, "sse": sse, "r2": r2, "adjusted_r2": adjusted_r2,
        "names": names,
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    df = add_features(pd.read_csv(TRAIN))

    # ------------------------------------------------------------------
    # 1. Interval estimation: t intervals are the primary project method.
    # ------------------------------------------------------------------
    ci_rows = []
    for group, g in [("Overall", df), ("Benign", df[df.type == "benign"]),
                     ("Jailbreak", df[df.type == "jailbreak"])]:
        for feature in ["word_count", "char_count"]:
            for confidence in CONFIDENCE_LEVELS:
                row = ci_mean(g[feature], confidence, "t")
                row.update({"group": group, "feature": feature,
                            "assumption_note": "Population SD unknown; t interval used."})
                ci_rows.append(row)
    pd.DataFrame(ci_rows).to_csv(OUT / "confidence_intervals.csv", index=False)

    # Lecture z-interval formula comparison, explicitly labeled as a plug-in
    # demonstration because the project has no known population sigma.
    z_rows = []
    for group, g in [("Overall", df), ("Benign", df[df.type == "benign"]),
                     ("Jailbreak", df[df.type == "jailbreak"])]:
        for feature in ["word_count", "char_count"]:
            for confidence in (0.90, 0.95):
                row = ci_mean(g[feature], confidence, "z_plugin")
                row.update({"group": group, "feature": feature,
                            "assumption_note": "Lecture z formula demonstrated with sample SD as plug-in; not a known-sigma CI."})
                z_rows.append(row)
    pd.DataFrame(z_rows).to_csv(OUT / "z_interval_formula_demonstration.csv", index=False)

    # Compare widths directly for the primary t intervals.
    comparison = pd.DataFrame(ci_rows)
    comparison["width"] = comparison["upper"] - comparison["lower"]
    comparison[["group", "feature", "confidence", "lower", "upper", "width", "method"]].to_csv(
        OUT / "confidence_interval_comparison.csv", index=False
    )

    # ------------------------------------------------------------------
    # 2. Proportion interval + sample-size planning.
    # ------------------------------------------------------------------
    successes = int((df.type == "jailbreak").sum())
    prop_rows = [proportion_ci(successes, len(df), c) for c in CONFIDENCE_LEVELS]
    pd.DataFrame(prop_rows).to_csv(OUT / "proportion_confidence_interval.csv", index=False)

    planning_rows = []
    planning_sigma = float(df["word_count"].std(ddof=1))
    for c in (0.90, 0.95):
        planning_rows.append(sample_size_mean(planning_sigma, margin=10, confidence=c))
    # Observed p is a planning estimate; p=0.50 is the conservative maximum-
    # variance choice and is also reported.
    for p_plan, label in [(successes / len(df), "observed proportion"), (0.50, "conservative p=0.50")]:
        for c in (0.90, 0.95):
            row = sample_size_proportion(p_plan, margin=0.05, confidence=c)
            row["planning_basis"] = label
            planning_rows.append(row)
    pd.DataFrame(planning_rows).to_csv(OUT / "sample_size_determination.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Bootstrap percentile intervals (5,000 resamples per group).
    # ------------------------------------------------------------------
    boot_summary = []
    boot_distribution_rows = []
    for group, g in [("Overall", df), ("Benign", df[df.type == "benign"]),
                      ("Jailbreak", df[df.type == "jailbreak"])]:
        summary_rows, means = bootstrap_mean(g["word_count"], group, rng)
        boot_summary.extend(summary_rows)
        boot_distribution_rows.extend(
            {"group": group, "resample": i + 1, "bootstrap_mean": float(v)}
            for i, v in enumerate(means)
        )
    pd.DataFrame(boot_summary).to_csv(OUT / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(boot_distribution_rows).to_csv(OUT / "bootstrap_mean_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    boot_df = pd.DataFrame(boot_distribution_rows)
    for group in ["Overall", "Benign", "Jailbreak"]:
        vals = boot_df.loc[boot_df.group == group, "bootstrap_mean"]
        ax.hist(vals, bins=45, alpha=0.45, label=group)
    ax.set_title("Bootstrap Sampling Distributions — Mean Word Count")
    ax.set_xlabel("Bootstrap mean word count")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_distribution_word_count.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Assumption diagnostics: normality and variance homogeneity.
    # These diagnostics inform the choice of parametric vs non-parametric
    # tests; with large, visibly skewed samples, the results are not used
    # as an automatic gatekeeper for the inferential conclusions.
    # ------------------------------------------------------------------
    diagnostic_rows = []
    for group, g in [("Benign", df[df.type == "benign"]), ("Jailbreak", df[df.type == "jailbreak"])]:
        stat, p = shapiro(g["word_count"].astype(float))
        diagnostic_rows.append({"diagnostic": "Shapiro-Wilk normality", "variable": "word_count", "group": group, "statistic": stat, "p_value": p, "alpha": ALPHA, "interpretation": "Reject normality" if p <= ALPHA else "No evidence against normality"})
    for label in ["Short (≤50)", "Medium (51–250)", "Long (>250)"]:
        vals = df.loc[df.length_bin == label, "punctuation_ratio"].astype(float)
        stat, p = shapiro(vals)
        diagnostic_rows.append({"diagnostic": "Shapiro-Wilk normality", "variable": "punctuation_ratio", "group": label, "statistic": stat, "p_value": p, "alpha": ALPHA, "interpretation": "Reject normality" if p <= ALPHA else "No evidence against normality"})
    stat, p = levene(*[df.loc[df.length_bin == label, "punctuation_ratio"].astype(float).to_numpy() for label in ["Short (≤50)", "Medium (51–250)", "Long (>250)"]])
    diagnostic_rows.append({"diagnostic": "Levene variance-homogeneity test", "variable": "punctuation_ratio", "group": "Short/Medium/Long", "statistic": stat, "p_value": p, "alpha": ALPHA, "interpretation": "Reject equal-variance assumption" if p <= ALPHA else "No evidence against equal variance"})
    pd.DataFrame(diagnostic_rows).to_csv(OUT / "assumption_diagnostics.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Hypothesis testing: independent groups.
    # ------------------------------------------------------------------
    benign_wc = df.loc[df.type == "benign", "word_count"].astype(float).to_numpy()
    jailbreak_wc = df.loc[df.type == "jailbreak", "word_count"].astype(float).to_numpy()
    test_rows = []

    # Two-tailed difference in means.
    stat, p = ttest_ind(jailbreak_wc, benign_wc, equal_var=False, alternative="two-sided")
    add_test(test_rows, "Welch independent t-test (two-tailed)", "Independent two-group parametric",
             "μ_jailbreak = μ_benign", "μ_jailbreak ≠ μ_benign", stat, p, decision(p),
             "Mean word counts differ significantly if H0 is rejected.")
    # Directional safety-oriented hypothesis.
    stat, p = ttest_ind(jailbreak_wc, benign_wc, equal_var=False, alternative="greater")
    add_test(test_rows, "Welch independent t-test (one-tailed)", "Independent two-group parametric",
             "μ_jailbreak ≤ μ_benign", "μ_jailbreak > μ_benign", stat, p, decision(p),
             "Provides evidence for whether jailbreak prompts have greater mean word count.")

    # Non-parametric counterpart.
    stat, p = mannwhitneyu(jailbreak_wc, benign_wc, alternative="two-sided")
    add_test(test_rows, "Mann–Whitney U test (two-tailed)", "Independent two-group non-parametric",
             "The two class distributions have the same location", "The class distributions differ in location",
             stat, p, decision(p), "Tests whether the two independent prompt-length distributions differ.")
    stat, p = mannwhitneyu(jailbreak_wc, benign_wc, alternative="greater")
    add_test(test_rows, "Mann–Whitney U test (one-tailed)", "Independent two-group non-parametric",
             "Jailbreak word-count distribution is not shifted higher", "Jailbreak word-count distribution is shifted higher",
             stat, p, decision(p), "Tests the directional distributional difference without normality assumptions.")

    # Paired tests are not applicable to this independent-observation dataset.
    test_rows.append({
        "test": "Paired t-test", "family": "Paired two-group parametric", "null_hypothesis": "N/A",
        "alternative_hypothesis": "N/A", "alpha": ALPHA, "test_statistic": np.nan, "p_value": np.nan,
        "decision": "Not applicable", "conclusion": "No matched before/after or naturally paired observations exist."
    })
    test_rows.append({
        "test": "Wilcoxon signed-rank test", "family": "Paired two-group non-parametric", "null_hypothesis": "N/A",
        "alternative_hypothesis": "N/A", "alpha": ALPHA, "test_statistic": np.nan, "p_value": np.nan,
        "decision": "Not applicable", "conclusion": "No matched before/after or naturally paired observations exist."
    })

    # ------------------------------------------------------------------
    # 6. Three-group tests: punctuation ratio across independent length bins.
    # The groups are defined by word count, while the tested outcome is
    # punctuation ratio, avoiding circularly testing word count against its own bins.
    # ------------------------------------------------------------------
    groups = [df.loc[df.length_bin == label, "punctuation_ratio"].to_numpy()
              for label in ["Short (≤50)", "Medium (51–250)", "Long (>250)"]]
    stat, p = f_oneway(*groups)
    add_test(test_rows, "One-way ANOVA: punctuation ratio by length bin", "Three-or-more-group parametric",
             "All length-bin group means are equal", "At least one group mean differs", stat, p, decision(p),
             "Tests whether mean punctuation ratio differs among the three prompt-length groups.")
    stat, p = kruskal(*groups)
    add_test(test_rows, "Kruskal–Wallis: punctuation ratio by length bin", "Three-or-more-group non-parametric",
             "All length-bin groups have the same distribution/location", "At least one group differs", stat, p, decision(p),
             "Tests the same three-group question without normality assumptions.")

    pd.DataFrame(test_rows).to_csv(OUT / "hypothesis_tests.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Pearson + Spearman inferential correlation.
    # ------------------------------------------------------------------
    x = df["word_count"].astype(float)
    y = df["char_count"].astype(float)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    corr_rows = [
        {"method": "Pearson", "coefficient": pearson_r, "p_value": pearson_p,
         "n": len(df), "df": len(df) - 2, "alpha": ALPHA,
         "null_hypothesis": "ρ = 0", "alternative_hypothesis": "ρ ≠ 0",
         "decision": decision(pearson_p),
         "interpretation": "Tests linear association between word count and character count."},
        {"method": "Spearman", "coefficient": spearman_rho, "p_value": spearman_p,
         "n": len(df), "df": np.nan, "alpha": ALPHA,
         "null_hypothesis": "ρ_s = 0", "alternative_hypothesis": "ρ_s ≠ 0",
         "decision": decision(spearman_p),
         "interpretation": "Tests monotonic association using ranks; useful because prompt lengths are strongly non-normal."},
    ]
    pd.DataFrame(corr_rows).to_csv(OUT / "correlation_significance_tests.csv", index=False)

    # Scatter with simple regression line is created after regression below.

    # ------------------------------------------------------------------
    # 8. Chi-square independence: keyword presence vs class.
    # ------------------------------------------------------------------
    table = pd.crosstab(df["any_keyword"].map({0: "Absent", 1: "Present"}), df["type"])
    table = table.reindex(index=["Absent", "Present"], columns=["benign", "jailbreak"], fill_value=0)
    chi2_stat, chi2_p, chi2_dof, expected = chi2_contingency(table, correction=False)
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    table.to_csv(OUT / "chi_square_contingency_table.csv")
    expected_df.to_csv(OUT / "chi_square_expected_frequencies.csv")
    chi_square_result = pd.DataFrame([{
        "test": "Chi-square test of independence",
        "variables": "Selected keyword presence × prompt class",
        "chi_square": chi2_stat,
        "degrees_of_freedom": chi2_dof,
        "p_value": chi2_p,
        "alpha": ALPHA,
        "decision": decision(chi2_p),
        "null_hypothesis": "Keyword presence and prompt class are independent.",
        "alternative_hypothesis": "Keyword presence and prompt class are associated.",
        "min_expected_frequency": float(expected.min()),
        "assumptions_satisfied": bool(expected.min() >= 5),
        "conclusion": "There is evidence of association between keyword presence and class if H0 is rejected."
    }])
    chi_square_result.to_csv(OUT / "chi_square_test.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Simple linear regression: character count from word count.
    # ------------------------------------------------------------------
    X_simple = np.column_stack([np.ones(len(df)), df["word_count"].to_numpy(dtype=float)])
    y_char = df["char_count"].to_numpy(dtype=float)
    simple = ols_fit(y_char, X_simple, ["intercept", "word_count"])
    simple_rows = []
    for i, name in enumerate(simple["names"]):
        simple_rows.append({
            "model": "simple_linear_regression",
            "term": name,
            "coefficient": simple["beta"][i],
            "standard_error": simple["se"][i],
            "t_statistic": simple["t_stats"][i],
            "p_value": simple["p_values"][i],
            "ci95_lower": simple["lower"][i],
            "ci95_upper": simple["upper"][i],
            "df_residual": simple["df_resid"],
            "r_squared": simple["r2"],
            "adjusted_r_squared": simple["adjusted_r2"],
        })
    pd.DataFrame(simple_rows).to_csv(OUT / "simple_regression_summary.csv", index=False)

    median_words = float(df["word_count"].median())
    prediction = float(simple["beta"][0] + simple["beta"][1] * median_words)
    pd.DataFrame([{
        "prediction_target": "character_count",
        "predictor": "word_count",
        "word_count_input": median_words,
        "predicted_character_count": prediction,
        "equation": f"char_count = {simple['beta'][0]:.4f} + {simple['beta'][1]:.4f} × word_count"
    }]).to_csv(OUT / "simple_regression_prediction.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["word_count"], df["char_count"], alpha=0.35, s=18, label="Prompts")
    x_line = np.linspace(float(df.word_count.min()), float(df.word_count.max()), 200)
    y_line = simple["beta"][0] + simple["beta"][1] * x_line
    ax.plot(x_line, y_line, linewidth=2, label="Least-squares line")
    ax.set_title("Simple Linear Regression — Character Count vs Word Count")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Character count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "simple_regression_best_fit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(simple["fitted"], simple["residuals"], alpha=0.4, s=18)
    ax.axhline(0, linewidth=1)
    ax.set_title("Simple Regression Residuals")
    ax.set_xlabel("Fitted character count")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(OUT / "simple_regression_residuals.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 10. Multiple linear regression: character count from independent text
    # characteristics. Word count is retained as the main predictor; the
    # remaining predictors capture additional formatting information.
    # ------------------------------------------------------------------
    predictors = ["word_count", "punctuation_count", "digit_count", "exclamation_count", "question_count"]
    X_multi = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in predictors])
    multi = ols_fit(y_char, X_multi, ["intercept"] + predictors)
    multi_rows = []
    for i, name in enumerate(multi["names"]):
        multi_rows.append({
            "model": "multiple_linear_regression",
            "term": name,
            "coefficient": multi["beta"][i],
            "standard_error": multi["se"][i],
            "t_statistic": multi["t_stats"][i],
            "p_value": multi["p_values"][i],
            "ci95_lower": multi["lower"][i],
            "ci95_upper": multi["upper"][i],
            "df_residual": multi["df_resid"],
            "r_squared": multi["r2"],
            "adjusted_r_squared": multi["adjusted_r2"],
        })
    pd.DataFrame(multi_rows).to_csv(OUT / "multiple_regression_summary.csv", index=False)
    pd.DataFrame([{"model": "multiple_linear_regression", "multiple_correlation_R": float(np.sqrt(max(multi["r2"], 0))), "r_squared": multi["r2"], "adjusted_r_squared": multi["adjusted_r2"]}]).to_csv(OUT / "multiple_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    coef_df = pd.DataFrame({"term": multi["names"], "coefficient": multi["beta"]})
    ax.barh(coef_df["term"], coef_df["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Multiple Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Predictor")
    fig.tight_layout()
    fig.savefig(OUT / "multiple_regression_coefficients.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 11. Applicability matrix + human-readable summary.
    # ------------------------------------------------------------------
    applicability = pd.DataFrame([
        {"lecture_topic": "Point estimation", "status": "Implemented", "artifact": "confidence_intervals.csv", "note": "Sample statistics estimate population parameters."},
        {"lecture_topic": "Z confidence interval", "status": "Demonstrated with limitation", "artifact": "z_interval_formula_demonstration.csv", "note": "Lecture known-sigma formula shown with sample SD as a plug-in; t CI is primary because population SD is unknown."},
        {"lecture_topic": "T confidence interval", "status": "Implemented", "artifact": "confidence_intervals.csv", "note": "Primary CI method for unknown population SD."},
        {"lecture_topic": "90/95/99% CI comparison", "status": "Implemented", "artifact": "confidence_interval_comparison.csv", "note": "Higher confidence produces wider intervals."},
        {"lecture_topic": "Proportion confidence interval", "status": "Implemented", "artifact": "proportion_confidence_interval.csv", "note": "Normal approximation used; sample is large and expected counts are adequate."},
        {"lecture_topic": "Sample-size determination", "status": "Implemented", "artifact": "sample_size_determination.csv", "note": "Mean and proportion planning calculations."},
        {"lecture_topic": "Bootstrapping", "status": "Implemented", "artifact": "bootstrap_confidence_intervals.csv", "note": "5,000 percentile bootstrap resamples with replacement."},
        {"lecture_topic": "Hypothesis testing", "status": "Implemented", "artifact": "hypothesis_tests.csv", "note": "H0/H1, alpha, statistic, p-value, decision and conclusion recorded."},
        {"lecture_topic": "Type I / Type II errors", "status": "Documented", "artifact": "summary.md", "note": "Statistical errors distinguished from classifier false positives/negatives."},
        {"lecture_topic": "Assumption diagnostics", "status": "Implemented", "artifact": "assumption_diagnostics.csv", "note": "Shapiro-Wilk normality and Levene variance-homogeneity diagnostics."},
        {"lecture_topic": "Independent t-test", "status": "Implemented", "artifact": "hypothesis_tests.csv", "note": "Welch test, two-tailed and one-tailed."},
        {"lecture_topic": "Mann–Whitney U", "status": "Implemented", "artifact": "hypothesis_tests.csv", "note": "Two-tailed and one-tailed."},
        {"lecture_topic": "Paired t-test", "status": "Not applicable", "artifact": "hypothesis_tests.csv", "note": "No paired observations."},
        {"lecture_topic": "Wilcoxon signed-rank", "status": "Not applicable", "artifact": "hypothesis_tests.csv", "note": "No paired observations."},
        {"lecture_topic": "One-way ANOVA", "status": "Implemented", "artifact": "hypothesis_tests.csv", "note": "Punctuation ratio compared across three word-length groups."},
        {"lecture_topic": "Kruskal–Wallis", "status": "Implemented", "artifact": "hypothesis_tests.csv", "note": "Non-parametric counterpart to the ANOVA question."},
        {"lecture_topic": "Multiple correlation", "status": "Implemented", "artifact": "multiple_correlation.csv", "note": "Multiple R reported from the multiple-regression R²."},
        {"lecture_topic": "Pearson correlation", "status": "Implemented", "artifact": "correlation_significance_tests.csv", "note": "Coefficient and significance test."},
        {"lecture_topic": "Spearman correlation", "status": "Implemented", "artifact": "correlation_significance_tests.csv", "note": "Rank-based correlation for non-normal prompt lengths."},
        {"lecture_topic": "Chi-square independence", "status": "Implemented", "artifact": "chi_square_test.csv", "note": "Keyword presence versus prompt class, with observed/expected tables."},
        {"lecture_topic": "Simple linear regression", "status": "Implemented", "artifact": "simple_regression_summary.csv", "note": "Least-squares character-count prediction from word count."},
        {"lecture_topic": "Regression best-fit line", "status": "Implemented", "artifact": "simple_regression_best_fit.png", "note": "Scatterplot with fitted line."},
        {"lecture_topic": "Regression residuals", "status": "Implemented", "artifact": "simple_regression_residuals.png", "note": "Residual diagnostic plot."},
        {"lecture_topic": "Regression prediction", "status": "Implemented", "artifact": "simple_regression_prediction.csv", "note": "Prediction at the sample median word count."},
        {"lecture_topic": "Multiple linear regression", "status": "Implemented", "artifact": "multiple_regression_summary.csv", "note": "OLS with five text-derived predictors."},
    ])
    applicability.to_csv(OUT / "lecture7_applicability_matrix.csv", index=False)

    # Statistical conclusions for report.
    t_ci_jb = [r for r in ci_rows if r["group"] == "Jailbreak" and r["feature"] == "word_count" and r["confidence"] == 0.95][0]
    t_ci_ben = [r for r in ci_rows if r["group"] == "Benign" and r["feature"] == "word_count" and r["confidence"] == 0.95][0]
    ttest_two = [r for r in test_rows if r["test"] == "Welch independent t-test (two-tailed)"][0]
    mw_two = [r for r in test_rows if r["test"] == "Mann–Whitney U test (two-tailed)"][0]
    chi_row = chi_square_result.iloc[0]
    simple_r2 = simple["r2"]
    lines = [
        "# Lecture 7 — Inferential Statistics",
        "",
        f"Source: `cleaned_train.csv` (N={len(df):,}). The official held-out test split is not used for inferential estimation or hypothesis testing.",
        "",
        "## 1. Interval estimation",
        "",
        "The project uses the t-distribution as the primary confidence-interval method because the population standard deviation is unknown. The lecture's known-sigma z formula is also demonstrated separately with the sample SD as a plug-in; that result is explicitly not treated as a known-sigma population CI.",
        f"For jailbreak prompts, the 95% CI for mean word count is {t_ci_jb['lower']:.2f} to {t_ci_jb['upper']:.2f} words. For benign prompts it is {t_ci_ben['lower']:.2f} to {t_ci_ben['upper']:.2f} words.",
        "The 90%, 95% and 99% intervals are stored in `confidence_interval_comparison.csv`; the higher-confidence intervals are wider because greater confidence requires a larger critical value and margin of error.",
        "",
        "## 2. Proportion and sample-size estimation",
        f"The sample jailbreak proportion is {successes / len(df):.4f} ({successes}/{len(df)}). Normal-approximation confidence intervals at 90%, 95% and 99% are provided in `proportion_confidence_interval.csv`.",
        "Sample-size planning is provided for a mean with a ±10-word margin of error and for a proportion with a ±5-percentage-point margin of error. The planning calculations use the training sample SD as an estimate of sigma and report both observed-p and conservative p=0.50 planning for proportions.",
        "",
        "## 3. Bootstrap inference",
        f"A percentile bootstrap with {BOOTSTRAP_RESAMPLES:,} resamples per group is used for mean word count. Each resample is drawn with replacement and has the original group sample size. 90% and 95% percentile intervals are stored in `bootstrap_confidence_intervals.csv`.",
        "",
        "## 4. Hypothesis testing",
        "### Independent groups",
        f"Welch's two-tailed t-test gives t={ttest_two['test_statistic']:.4f}, p={ttest_two['p_value']:.6g}. Decision: {ttest_two['decision']}. The one-tailed version tests the lecture-style directional claim that jailbreak prompts have greater mean word count.",
        f"The two-tailed Mann–Whitney U test gives U={mw_two['test_statistic']:.4f}, p={mw_two['p_value']:.6g}, providing a non-parametric comparison of the two independent distributions.",
        "Paired t-test and Wilcoxon signed-rank test are documented as not applicable because the dataset contains independent prompts, not matched pairs.",
        "",
        "### Three groups",
        "One-way ANOVA and Kruskal–Wallis compare punctuation ratio across the three prompt-length groups (Short, Medium, Long). The outcome is punctuation ratio, not word count itself, so the test is not circular.",
        "",
        "## 5. Correlation analysis",
        f"Pearson and Spearman significance tests are provided for word count versus character count. Pearson r={pearson_r:.6f} (p={pearson_p:.6g}); Spearman rho={spearman_rho:.6f} (p={spearman_p:.6g}). Spearman is particularly informative because the prompt-length variables are strongly non-normal.",
        "",
        "## 6. Chi-square independence",
        f"The chi-square test evaluates selected-keyword presence versus prompt class. χ²={chi2_stat:.4f}, df={chi2_dof}, p={chi2_p:.6g}; decision: {decision(chi2_p)}. The observed and expected contingency tables are saved separately, and the minimum expected frequency is {expected.min():.2f}.",
        "This is an inferential chi-square independence test and is distinct from the chi-square feature-selection procedure used earlier in the project.",
        "",
        "## 7. Simple linear regression",
        f"The fitted equation is `char_count = {simple['beta'][0]:.4f} + {simple['beta'][1]:.4f} × word_count`, with R²={simple_r2:.6f}. The slope p-value is {simple['p_values'][1]:.6g}.",
        f"At the sample median of {median_words:.0f} words, the fitted model predicts approximately {prediction:.2f} characters. The best-fit line and residual plot are saved in the output directory.",
        "",
        "## 8. Multiple linear regression",
        f"An OLS model predicts character count from word count, punctuation count, digit count, exclamation count and question count. The complete coefficient table, standard errors, t statistics, p-values and 95% coefficient intervals are in `multiple_regression_summary.csv`.",
        "",
        "## 9. Statistical error interpretation",
        "Type I error means rejecting a true null hypothesis; Type II error means failing to reject a false null hypothesis. These statistical-testing concepts are kept distinct from the classifier's false-positive and false-negative classification errors.",
        "",
        "## 10. Reproducibility",
        f"Random seed: {RANDOM_SEED}. Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}. Significance level α={ALPHA}. All calculations are generated by `inferential_statistics.py` using SciPy/Pandas/NumPy and all output tables are CSV files. Assumption diagnostics are in `assumption_diagnostics.csv`.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Lecture 7 inferential-statistics artifacts written to {OUT}")
    print(f"Welch t-test two-sided p={ttest_two['p_value']:.6g}; chi-square p={chi2_p:.6g}; Pearson p={pearson_p:.6g}; Spearman p={spearman_p:.6g}")


if __name__ == "__main__":
    main()
