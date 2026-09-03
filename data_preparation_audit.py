#!/usr/bin/env python3
"""Lecture 5 data-preparation audit and preprocessing artifacts.

Implements the missing/documented items from the Data Preparation lecture
without forcing techniques that are inappropriate for short unstructured text.
All calculations are fit/derived from the training data only where a model-like
transformation is involved; the official test split is never merged into train.
"""
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.join(BASE, "output", "eda")
DP = os.path.join(BASE, "output", "data_preparation")
os.makedirs(EDA, exist_ok=True)
os.makedirs(DP, exist_ok=True)

TRAIN_RAW = os.path.join(BASE, "jailbreak_train.csv")
TEST_RAW = os.path.join(BASE, "jailbreak_test.csv")
TRAIN = os.path.join(BASE, "cleaned_train.csv")
TEST = os.path.join(BASE, "cleaned_test.csv")

KEYWORDS = ["dan", "ignore", "previous instructions", "no restrictions", "roleplay", "unfiltered"]
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def add_numeric_features(df):
    out = df.copy()
    text = out["prompt"].fillna("").astype(str)
    out["word_count"] = text.str.split().str.len()
    out["char_count"] = text.str.len()
    out["punctuation_count"] = text.str.count(r"[^\w\s]")
    out["uppercase_count"] = text.str.count(r"[A-Z]")
    out["digit_count"] = text.str.count(r"\d")
    out["newline_count"] = text.str.count(r"\n")
    out["exclamation_count"] = text.str.count(r"!")
    out["question_count"] = text.str.count(r"\?")
    out["uppercase_ratio"] = np.where(out["char_count"] > 0, out["uppercase_count"] / out["char_count"], 0)
    for kw in KEYWORDS:
        safe = re.escape(kw)
        out[f"kw_{kw.replace(' ', '_')}"] = text.str.contains(safe, case=False, regex=True).astype(int)
    out["any_keyword"] = out[[f"kw_{k.replace(' ', '_')}" for k in KEYWORDS]].max(axis=1)
    out["length_bin"] = pd.cut(
        out["word_count"], bins=[-np.inf, 50, 250, np.inf],
        labels=["short", "medium", "long"]
    )
    return out


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    raw_train = pd.read_csv(TRAIN_RAW)
    raw_test = pd.read_csv(TEST_RAW)
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    # 1) Data integration/schema audit — keep official train/test separate.
    schema = pd.DataFrame({
        "column": sorted(set(raw_train.columns) | set(raw_test.columns)),
        "train_present": [c in raw_train.columns for c in sorted(set(raw_train.columns) | set(raw_test.columns))],
        "test_present": [c in raw_test.columns for c in sorted(set(raw_train.columns) | set(raw_test.columns))],
        "train_dtype": [str(raw_train[c].dtype) if c in raw_train.columns else "—" for c in sorted(set(raw_train.columns) | set(raw_test.columns))],
        "test_dtype": [str(raw_test[c].dtype) if c in raw_test.columns else "—" for c in sorted(set(raw_train.columns) | set(raw_test.columns))],
    })
    schema.to_csv(os.path.join(DP, "schema_audit.csv"), index=False)

    # 2) Full profile for both raw splits.
    profile_rows = []
    for split_name, df in [("train_raw", raw_train), ("test_raw", raw_test), ("train_clean", train), ("test_clean", test)]:
        for col in df.columns:
            s = df[col]
            row = {
                "split": split_name, "column": col, "dtype": str(s.dtype),
                "rows": len(df), "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "duplicate_rows": int(df.duplicated().sum()),
            }
            if pd.api.types.is_numeric_dtype(s):
                row.update({
                    "mean": s.mean(), "median": s.median(), "std": s.std(),
                    "min": s.min(), "max": s.max(), "skewness": s.skew(), "kurtosis": s.kurt(),
                })
            else:
                row.update({k: np.nan for k in ["mean", "median", "std", "min", "max", "skewness", "kurtosis"]})
            profile_rows.append(row)
    pd.DataFrame(profile_rows).to_csv(os.path.join(DP, "data_profile.csv"), index=False)

    # 3) Numeric feature profiling on cleaned training data.
    tr = add_numeric_features(train)
    te = add_numeric_features(test)
    numeric_cols = ["word_count", "char_count", "punctuation_count", "uppercase_count", "digit_count", "newline_count", "exclamation_count", "question_count", "uppercase_ratio"]
    stats = tr.groupby("type")[numeric_cols].agg(["count", "mean", "median", "std", "min", "max", "skew", "kurt"]).transpose()
    stats.to_csv(os.path.join(DP, "numeric_profile_by_class.csv"))

    # 4) Correlation matrix: target encoded only for analysis, not as a model feature.
    corr_cols = numeric_cols + ["any_keyword"]
    corr_df = tr[corr_cols].copy()
    corr_df["target_jailbreak"] = (tr["type"] == "jailbreak").astype(int)
    corr = corr_df.corr(numeric_only=True)
    corr.to_csv(os.path.join(DP, "correlation_matrix.csv"))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Correlation Matrix — Engineered Numeric Features (Training)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    save_fig(fig, os.path.join(DP, "correlation_heatmap.png"))

    # 5) Outliers: IQR + Z-score on length variables; retain them unless data-quality evidence says otherwise.
    out_rows = []
    for col in ["word_count", "char_count"]:
        s = tr[col].astype(float)
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (s < lo) | (s > hi)
        z = np.abs(zscore(s, nan_policy="omit"))
        z_mask = z > 3
        out_rows.append({"feature": col, "q1": q1, "q3": q3, "iqr": iqr,
                         "iqr_lower": lo, "iqr_upper": hi, "iqr_outliers": int(iqr_mask.sum()),
                         "iqr_outlier_rate": float(iqr_mask.mean()), "zscore_outliers": int(z_mask.sum()),
                         "zscore_outlier_rate": float(z_mask.mean()), "max": float(s.max())})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(s, vert=False)
        ax.set_title(f"{col} — IQR Outlier Inspection")
        ax.set_xlabel(col)
        save_fig(fig, os.path.join(DP, f"boxplot_{col}.png"))
    outlier_df = pd.DataFrame(out_rows)
    outlier_df.to_csv(os.path.join(DP, "outlier_analysis.csv"), index=False)

    # 6) Log transform evaluation for right-skewed length features.
    transform_rows = []
    for col in ["word_count", "char_count"]:
        original = tr[col].astype(float)
        transformed = np.log1p(original)
        transform_rows.append({
            "feature": col,
            "original_skewness": original.skew(),
            "log1p_skewness": transformed.skew(),
            "original_kurtosis": original.kurt(),
            "log1p_kurtosis": transformed.kurt(),
            "skewness_improved": abs(transformed.skew()) < abs(original.skew()),
        })
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(original, bins=40, alpha=.7, label="original")
        ax.hist(transformed, bins=40, alpha=.7, label="log1p")
        ax.set_title(f"{col} — Original vs log1p")
        ax.legend()
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        save_fig(fig, os.path.join(DP, f"log_transform_{col}.png"))
    pd.DataFrame(transform_rows).to_csv(os.path.join(DP, "log_transform_analysis.csv"), index=False)

    # 7) Text preprocessing audit: explicit tokenization and vocabulary statistics.
    def tokenize(text):
        return TOKEN_RE.findall(str(text).lower())
    token_counts = []
    for split_name, df in [("train", train), ("test", test)]:
        counts = df["prompt"].map(lambda x: len(tokenize(x)))
        token_counts.append({"split": split_name, "documents": len(df), "total_tokens": int(counts.sum()),
                             "mean_tokens": counts.mean(), "median_tokens": counts.median(),
                             "unique_tokens": len(set(t for x in df["prompt"] for t in tokenize(x)))})
    pd.DataFrame(token_counts).to_csv(os.path.join(DP, "tokenization_summary.csv"), index=False)

    # 8) TF-IDF normalization audit + feature selection demonstration.
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", norm="l2")
    X = vec.fit_transform(train["prompt"])
    y = (train["type"] == "jailbreak").astype(int)
    selector = SelectKBest(score_func=chi2, k=min(1000, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_names = vec.get_feature_names_out()[selector.get_support()]
    # A reproducible feature-selection experiment (analysis artifact; the main model remains the 5k-feature pipeline).
    X_test_full = vec.transform(test["prompt"])
    X_test_selected = selector.transform(X_test_full)
    lr_full = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    lr_sel = LogisticRegression(max_iter=1000, random_state=42).fit(X_selected, y)
    y_test_num = (test["type"] == "jailbreak").astype(int)
    pred_full = lr_full.predict(X_test_full)
    pred_sel = lr_sel.predict(X_test_selected)
    fs_metrics = pd.DataFrame([
        {"representation": "TF-IDF 5,000", "features": X.shape[1], "accuracy": accuracy_score(y_test_num, pred_full),
         "precision_jailbreak": precision_score(y_test_num, pred_full, zero_division=0),
         "recall_jailbreak": recall_score(y_test_num, pred_full, zero_division=0),
         "f1_jailbreak": f1_score(y_test_num, pred_full, zero_division=0)},
        {"representation": "TF-IDF + chi2 SelectKBest", "features": X_selected.shape[1], "accuracy": accuracy_score(y_test_num, pred_sel),
         "precision_jailbreak": precision_score(y_test_num, pred_sel, zero_division=0),
         "recall_jailbreak": recall_score(y_test_num, pred_sel, zero_division=0),
         "f1_jailbreak": f1_score(y_test_num, pred_sel, zero_division=0)},
    ])
    fs_metrics.to_csv(os.path.join(DP, "feature_selection_comparison.csv"), index=False)
    row_norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    pd.DataFrame([{
        "norm": "L2", "mean_row_norm": row_norms.mean(), "median_row_norm": np.median(row_norms),
        "min_row_norm": row_norms.min(), "max_row_norm": row_norms.max(),
        "target_value_encoding": "benign=0, jailbreak=1 (analysis only)"
    }]).to_csv(os.path.join(DP, "normalization_encoding_audit.csv"), index=False)
    pd.DataFrame({"original_label": ["benign", "jailbreak"], "encoded_value": [0, 1]}).to_csv(os.path.join(DP, "label_encoding.csv"), index=False)
    pd.DataFrame({"feature": selected_names, "chi2_score": selector.scores_[selector.get_support()]}) \
      .sort_values("chi2_score", ascending=False).to_csv(os.path.join(DP, "selected_features_chi2.csv"), index=False)
    pd.DataFrame([{
        "tfidf_features_before_selection": X.shape[1],
        "features_after_chi2_selection": X_selected.shape[1],
        "tfidf_norm": vec.norm,
        "stop_words": "english",
        "ngram_range": "(1,2)",
    }]).to_csv(os.path.join(DP, "feature_extraction_audit.csv"), index=False)

    # 9) Binning/binarization summaries for report/EDA.
    bin_summary = tr.groupby(["type", "length_bin"], observed=False).size().reset_index(name="count")
    bin_summary.to_csv(os.path.join(DP, "length_bins.csv"), index=False)
    kw_summary = tr.groupby("type")["any_keyword"].agg(["sum", "mean"]).reset_index()
    kw_summary.columns = ["type", "keyword_hit_count", "keyword_hit_rate"]
    kw_summary.to_csv(os.path.join(DP, "keyword_binarization.csv"), index=False)

    # 10) Applicability matrix for all lecture techniques.
    applicability = [
        ("Data integration", "Applied", "Train/test schemas audited; official split kept separate to prevent leakage."),
        ("Data aggregation", "Not applied", "There is no repeated transactional/time-series grain requiring grouped aggregation; prompts remain independent observations."),
        ("Sampling", "Not applied", "Dataset is small enough to process in memory; sampling would discard training examples."),
        ("Compression/indexing/chunking", "Not applied", "Not needed for a 1,306-row text dataset."),
        ("Missing-value imputation/interpolation", "Not applied", "No missing prompt/label values exist."),
        ("Duplicate removal", "Applied", "Exact duplicates removed before modeling; normalized duplicates also checked."),
        ("Data types/format standardization", "Applied", "CSV schema validated; prompts normalized to lowercase and collapsed whitespace."),
        ("Outlier detection", "Applied", "IQR and |Z|>3 analyses performed on word/character length; observations retained."),
        ("Data smoothing", "Not applied", "Moving averages are inappropriate for independent text prompts."),
        ("SMOTE", "Not applied", "Training classes are balanced (516 benign / 515 jailbreak)."),
        ("Feature scaling", "Applied implicitly", "TF-IDF provides a normalized sparse feature representation; conventional scalar scaling is not required for the primary text pipeline."),
        ("Normalization", "Applied", "TF-IDF uses L2 normalization; no Min-Max transform was forced onto sparse text features."),
        ("Standardization", "Not applied", "No dense continuous feature block is required by the primary TF-IDF pipeline."),
        ("Log transformation", "Evaluated", "log1p length transformation compared against original skewness/kurtosis; not inserted into the primary text model."),
        ("Categorical encoding", "Applied implicitly", "Binary target classes are handled by scikit-learn; explicit manual encoding is unnecessary."),
        ("Binning", "Applied for analysis", "Word count categorized as short/medium/long."),
        ("Binarization", "Applied for analysis", "Six jailbreak-keyword flags and any-keyword indicator created."),
        ("Feature creation", "Applied", "Length, punctuation, case, digit, newline, keyword and related text features created."),
        ("Feature selection", "Applied for analysis", "Chi-square SelectKBest reduced 5,000 TF-IDF features to 1,000 for a documented experiment."),
        ("Feature extraction", "Applied", "TF-IDF unigram+bigram representation."),
        ("PCA", "Not applied", "Sparse TF-IDF is better preserved by sparse-aware text methods; PCA is unnecessary for the primary classifier."),
        ("Tokenization", "Applied", "Explicit tokenization audit plus TF-IDF tokenization."),
        ("Stopword removal", "Applied", "English stopwords removed inside TF-IDF."),
        ("Lowercasing", "Applied", "All modeling text lowercased during cleaning."),
        ("Punctuation/special characters", "Preserved intentionally", "Not aggressively stripped because prompt structure can be discriminative."),
        ("Stemming", "Not applied", "Could collapse meaningful adversarial lexical forms; not required for the baseline."),
        ("Lemmatization", "Not applied", "Not required for the baseline; exact lexical forms can carry signal."),
        ("POS tagging", "Not applied", "Not required for the binary intent-classification objective."),
        ("Spell correction", "Not applied", "Could alter adversarial/obfuscated wording and is not necessary for the baseline."),
        ("Named entity recognition", "Not applied", "Entity identity is not the classification target."),
        ("Sentiment analysis", "Not applied", "Sentiment is not the target and is not necessary for jailbreak detection."),
    ]
    pd.DataFrame(applicability, columns=["technique", "status", "project_decision"]).to_csv(os.path.join(DP, "technique_applicability.csv"), index=False)

    # 11) Human-readable audit summary.
    lines = ["# Data Preparation Audit — Lecture 5", "", "All relevant techniques were evaluated; inappropriate transformations were documented rather than forced.", ""]
    lines += [f"- Raw train/test: {len(raw_train)}/{len(raw_test)} rows; cleaned: {len(train)}/{len(test)}.",
              f"- Missing values: train={int(raw_train.isna().sum().sum())}, test={int(raw_test.isna().sum().sum())}.",
              f"- Raw duplicate rows: train={int(raw_train.duplicated().sum())}, test={int(raw_test.duplicated().sum())}.",
              f"- TF-IDF matrix: {X.shape[0]} documents × {X.shape[1]} features; L2 normalization.",
              f"- Chi-square feature-selection audit: {X.shape[1]} → {X_selected.shape[1]} features."]
    lines += ["", "See CSV/PNG artifacts in `output/data_preparation/` for the complete calculations."]
    with open(os.path.join(DP, "audit_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Data-preparation audit complete.")
    print(f"Artifacts: {DP}")


if __name__ == "__main__":
    main()
