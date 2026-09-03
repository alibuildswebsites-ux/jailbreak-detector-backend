#!/usr/bin/env python3
"""Lecture 6 EDA and visualization agent.

Produces Matplotlib and Seaborn visualizations with titles, axes, legends
where appropriate, and interpretation/caption text. The dataset has no time
variable, so a line plot is documented as not applicable rather than creating
a misleading trend chart.
"""
from pathlib import Path
from collections import Counter
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "cleaned_train.csv"
OUT = BASE / "output" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

STOPWORDS = set("a about above after again against all am an and any are as at be because been before being below between both but by can cannot could did do does doing down during each few for from further had has have having he her here him his how i if in into is it its itself just me more most my no nor not of off on once only or other our out over own same she should so some such than that the their them then there these they this those through to too under until up very was we were what when where which while who why with would you your".split())
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
KEYWORDS = ["dan", "ignore", "previous instructions", "no restrictions", "roleplay", "unfiltered"]


def tokenize(text):
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS and len(t) > 1]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(TRAIN)
    df["word_count"] = df["prompt"].str.split().str.len()
    df["char_count"] = df["prompt"].str.len()
    df["length_bin"] = pd.cut(df.word_count, [-float("inf"), 50, 250, float("inf")], labels=["Short (≤50)", "Medium (51–250)", "Long (>250)"])

    # 1. Class balance — Matplotlib bar.
    counts = df["type"].value_counts().reindex(["benign", "jailbreak"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars, padding=3)
    ax.set(title="Training Class Balance", xlabel="Class", ylabel="Number of prompts")
    ax.text(0.5, -0.20, "Caption: The cleaned training split is essentially balanced (516 benign, 515 jailbreak).", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "class_balance.png")

    # 2. Matplotlib histogram — distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in ["benign", "jailbreak"]:
        x = df.loc[df.type == cls, "word_count"]
        ax.hist(x, bins=40, alpha=.55, label=cls)
    ax.set_xlim(0, df.word_count.quantile(.995) * 1.05)
    ax.set(title="Prompt-Length Distribution by Class", xlabel="Prompt length (words)", ylabel="Frequency")
    ax.legend(title="Class")
    ax.text(0.5, -0.22, "Caption: Jailbreak prompts have a much higher center and broader word-count distribution.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "length_dist.png")

    # 3. Matplotlib scatter — relationship.
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, marker in [("benign", "o"), ("jailbreak", "x")]:
        x = df.loc[df.type == cls]
        ax.scatter(x.word_count, x.char_count, alpha=.45, s=22, label=cls, marker=marker)
    ax.set(title="Word Count vs Character Count", xlabel="Word count", ylabel="Character count")
    ax.legend(title="Class")
    ax.set_xlim(0, df.word_count.quantile(.995) * 1.05)
    ax.set_ylim(0, df.char_count.quantile(.995) * 1.05)
    ax.text(0.5, -0.20, "Caption: Word and character counts are strongly related; class separation is also visible through prompt length.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "matplotlib_scatter_length.png")

    # 4. Top words.
    top_words = {}
    for cls in ["benign", "jailbreak"]:
        toks = [t for txt in df.loc[df.type == cls, "prompt"] for t in tokenize(txt)]
        top_words[cls] = Counter(toks).most_common(20)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, cls in zip(axes, ["benign", "jailbreak"]):
        words, freqs = zip(*top_words[cls][::-1])
        ax.barh(words, freqs)
        ax.set(title=f"{cls.title()}: Top 20 Words", xlabel="Frequency", ylabel="Token")
    fig.suptitle("Most Frequent Non-Stopword Tokens", fontweight="bold")
    save(fig, OUT / "top_words.png")

    # 5. Keyword rates.
    rows = []
    for kw in KEYWORDS:
        hit = df.prompt.str.contains(re.escape(kw), case=False, regex=True)
        for cls in ["benign", "jailbreak"]:
            mask = df.type.eq(cls)
            rows.append({"keyword": kw, "class": cls, "count": int((hit & mask).sum()), "rate": float((hit & mask).sum() / mask.sum())})
    kw_df = pd.DataFrame(rows)
    kw_df.to_csv(OUT / "keywords.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=kw_df, x="rate", y="keyword", hue="class", ax=ax)
    ax.set(title="Selected Jailbreak Keyword Hit Rates", xlabel="Hit rate", ylabel="Keyword")
    ax.legend(title="Class")
    ax.text(0.5, -0.20, "Caption: Most selected markers are more common in jailbreak prompts; roleplay is nearly non-discriminative.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "keyword_hit_rates.png")

    # 6. Top bigrams.
    top_bigrams = {}
    for cls in ["benign", "jailbreak"]:
        bg = Counter()
        for txt in df.loc[df.type == cls, "prompt"]:
            toks = tokenize(txt)
            bg.update(zip(toks, toks[1:]))
        top_bigrams[cls] = bg.most_common(15)
    bigram_rows = []
    for cls in ["benign", "jailbreak"]:
        for (a, b), count in top_bigrams[cls]:
            bigram_rows.append({"class": cls, "bigram": f"{a} {b}", "frequency": count})
    pd.DataFrame(bigram_rows).to_csv(OUT / "top_bigrams.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, cls in zip(axes, ["benign", "jailbreak"]):
        pairs, freqs = zip(*top_bigrams[cls][::-1])
        ax.barh([" ".join(p) for p in pairs], freqs)
        ax.set(title=f"{cls.title()}: Top 15 Bigrams", xlabel="Frequency", ylabel="Bigram")
    fig.suptitle("Top Bigrams by Class", fontweight="bold")
    save(fig, OUT / "top_bigrams.png")

    # 7. Seaborn scatterplot.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="word_count", y="char_count", hue="type", style="type", alpha=.55, ax=ax)
    ax.set(title="Seaborn Scatterplot: Prompt Length Relationship", xlabel="Word count", ylabel="Character count")
    ax.set_xlim(0, df.word_count.quantile(.995) * 1.05)
    ax.set_ylim(0, df.char_count.quantile(.995) * 1.05)
    ax.legend(title="Class")
    ax.text(0.5, -0.20, "Caption: Seaborn confirms the positive word/character-length relationship and class separation.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "seaborn_scatter_length.png")

    # 8. Seaborn KDE.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="word_count", hue="type", fill=False, common_norm=False, ax=ax)
    ax.set_xlim(0, df.word_count.quantile(.995) * 1.05)
    ax.set(title="Seaborn KDE: Prompt-Length Distribution", xlabel="Word count", ylabel="Density")
    ax.text(0.5, -0.20, "Caption: KDE smooths the distribution; neither class resembles a simple symmetric normal pattern.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "seaborn_kde_length.png")

    # 9. Seaborn categorical plot.
    cat = df.groupby(["length_bin", "type"], observed=False).size().reset_index(name="count")
    cat["proportion_within_class"] = cat.groupby("type")["count"].transform(lambda s: s / s.sum())
    cat.to_csv(OUT / "length_bin_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=cat, x="length_bin", y="proportion_within_class", hue="type", ax=ax)
    ax.set(title="Seaborn Categorical Plot: Prompt-Length Bins", xlabel="Length category", ylabel="Proportion within class")
    ax.legend(title="Class")
    ax.text(0.5, -0.20, "Caption: Jailbreak prompts have a much larger share of long prompts than benign prompts.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "seaborn_length_categories.png")

    # 10. Existing heatmap retained, now generated through Seaborn.
    numeric = df[["word_count", "char_count"]].copy()
    numeric["jailbreak"] = (df.type == "jailbreak").astype(int)
    corr = numeric.corr()
    corr.to_csv(OUT / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set(title="Correlation Heatmap", xlabel="Variable", ylabel="Variable")
    ax.text(0.5, -0.22, "Caption: Length variables are highly correlated with each other and positively associated with the jailbreak label.", transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, OUT / "correlation_heatmap.png")

    # 11. Applicability note for the lecture's line plot.
    (OUT / "plot_applicability.md").write_text(
        "# Lecture 6 Plot Applicability\n\n"
        "- **Matplotlib line plot:** Not applied. The dataset contains no meaningful time/order variable, while the lecture describes line plots as generally used for trends over time. Creating one from arbitrary row order would be misleading.\n"
        "- **Matplotlib scatter:** Applied (`matplotlib_scatter_length.png`).\n"
        "- **Matplotlib histogram:** Applied (`length_dist.png`).\n"
        "- **Seaborn scatter:** Applied (`seaborn_scatter_length.png`).\n"
        "- **Seaborn boxplot:** Applied in the data-preparation audit (`output/data_preparation/boxplot_*.png`).\n"
        "- **Seaborn KDE:** Applied (`seaborn_kde_length.png`).\n"
        "- **Seaborn heatmap:** Applied (`correlation_heatmap.png`).\n"
        "- **Seaborn categorical plot:** Applied (`seaborn_length_categories.png`).\n", encoding="utf-8")

    # 12. EDA insights.
    n_b, n_j = (df.type == "benign").sum(), (df.type == "jailbreak").sum()
    any_kw = df.prompt.str.contains("|".join(map(re.escape, KEYWORDS)), case=False, regex=True)
    overlap = set(w for w, _ in top_words["benign"]) & set(w for w, _ in top_words["jailbreak"])
    lines = [
        "# EDA Insights — Lecture 6", "",
        f"Source: `cleaned_train.csv` ({len(df):,} prompts).", "",
        f"1. **Balanced classes:** {n_b} benign and {n_j} jailbreak prompts.",
        f"2. **Length separates classes:** median word count is {df.loc[df.type=='benign','word_count'].median():.0f} for benign vs {df.loc[df.type=='jailbreak','word_count'].median():.0f} for jailbreak.",
        f"3. **Keyword coverage:** at least one selected keyword occurs in {any_kw[df.type.eq('benign')].mean():.1%} of benign and {any_kw[df.type.eq('jailbreak')].mean():.1%} of jailbreak prompts.",
        f"4. **Lexical overlap:** {len(overlap)} of the top 20 tokens are shared between classes, so many class-specific lexical signals remain.",
        "5. **Scatter relationship:** word count and character count rise together, so they should not be interpreted as independent evidence.",
        "6. **Distribution shape:** histograms and KDEs show strong right skew and long tails, supporting median/IQR alongside mean/SD.",
        "7. **Outliers:** boxplots from the data-preparation audit identify extreme prompt lengths; they are retained because unusually long prompts can be genuine examples.",
        "8. **Visualization decision:** no line plot is created because row order is not a time variable and an arbitrary line would imply a false trend.",
        "",
    ]
    (OUT / "insights.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"All EDA outputs written to {OUT}")


if __name__ == "__main__":
    main()
