# REPORT — Jailbreak Prompt Detection

Binary text classification of LLM prompts as **benign** or **jailbreak** using classic machine learning. The final baseline compares **TF-IDF + Logistic Regression, Naive Bayes, and Random Forest**. The selected model is **Logistic Regression**, with **97.71% accuracy, 97.81% jailbreak F1, and 96.40% jailbreak recall** on the official held-out test set.

---

## 1. Problem Statement

Jailbreak prompts are adversarially crafted inputs intended to bypass an LLM's safety alignment. Examples include instruction overrides, persona or mode framing, and attempts to remove restrictions. This project builds a lightweight first-line detector that classifies a prompt as `benign` or `jailbreak` before it reaches an LLM.

The task is a **binary text-classification** problem. Because a missed jailbreak is more safety-critical than rejecting a benign prompt, **jailbreak recall and F1** are emphasized when selecting the final model.

## 2. Dataset

Source files: `jailbreak_train.csv` and `jailbreak_test.csv`.

| Property | Train | Test |
|---|---:|---:|
| Raw rows | 1,044 | 262 |
| Cleaned rows | 1,031 | 262 |
| Rows dropped | 13 | 0 |
| Benign | 517 raw / 516 clean | 123 |
| Jailbreak | 527 raw / 515 clean | 139 |
| Columns | `prompt`, `type` | `prompt`, `type` |

The supplied train/test schemas were identical, so no merge was performed. The official split was deliberately preserved to prevent train/test leakage and to make the reported evaluation reproducible.

### Data-quality findings

- No missing values were found in either raw dataset.
- The raw training set contained **11 exact duplicate rows**.
- Cleaning removed those 11 duplicates plus **2 additional duplicates created after normalization**.
- No empty prompts remained after cleaning.
- No label conflicts were detected after cleaning.
- The classes are effectively balanced in training (**516 benign / 515 jailbreak**), so SMOTE or class reweighting was not required.

---

## 3. Data Preparation and Preprocessing

The preprocessing stage was expanded to cover the relevant techniques from the Data Preparation lecture. Techniques that do not fit independent unstructured text were evaluated and explicitly documented rather than applied mechanically.

### 3.1 Data integration and schema validation

The train and test files were checked for column consistency and data types. Both contain the same two fields: `prompt` (text) and `type` (categorical target). They remain separate throughout modeling because combining them before fitting would create leakage.

Artifact: `output/data_preparation/schema_audit.csv`.

### 3.2 Cleaning and standardization

Applied operations:

1. Lowercase text.
2. Strip leading/trailing whitespace.
3. Collapse repeated whitespace to a single space.
4. Remove exact duplicate rows.
5. Remove duplicates created after normalization.
6. Remove empty prompts.
7. Preserve punctuation and original phrasing rather than aggressively deleting special characters.

The last decision is intentional: adversarial prompts can contain meaningful formatting, quotations, role-play markers, and instruction structure.

### 3.3 Missing values and imputation

Missing-value analysis found **zero missing values** in `prompt` and `type` for both raw splits. Therefore mean/median/mode imputation, interpolation, forward fill, and backward fill were unnecessary and were not applied.

### 3.4 Statistical profiling

For engineered numeric features, the project now reports count, mean, median, standard deviation, minimum, maximum, skewness, and kurtosis by class.

Key training-set length statistics:

| Feature | Class | Mean | Median | Std | Skewness | Kurtosis | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| Word count | Benign | 85.55 | 36 | 145.09 | 5.82 | 56.63 | 1,939 |
| Word count | Jailbreak | 329.73 | 259 | 267.97 | 1.78 | 5.66 | 1,973 |
| Character count | Benign | 510.07 | 220 | 866.96 | 6.18 | 64.10 | 11,977 |
| Character count | Jailbreak | 1,945.94 | 1,553 | 1,542.05 | 1.76 | 6.24 | 11,869 |

The large positive skew and high kurtosis show that prompt length has a long right tail, especially for benign prompts. Jailbreak prompts are substantially longer on average and by median, creating a known length confound.

Artifact: `output/data_preparation/data_profile.csv` and `numeric_profile_by_class.csv`.

### 3.5 Correlation analysis

Numeric text-derived features were correlated with an analysis-only binary target (`benign=0`, `jailbreak=1`). Word count, character count, punctuation count, and the keyword indicator all show positive association with the jailbreak label. Word count and character count are also highly correlated with each other, so they should not be interpreted as independent evidence.

The strongest useful relationship is between **presence of at least one selected jailbreak keyword and the target** (Pearson correlation ≈ 0.493). This supports the EDA finding that lexical triggers are useful, while also showing why a classifier is preferable to a single keyword rule.

Artifact: `output/data_preparation/correlation_matrix.csv` and `correlation_heatmap.png`.

### 3.6 Outlier detection

Prompt length was evaluated with both lecture-style methods:

- **IQR rule:** observations below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.
- **Z-score:** observations with |Z| > 3.

| Feature | IQR outliers | IQR rate | Z-score outliers | Z-score rate |
|---|---:|---:|---:|---:|
| Word count | 54 | 5.17% | 10 | 0.96% |
| Character count | 44 | 4.21% | 11 | 1.05% |

These observations were **retained**. An unusually long prompt is not automatically a data-quality error in this problem; long prompts can be genuine jailbreak examples. Removing them could remove exactly the behavior the detector is meant to learn.

Artifacts: `output/data_preparation/outlier_analysis.csv`, `boxplot_word_count.png`, and `boxplot_char_count.png`.

### 3.7 Distribution and log transformation

`word_count` and `char_count` were both right-skewed. A `log1p` transformation was evaluated:

| Feature | Original skew | log1p skew | Original kurtosis | log1p kurtosis |
|---|---:|---:|---:|---:|
| Word count | 2.33 | -0.11 | 8.67 | -1.04 |
| Character count | 2.33 | -0.14 | 9.36 | -1.05 |

The transformation substantially reduces skewness and kurtosis. It is retained as an **analytical transformation**, but it is not inserted into the primary TF-IDF text model because the final classifier does not depend on these continuous length variables.

Artifacts: `output/data_preparation/log_transform_analysis.csv` and the two log-transform plots.

### 3.8 Class imbalance and SMOTE

The cleaned training classes are essentially perfectly balanced: **516 benign vs 515 jailbreak**. Therefore SMOTE, random oversampling, and class reweighting were not required.

### 3.9 Data aggregation, smoothing, and data reduction

No data aggregation or grouped aggregation is required because each prompt is an independent observation rather than a repeated transactional/time-series grain. Moving-average smoothing is likewise inappropriate because the observations are independent text prompts rather than a time series. Sampling, chunking, indexing, and compression were not needed for a 1,306-row dataset that fits comfortably in memory.

### 3.10 Encoding, normalization, and standardization

The binary target is represented analytically as `benign=0` and `jailbreak=1`; the saved classifiers can directly consume the original string labels, so manual label encoding is not required for model fitting.

The primary text representation is TF-IDF with `norm='l2'`. Therefore each non-empty TF-IDF document is normalized by its L2 norm, providing the required feature scaling/normalization for the sparse representation. Conventional Min-Max scaling or StandardScaler was not forced onto the sparse TF-IDF matrix because that would be unnecessary for this representation.

Artifacts: `label_encoding.csv` and `normalization_encoding_audit.csv`.

### 3.11 Binning and binarization

For exploratory analysis, word count was discretized into:

- **Short:** ≤50 words
- **Medium:** 51–250 words
- **Long:** >250 words

Six binary keyword indicators were also created: `dan`, `ignore`, `previous instructions`, `no restrictions`, `roleplay`, and `unfiltered`. An `any_keyword` indicator records whether at least one was present.

These are analytical/feature-engineering artifacts and are not substituted for the primary TF-IDF representation.

Artifacts: `length_bins.csv` and `keyword_binarization.csv`.

### 3.12 Text preprocessing

The project explicitly evaluates the lecture's NLP preprocessing techniques:

| Technique | Decision | Reason |
|---|---|---|
| Lowercasing | Applied | Reduces case variation for TF-IDF. |
| Whitespace normalization | Applied | Standardizes text formatting. |
| Tokenization | Applied | Required for token-based TF-IDF; independently audited. |
| Stopword removal | Applied | English stopwords removed by TF-IDF. |
| Punctuation removal | Not applied | Prompt punctuation/structure can carry adversarial signal. |
| Stemming | Not applied | May collapse useful adversarial lexical distinctions. |
| Lemmatization | Not applied | Not necessary for the baseline; exact wording can be informative. |
| POS tagging | Not applied | Not required for the binary classification objective. |
| Spell correction | Not applied | Could alter intentional obfuscation/adversarial wording. |
| Named entity recognition | Not applied | Entity identification is not the prediction target. |
| Sentiment analysis | Not applied | Sentiment is not the prediction target. |

The cleaned training set contains **211,971 explicit tokens**, with **12,569 unique tokens** under the audit tokenizer.

Artifact: `output/data_preparation/tokenization_summary.csv`.

---

## 4. Descriptive Statistical Analysis (Lecture 6)

The descriptive-statistics analysis is based on the **cleaned training split (N=1,031)**. The official test split is intentionally excluded from descriptive analysis used to understand the training data. This section implements the Lecture 6 topics: central tendency, dispersion, coefficient of variation, position, z-scores, frequency tables, and distribution shape. The lecture also demonstrates `pandas.describe()` for numerical summaries.

### 4.1 Central tendency

| Feature | Mean | Median | Mode | 10% Trimmed Mean | 20% Trimmed Mean |
|---|---:|---:|---:|---:|---:|
| Word count | 207.54 | 108.00 | 12 | 160.06 | 137.69 |
| Character count | 1,220.79 | 671.00 | 114 | 951.36 | 820.47 |

The mean is substantially above the median for both features, showing the effect of the long right tail. Trimmed means move closer to the median, illustrating the lecture's point that trimming reduces the influence of extreme observations.

### 4.2 Dispersion

| Feature | Range | Sample Variance | Sample SD | CV | IQR |
|---|---:|---:|---:|---:|---:|
| Word count | 1,969 | 61,671.71 | 248.34 | 119.66% | 274.00 |
| Character count | 11,902 | 2,068,331.44 | 1,438.17 | 117.81% | 1,642.50 |

The very large coefficients of variation indicate substantial relative variability. Range is especially sensitive to the extreme maximum values, so IQR and standard deviation are reported alongside it.

### 4.3 Measures of position

| Group | Feature | P10 | Q1/P25 | Q2/P50 | Q3/P75 | P90 | P95 | IQR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | Word count | 16.0 | 32.5 | 108.0 | 306.5 | 551.0 | 723.5 | 274.0 |
| Overall | Character count | 96.0 | 196.0 | 671.0 | 1,838.5 | 3,246.0 | 4,169.5 | 1,642.5 |
| Benign | Word count | 12.0 | 18.75 | 36.5 | 92.25 | 211.0 | 362.75 | 73.5 |
| Jailbreak | Word count | 61.4 | 136.0 | 259.0 | 448.0 | 694.8 | 815.6 | 312.0 |

The complete class-wise character-count table is stored in `output/descriptive_statistics/percentiles_quartiles_iqr.csv`.

### 4.4 Z-score analysis

The cleaned training data has 10 word-count observations and 11 character-count observations beyond |Z| > 3. The observations are retained because an unusually long prompt is not necessarily erroneous in this problem; it can be a genuine jailbreak example. The complete summary is stored in `output/descriptive_statistics/zscore_summary.csv`.

### 4.5 Shape and distribution

| Feature | Skewness | Kurtosis | Interpretation |
|---|---:|---:|---|
| Word count | 2.33 | 8.67 | Strong right skew and heavy tail |
| Character count | 2.33 | 9.36 | Strong right skew and heavy tail |

The distributions are therefore not simple symmetric normal distributions. Exploratory KDE peak estimates are provided in `output/descriptive_statistics/modality_kde_summary.csv`; these are treated as EDA evidence rather than a formal modality test.

### 4.6 Frequency distributions and `describe()`

Class-frequency and word-length-bin tables are saved under `output/descriptive_statistics/`. `pandas_describe.csv` records the standard count, mean, standard deviation, minimum, quartiles and maximum summary for the two engineered numeric features.

### 4.7 Statistical-analysis decisions

- Mean/variance/standard deviation use sample formulas (`ddof=1`).
- Median and IQR are emphasized because the length variables are strongly skewed.
- Z-score outliers are inspected but not automatically removed.
- A line plot is **not** created: the dataset has no meaningful temporal/order variable, and the lecture describes line plots as generally appropriate for trends over time.

**Artifacts:** `output/descriptive_statistics/`.

## 5. Exploratory Data Analysis and Visualization (Lecture 6)

The lecture describes visualization as a way to understand structure, detect outliers, identify relationships, assess distributions, and support decisions. The project now uses both Matplotlib and Seaborn with explicit titles, axis labels, legends where required, and captions.

### Visualizations implemented

- **Matplotlib bar:** `output/eda/class_balance.png` — class frequency.
- **Matplotlib histogram:** `output/eda/length_dist.png` — word-count distribution by class.
- **Matplotlib scatter:** `output/eda/matplotlib_scatter_length.png` — word count vs character count.
- **Seaborn scatter:** `output/eda/seaborn_scatter_length.png` — same relationship with statistical-EDA styling.
- **Seaborn KDE:** `output/eda/seaborn_kde_length.png` — smoothed word-count distributions.
- **Seaborn categorical plot:** `output/eda/seaborn_length_categories.png` — short/medium/long prompt proportions.
- **Seaborn heatmap:** `output/eda/correlation_heatmap.png` — correlation among engineered numeric variables and class.
- **Boxplots:** `output/data_preparation/boxplot_word_count.png` and `boxplot_char_count.png` — IQR/outlier inspection.
- **Top-word and bigram plots:** lexical EDA.

The line-plot requirement is explicitly documented as not applicable because arbitrary row order would create a misleading trend. The full decision is recorded in `output/eda/plot_applicability.md`.

### Matplotlib vs Seaborn

Matplotlib is used where direct plot construction and annotation are useful. Seaborn is used for statistical EDA such as scatterplots, KDE, categorical comparisons and heatmaps. This follows the lecture's distinction between Matplotlib's fine-grained control and Seaborn's high-level statistical plotting.

## 6. Feature Engineering and Extraction

### 6.1 Primary feature extraction: TF-IDF

The main representation is:

```text
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    norm="l2"
)
```

The vectorizer is fitted **only on the cleaned training set** and then used to transform the test set. This prevents test-set vocabulary leakage.

Unigrams capture individual lexical indicators while bigrams capture phrases such as `ignore previous` and `developer mode`.

### 6.2 Engineered features

The data-preparation audit creates additional features for analysis:

- word count
- character count
- punctuation count
- uppercase count and ratio
- digit count
- newline count
- exclamation/question counts
- six keyword flags
- any-keyword indicator
- length bins.

These demonstrate feature creation from the lecture without replacing the proven TF-IDF baseline.

### 6.3 Feature selection

Chi-square `SelectKBest` was evaluated as a formal feature-selection method. It reduced the 5,000-feature TF-IDF representation to the top 1,000 features.

| Representation | Features | Accuracy | Jailbreak Precision | Jailbreak Recall | Jailbreak F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 5,000 | **97.71%** | 99.26% | **96.40%** | **97.81%** |
| TF-IDF + Chi-square | 1,000 | 96.95% | 99.25% | 94.96% | 97.06% |

Because feature selection reduced performance, the **5,000-feature representation remains the final model input**.

Artifact: `output/data_preparation/feature_selection_comparison.csv` and `selected_features_chi2.csv`.

### 6.4 PCA / dimensionality reduction

PCA was considered but not used in the primary pipeline. The main representation is a high-dimensional sparse TF-IDF matrix, and PCA is not necessary for the selected linear classifier. The project therefore uses feature selection rather than forcing PCA into the final pipeline.

---

## 7. Model Training and Results

Three classic classifiers were trained on the official cleaned training split and evaluated on the official cleaned test split (`N=262`).

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Jailbreak Precision | Jailbreak Recall | Jailbreak F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **97.71%** | 97.66% | 97.79% | 97.71% | 99.26% | **96.40%** | **97.81%** |
| Random Forest | **97.71%** | 97.67% | 97.84% | **97.71%** | 100.00% | 95.68% | 97.79% |
| Naive Bayes | 96.18% | 96.14% | 96.22% | 96.17% | 97.08% | 95.68% | 96.38% |

### Confusion matrices

**Logistic Regression**

| Actual \ Predicted | Benign | Jailbreak |
|---|---:|---:|
| Benign | 122 | 1 |
| Jailbreak | 5 | 134 |

**Naive Bayes**

| Actual \ Predicted | Benign | Jailbreak |
|---|---:|---:|
| Benign | 119 | 4 |
| Jailbreak | 6 | 133 |

**Random Forest**

| Actual \ Predicted | Benign | Jailbreak |
|---|---:|---:|
| Benign | 123 | 0 |
| Jailbreak | 6 | 133 |

### Model selection

**Logistic Regression is selected as the final model.** It matches Random Forest's accuracy and macro F1 while achieving the best jailbreak recall and jailbreak F1. For this safety-oriented problem, reducing false negatives is more important than achieving perfect precision.

The Logistic Regression model has **5 false negatives** and **1 false positive**.

---

## 8. Error Analysis and Interpretability

### 8.1 Important Logistic Regression signals

The strongest positive coefficients include `chatgpt`, `prompt`, `user`, `character`, `ai`, `dan`, `response`, `respond`, `rules`, `mode`, `content`, `openai`, `illegal`, `code`, and `unethical`.

These indicate that jailbreak prompts frequently contain **meta-prompting language**: they discuss the AI, its rules, modes, responses, users, and restrictions rather than simply asking about a subject.

Negative/benign-side signals are documented in `output/results/logistic_feature_coefficients.csv`; representative terms include `history`, `options`, `review`, `article`, `following question`, `explain`, `sentence`, and `roleplay`. Importantly, `roleplay` alone is not discriminative according to EDA.

### 8.2 Failure patterns

The best model's six errors are saved in `output/results/misclassified_test_prompts.csv` and fall into recurring categories:

1. **Obfuscated or indirect jailbreaks** with few surface trigger words.
2. **Code/CLI simulation jailbreaks** where harmful intent is hidden inside a simulated function or environment.
3. **Short instruction overrides** with too little lexical context.
4. **Benign semantic/definition questions** that contain wording resembling model-instruction language.

The key limitation is that TF-IDF captures surface vocabulary rather than deep semantic intent.

---

## 9. Inferential Statistical Analysis (Lecture 7)

The inferential layer uses the **cleaned training split only (N=1,031)** as the sample. The official held-out test split is deliberately excluded from estimation and hypothesis testing so that the test set remains reserved for model evaluation. The implementation follows the lecture's emphasis on confidence intervals, p-values, hypothesis testing, correlation, and regression. The lecture presents bootstrapping as repeated sampling with replacement and percentile-based confidence intervals, and it lists independent t-test/Mann–Whitney, ANOVA/Kruskal–Wallis, Pearson/Spearman, and chi-square as the principal inferential families. fileciteturn9file0L10-L23 fileciteturn9file6L135-L147

### 9.1 Point and interval estimation

Sample means are treated as point estimates of the corresponding population means. Because the population standard deviation is unknown, the **t-distribution is the primary confidence-interval method**. The lecture's z-based known-σ formula is also implemented as a separate demonstration using the sample SD as a plug-in; it is explicitly not presented as a true known-population-σ interval because σ is not supplied by this dataset. The lecture specifies the known-σ conditions and distinguishes this case from the t-distribution case. fileciteturn5file8L180-L189

| Group | Feature | 90% CI | 95% CI | 99% CI |
|---|---|---|---|---|
| Benign | Word count | 75.17–96.24 | 73.15–98.26 | 69.18–102.23 |
| Jailbreak | Word count | 310.04–349.18 | 306.28–352.94 | 298.91–360.32 |

The complete intervals for word count and character count are stored in `output/inferential_statistics/confidence_intervals.csv`. As expected from the lecture, the 95% interval is wider than the 90% interval, and the 99% interval is wider still: greater confidence requires a larger margin of error. fileciteturn9file4L91-L102

### 9.2 Confidence interval for the jailbreak proportion

The cleaned training sample contains 515 jailbreak prompts out of 1,031 observations, giving a sample proportion of **0.4995 (49.95%)**. The lecture uses the normal approximation to the binomial distribution for a proportion CI when both `n p̂` and `n(1−p̂)` are at least 5. fileciteturn9file1L34-L38

| Confidence | Proportion CI |
|---:|---|
| 90% | 47.39%–52.51% |
| 95% | 46.90%–53.00% |
| 99% | 45.94%–53.96% |

### 9.3 Sample-size determination

Using the lecture's margin-of-error formulas, planning estimates are also provided. With the observed training SD of 248.34 words and a desired ±10-word margin of error, the required sample size is **1,669 at 90% confidence** and **2,370 at 95% confidence**. For estimating a proportion within ±5 percentage points, the required sample is **271 at 90%** and **385 at 95%**; the conservative `p=0.50` calculation gives the same rounded values. The lecture derives minimum sample size by solving the margin-of-error formula for n. fileciteturn5file0L10-L18

Artifact: `output/inferential_statistics/sample_size_determination.csv`.

### 9.4 Bootstrap inference

Because prompt lengths are strongly right-skewed and heavy-tailed, a non-parametric percentile bootstrap was implemented. Each group is resampled **5,000 times with replacement**, using the original group sample size, and the mean is calculated for every resample. The lecture describes exactly this repeated-resampling procedure and uses percentile cutoffs for 90% and 95% intervals. fileciteturn9file0L10-L23

For jailbreak word count, the bootstrap 95% CI is **306.78–354.00 words**, compared with **306.28–352.94** from the t interval. For benign word count, the bootstrap 95% CI is **74.18–99.14**, compared with **73.15–98.26** from the t interval. These close but not identical intervals provide a useful robustness comparison under the observed non-normal distribution.

Artifacts: `bootstrap_confidence_intervals.csv`, `bootstrap_mean_distribution.csv`, and `bootstrap_distribution_word_count.png`.

### 9.5 Hypothesis testing framework

Each formal test records the null hypothesis (H₀), alternative hypothesis (H₁), significance level α=0.05, test statistic, p-value, decision, and plain-language conclusion. The project distinguishes statistical Type I/Type II errors from classifier false positives/false negatives: a Type I error is rejecting a true H₀, while a Type II error is failing to reject a false H₀. Shapiro–Wilk diagnostics reject normality for the tested prompt-length variables, and Levene's test rejects equal variance for punctuation ratio across the three length groups (p=0.0222). These diagnostics support the use of Welch and non-parametric methods rather than relying only on equal-variance/normality assumptions.

### 9.6 Independent two-group comparison

The central question is whether jailbreak prompts are longer than benign prompts.

- **Two-tailed H₀:** μ_jailbreak = μ_benign; H₁: μ_jailbreak ≠ μ_benign.
- **Directional H₀:** μ_jailbreak ≤ μ_benign; H₁: μ_jailbreak > μ_benign.
- **Welch independent t-test:** two-tailed `t=18.0857`, `p=1.99×10⁻⁶¹`; reject H₀. The one-tailed test gives `p=9.96×10⁻⁶²` and also rejects H₀.
- **Mann–Whitney U:** two-tailed `U=230451.5`, `p=1.29×10⁻⁹²`; reject H₀. The one-tailed version gives `p=6.47×10⁻⁹³`.

The agreement between the parametric and non-parametric tests strengthens the conclusion that the two independent prompt-length distributions differ substantially. The lecture specifically pairs independent t-tests with Mann–Whitney U for two independent groups. fileciteturn9file6L135-L144

Paired t-test and Wilcoxon signed-rank were assessed but marked **not applicable** because the dataset contains independent prompts rather than matched pairs.

### 9.7 Three-group comparison

To avoid circularly testing word count against bins created from word count, the three lecture-style groups—Short (≤50), Medium (51–250), and Long (>250)—are used to compare **punctuation ratio** instead.

- **One-way ANOVA:** `F=2.2636`, `p=0.1045`; fail to reject H₀ at α=0.05.
- **Kruskal–Wallis:** `H=65.7856`, `p=5.19×10⁻¹⁵`; reject H₀.

The difference between these conclusions is itself informative: the non-parametric test detects a distributional difference that the mean-based ANOVA does not. Because the underlying prompt-derived variables are non-normal, the Kruskal–Wallis result is treated as the more robust evidence for a group difference. The lecture presents ANOVA and Kruskal–Wallis as the parametric/non-parametric pair for three or more groups. fileciteturn9file6L142-L147

### 9.8 Pearson and Spearman correlation

The existing descriptive correlation matrix has been extended to formal significance testing.

| Method | Coefficient | p-value | Decision |
|---|---:|---:|---|
| Pearson | r=0.992403 | <1×10⁻³⁰⁰* | Reject H₀ |
| Spearman | ρ=0.996480 | <1×10⁻³⁰⁰* | Reject H₀ |

`*` SciPy reports the extremely small p-values as numerical zero because they underflow floating-point representation. They are therefore reported here as `<1×10⁻³⁰⁰`, not as literal mathematical zero.

For Pearson correlation, H₀ is ρ=0 and H₁ is ρ≠0, with `df=n−2=1029`. The lecture specifies t-based significance testing for numeric correlation and identifies Spearman as appropriate when Pearson assumptions are not met, when the relationship is monotonic rather than necessarily linear, or when ranked data are used. fileciteturn10file5L76-L82 fileciteturn5file3L76-L89

Artifacts: `correlation_significance_tests.csv` and the existing scatter/EDA plots.

### 9.9 Chi-square test of independence

The earlier project used `SelectKBest(chi2)` for feature selection. That is **not** the inferential χ² independence test taught in Lecture 7. A separate contingency-table analysis now tests **selected-keyword presence × prompt class**.

- **H₀:** keyword presence and prompt class are independent.
- **H₁:** keyword presence and prompt class are associated.
- χ² = **250.2409**, df=**1**, p=**2.30×10⁻⁵⁶**.
- Decision: **Reject H₀**.
- Minimum expected frequency = **177.33**, satisfying the expected-frequency condition.

The observed and expected frequencies are saved separately. This follows the lecture's sequence of contingency table → expected frequencies → χ² statistic → degrees of freedom → independence conclusion. fileciteturn5file4L99-L104

Artifacts: `chi_square_contingency_table.csv`, `chi_square_expected_frequencies.csv`, and `chi_square_test.csv`.

### 9.10 Simple linear regression

Following the lecture's workflow of assessing correlation before regression, a least-squares regression predicts character count from word count. fileciteturn5file9L200-L212

The fitted equation is:

```text
char_count = 28.0207 + 5.7472 × word_count
```

The model has **R²=0.984863**. The word-count slope is highly significant (`t=258.749`, `p<1×10⁻³⁰⁰`). Interpreting the coefficients, each additional word is associated with approximately **5.75 additional characters on average** in this fitted linear relationship, while the intercept represents the model's predicted character count at zero words. At the sample median of 108 words, the model predicts approximately **648.72 characters**.

Artifacts: `simple_regression_summary.csv`, `simple_regression_prediction.csv`, `simple_regression_best_fit.png`, and `simple_regression_residuals.png`.

### 9.11 Multiple correlation and multiple linear regression

A multiple OLS model predicts character count from:

- word count
- punctuation count
- digit count
- exclamation count
- question count

The model achieves **R²=0.991291**, adjusted R²=0.991248, and multiple correlation **R=0.995636**. The coefficient table contains standard errors, t statistics, p-values and 95% confidence intervals for each predictor. The lecture describes multiple linear regression as a fundamental statistical/ML model with multiple predictors. fileciteturn5file6L138-L147

Artifacts: `multiple_regression_summary.csv`, `multiple_correlation.csv`, and `multiple_regression_coefficients.png`.

### 9.12 Lecture 7 applicability matrix

`output/inferential_statistics/lecture7_applicability_matrix.csv` records the implementation status of each lecture topic. In particular, paired tests are explicitly marked not applicable rather than omitted, while the known-σ z interval is marked as a limited demonstration because the supplied data do not contain a known population σ.

## 10. Limitations

- The dataset contains only **1,306 prompts**, so generalization to modern unseen attacks is uncertain.
- The supplied dataset is artificially balanced; real-world traffic would likely contain far more benign prompts.
- The dataset represents a particular collection period and jailbreak style; detector performance can decay under distribution shift.
- TF-IDF cannot reliably understand semantic obfuscation.
- The evaluation uses one official held-out split rather than temporal/rolling validation.
- Threshold calibration and precision-recall trade-offs were not optimized.
- No transformer/embedding baseline was used in the final comparison.
- Inferential conclusions are based on the supplied training sample and should not be interpreted as population truths beyond the sampling assumptions; the dataset is not a documented random sample of all real-world prompts.

## 11. Future Work

1. Add sentence-embedding features such as SBERT and compare them with TF-IDF.
2. Add temporal/era-based evaluation to measure jailbreak distribution drift.
3. Fine-tune a compact transformer such as DistilBERT as a neural comparison.
4. Tune the classification threshold for an explicit false-negative budget.
5. Combine an interpretable keyword/rule layer with the ML classifier.
6. Expand the dataset with newer, obfuscated, multilingual, and adversarially generated jailbreaks.
7. Extend inferential analysis to a genuinely sampled external population or temporal validation set so confidence intervals and hypothesis tests can support broader generalization claims.

## 12. Reproducibility and Artifacts

### Source scripts

- `prepare_data.py` — cleaning and dataset preparation
- `eda_agent.py` — exploratory analysis
- `data_preparation_audit.py` — lecture-aligned profiling, outliers, transformations, feature engineering/selection audit
- `descriptive_statistics.py` — Lecture 6 descriptive statistics
- `inferential_statistics.py` — Lecture 7 confidence intervals, bootstrap, hypothesis tests, correlations, χ² and regression
- `run_modeling.py` — TF-IDF and model training
- `test_prompt.py` — local inference utility

### Generated artifacts

- `output/data_preparation/` — Lecture 5 audit calculations and plots
- `output/descriptive_statistics/` — Lecture 6 descriptive statistics
- `output/eda/` — EDA plots and keyword analysis
- `output/inferential_statistics/` — Lecture 7 inferential calculations and plots
- `output/models/` — trained models and vectorizer
- `output/results/` — metrics, confusion matrices, feature importance and errors

The authoritative model results are the values in `output/results/metrics.csv`; inferential results are generated reproducibly by `inferential_statistics.py` from `cleaned_train.csv`.


## 13. Machine Learning Analysis (Lecture 8)

This section documents the machine-learning concepts and workflow required by the Machine Learning lecture. The project is a **supervised binary classification** task because each prompt has a known `benign` or `jailbreak` label. Unsupervised learning, regression metrics, and reinforcement learning are not added because they do not answer the project's classification objective.

### 13.1 Machine-learning lifecycle

The implemented workflow follows the lecture lifecycle: problem identification → data preparation → feature extraction/selection → algorithm selection → model training → validation → final test evaluation → error analysis and refinement. The official test set remains held out for final evaluation.

The ML system consists of historical prompt data, TF-IDF features, a learned classifier, a learning algorithm that estimates model parameters, and evaluation metrics. The vectorizer is fitted only on training data in the primary pipeline.

### 13.2 Dataset split and validation

The supplied official split is retained: **1,031 cleaned training prompts and 262 test prompts**. The test set is not used for model selection or cross-validation.

Because the current baseline has no hyperparameter tuning, a separate permanent validation set is not necessary. To provide the lecture's validation evidence without sacrificing the official test set, **Stratified 5-Fold Cross-Validation** was added on the training set only. TF-IDF is fitted independently inside each fold, preventing vocabulary leakage.

| Model | 5-fold mean accuracy | 5-fold mean macro F1 | F1 SD |
|---|---:|---:|---:|
| Logistic Regression | 95.54% | 95.53% | 1.99 percentage points |
| Random Forest | 95.34% | 95.33% | 1.97 percentage points |
| Naive Bayes | 94.47% | 94.47% | 1.26 percentage points |

Full results are in `output/ml_audit/stratified_5fold_cv.csv`.

### 13.3 Generalization, overfitting and underfitting

Train-vs-test diagnostics were added for all three models. The observed jailbreak-F1 gaps are small:

| Model | Train jailbreak F1 | Test jailbreak F1 | Gap |
|---|---:|---:|---:|
| Logistic Regression | 98.52% | 97.81% | 0.71 pp |
| Naive Bayes | 97.84% | 96.38% | 1.46 pp |
| Random Forest | 99.90% | 97.79% | 2.11 pp |

These gaps do **not** show a strong overfitting signal under the project's diagnostic threshold. Random Forest has the largest gap, so it is the model most worth watching for variance, but its held-out performance remains strong. None of the models shows a strong underfitting signal: training and held-out performance are both high. These diagnostics are evidence rather than proof that future unseen distributions will behave similarly.

Artifact: `output/ml_audit/train_test_generalization.csv` and `train_vs_test_f1.png`.

### 13.4 Model comparison and metric choice

All three models are evaluated with accuracy, macro precision, macro recall, macro F1, and class-specific jailbreak precision/recall/F1. The dataset is approximately balanced, so accuracy is informative, but **jailbreak recall and F1 remain the principal safety-oriented metrics** because a false negative means a jailbreak was missed. The final Logistic Regression choice remains justified by its highest jailbreak F1 and recall while matching Random Forest's accuracy.

### 13.5 Bias and data-compliance assessment

The lecture requires attention to fairness, privacy, transparency and bias. A dataset-specific bias audit was therefore added rather than making generic claims. The audit covers sampling, selection, measurement, labeling, historical, exclusion, evaluation, algorithmic/model, confirmation and observer bias.

Important findings are:

- The benchmark is curated and artificially close to 50/50 class balance, so **sampling/selection bias and deployment-prevalence shift are real limitations**.
- The available schema contains only prompt text and the binary label. There are **no demographic attributes**, so demographic subgroup fairness metrics cannot be honestly computed.
- Annotation metadata such as annotator agreement are not supplied; therefore labeling bias cannot be quantified from the project files.
- TF-IDF is inherently surface-lexical and can be biased toward vocabulary that happens to characterize the benchmark, which is supported by the project's error analysis and lexical EDA.
- Evaluation bias is reduced by keeping the official test set untouched during training/CV, but external and temporal representativeness remains untested.
- The report does not claim that the dataset contains no personal information; prompt text itself can potentially contain such information. The project should therefore be handled as data requiring appropriate privacy-aware use.
- Dataset provenance/license information already documented in the project is retained; this is documentation of provenance, **not a legal certification**.

Artifacts: `output/ml_audit/bias_and_compliance_audit.csv` and `data_compliance_ethics_audit.csv`.

### 13.6 Bias–variance trade-off

The lecture describes the bias–variance trade-off as balancing overly simple models (high bias) against overly complex models (high variance). In this project, Logistic Regression provides a strong, interpretable baseline with a small train/test gap; Random Forest reaches almost perfect training F1 but has a somewhat larger generalization gap. The 5-fold CV results and train/test comparison therefore provide concrete evidence for preferring the simpler Logistic Regression model for the final baseline rather than choosing a model solely because its training score is highest.

### 13.7 ML applicability decisions

`output/ml_audit/ml_lecture_applicability_matrix.csv` records each lecture requirement and its project-specific status. Unsupervised learning and regression metrics are marked **not applicable** because this dataset and research question are binary text classification. No unrelated algorithm has been added merely to satisfy a lecture keyword.

## 14. Machine Learning Audit Artifacts

The following new artifacts were generated specifically for the Machine Learning lecture:

- `output/ml_audit/stratified_5fold_cv.csv` — training-only Stratified 5-Fold validation.
- `output/ml_audit/train_test_generalization.csv` — train/test accuracy and jailbreak-F1 gaps.
- `output/ml_audit/train_vs_test_f1.png` — generalization diagnostic chart.
- `output/ml_audit/bias_and_compliance_audit.csv` — dataset/model bias assessment.
- `output/ml_audit/data_compliance_ethics_audit.csv` — privacy, provenance, fairness, transparency and safe-use assessment.
- `output/ml_audit/ml_lecture_applicability_matrix.csv` — requirement-by-requirement implementation matrix.
- `output/ml_audit/ml_audit_summary.json` — reproducibility metadata and validation safeguards.

