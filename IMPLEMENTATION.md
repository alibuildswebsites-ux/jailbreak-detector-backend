# Jailbreak Prompt Detection — Research-Based Implementation Plan

**Project:** Classifying LLM Prompts as Jailbreak vs. Benign
**Course:** Data Science — Iqra University, Karachi
**Instructor:** Dr. Mehak Usmani
**Team:** Raza Ali (IU04-0324-0201) · Nehad Shakoor (IU04-0123-0692) · Humaiz Ahmed (IU04-0122-0741) · Nabiha (IU04-0223-0068)
**Approach chosen:** Classic Machine Learning (TF-IDF + Logistic Regression / Random Forest / Naive Bayes)
**Created:** 2026-08-11

---

## 📁 Project Structure

```
ds1/
├── jailbreak_train.csv        # 1,044 training prompts
├── jailbreak_test.csv         # 262 test prompts
├── cleaned_train.csv          # cleaned training data
├── cleaned_test.csv           # cleaned test data
├── prepare_data.py            # cleaning
├── eda_agent.py               # EDA
├── data_preparation_audit.py  # Lecture 5 audit/preprocessing analysis
├── run_modeling.py            # TF-IDF + classifiers
├── test_prompt.py             # local inference utility
├── requirements.txt            # Python dependencies
├── IMPLEMENTATION.md           # implementation plan/checklist
├── REPORT.md                   # final project report
└── output/                    # models, plots, results, data-preparation audit
```

---

## 🔬 Research Summary (what the literature says)

Before implementation, we researched current jailbreak-detection literature (Aug 2026):

| Paper / Model | Approach | Key Result |
|---|---|---|
| Galinkin & Sablotny 2024 (arXiv 2412.01547) | Embeddings + Random Forest / XGBoost | Outperformed all open-source detectors (~3× F1 vs. second-best on JailbreakHub) |
| GradSafe (ACL 2024) | Gradient analysis of safety-critical params | Beats Llama Guard without training |
| FJD (EMNLP 2025) | First-token confidence, logit scaling | Near-zero-cost detection |
| Hawkins et al. 2025 (arXiv 2510.01644) | BERT fine-tuning + keyword analysis | Best accuracy on current datasets; explicit reflexivity = jailbreak signal |
| JailbreaksOverTime (arXiv 2504.19440) | Continuous learning / drift detection | Detectors decay within months as new jailbreaks appear |
| `AIccel_Jailbreak` (HF) | DeBERTa-v3-small fine-tuned on **our exact dataset** | 99.62% accuracy, 99.64% F1 |
| `Jailbreak-Detector-2-XL` (HF) | Qwen2.5-0.5B + LoRA (1.8M samples) | 99.48% accuracy |

**Decision:** Classic ML chosen as main approach — measurable baseline performance ≈ 96–98%, runs in seconds, no GPU needed, fully explainable (suits a course project). Embedding-based features (per Galinkin & Sablotny) kept as a stretch goal.

---

## 📊 Dataset Overview (verified 2026-08-11)

| Property | Value |
|---|---|
| Name | `jackhhao/jailbreak-classification` (Hugging Face) |
| License | Apache 2.0 — free for academic use |
| Total | 1,306 prompts |
| Train / Test | 1,044 / 262 (balanced split provided) |
| Class balance | Train: 517 benign / 527 jailbreak · Test: 123 benign / 139 jailbreak |
| Columns | `prompt` (text), `type` (label: `jailbreak` \| `benign`) |
| Prompt length | 33 → 10,412 chars (avg ≈ 1,000–1,400) |
| Jailbreak source | `verazuo/jailbreak_llms` (real in-the-wild attacks) |
| Benign source | OpenOrca + GPTeacher (normal instruction requests) |

**Files downloaded locally:** `jailbreak_train.csv`, `jailbreak_test.csv` (verified row counts above).

---

## 🗺️ Implementation Phases

### Phase 0 — Environment Setup

**Goal:** reproducible Python environment with required libraries.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn
```

| Dependency | Purpose |
|---|---|
| `pandas`, `numpy` | data loading & manipulation |
| `scikit-learn` | TF-IDF, models, metrics, train/test split |
| `matplotlib`, `seaborn` | EDA visualizations |

**Acceptance:** `import pandas, sklearn, matplotlib` succeeds.

---

### Phase 1 — Data Loading & Inspection

**Goal:** load CSVs, verify structure, confirm class balance, identify data-quality issues.

**Steps:**
1. `pd.read_csv("jailbreak_train.csv")` — same for test.
2. Print `df.info()`, `df.head()`, `df["type"].value_counts()`.
3. Check for nulls, duplicates, empty prompts.
4. Inspect prompt-length distribution (chars per prompt) per class.

**Expected observations:**
- 2 columns, 1,044 rows (train), 262 rows (test).
- ~50/50 class split (slightly more `jailbreak`).
- Some prompts contain newlines / quotes — CSV quoting already handles this, but keep `quoting=csv.QUOTE_ALL` in mind when re-saving.

**Research note:** the near-perfect 50/50 balance is *artificial* (real-world traffic is mostly benign). We report this as a known limitation.

---

### Phase 2 — Data Cleaning & Preprocessing

**Goal:** normalize text for consistent feature extraction.

| Step | Operation | Why |
|---|---|---|
| Lowercasing | `str.lower()` | TF-IDF treats `DAN` and `dan` the same |
| Whitespace | collapse `\s+` → single space | avoid counting repeated spaces |
| Special chars | optional: keep punctuation (jailbreaks often use `!`, quotes, role-play markers) | study shows signal may live in phrasing |
| Duplicates | drop exact duplicate prompts (if any) | prevent train/test leakage |
| Empty rows | drop if `prompt` empty | safe data hygiene |

**⚠️ Design decision:** do **NOT** strip punctuation aggressively — keyword indicators like `"DAN"`, `"ignore all previous instructions"`, `"no restrictions"` and role-play framing are the strongest signals per Hawkins et al. 2025.

**Stretch (research-backed):** keep a raw-text copy — Galinkin & Sablotny found embeddings capture jailbreak intent better than token statistics, so preserving original phrasing enables the Phase 5 stretch goal.

---

### Phase 3 — Exploratory Data Analysis (EDA)

**Goal:** understand what separates jailbreak from benign prompts.

**Analysis:**
1. **Class balance bar chart** — confirm split.
2. **Prompt-length histograms** per class (jailbreak vs benign) — is length predictive?
3. **Most frequent words** per class (before and after removing stopwords) — word clouds / bar charts.
4. **Keyword hit rates:** `"DAN"`, `"ignore"`, `"previous instructions"`, `"no restrictions"`, `"roleplay"`, `"unfiltered"` — what % of each class contains them?
5. **N-gram analysis:** top bigrams per class (e.g., `"ignore previous"`, `"do anything"`).

**Expected insights (from our research + live sample):**
- Jailbreak prompts skew longer, more instruction-like, use imperative phrasing ("Act as...", "Pretend...", "Ignore...").
- Benign prompts are ordinary Q&A / role-play requests.
- Keywords are strong but not sufficient → justifies ML over rule-based detection.

**Deliverable:** `output/eda/` plots + a short "insights" markdown.

---


### Phase 3.5 — Lecture 6 Descriptive Statistics & Visualization (implemented)

The project now explicitly implements the descriptive-statistics and EDA requirements from Lecture 6.

- `descriptive_statistics.py` produces mean, median, mode, 10%/20% trimmed means, range, sample variance, sample SD, coefficient of variation, percentiles, quartiles, IQR, z-score summary, frequency tables, `pandas.describe()` output, skewness, kurtosis, and exploratory KDE modality estimates.
- `eda_agent.py` produces Matplotlib bar/histogram/scatter plots plus Seaborn scatter/KDE/categorical/heatmap plots, with titles, axes, legends and captions.
- A line plot is documented as not applicable because there is no temporal/order variable.
- All descriptive statistics use `cleaned_train.csv`; the official test split is not used for this analysis.
- `seaborn>=0.13` is now included in `requirements.txt`.
- The main scripts use paths relative to the project directory for reproducibility.

Artifacts: `output/descriptive_statistics/` and expanded `output/eda/`.

### Phase 4 — Feature Engineering

**Goal:** convert text → numeric features for classifiers.

**Primary: TF-IDF (term frequency–inverse document frequency)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_test  = vectorizer.transform(test_texts)
```

| Parameter | Choice | Rationale |
|---|---|---|
| `max_features` | 5,000 | dataset is small; caps dimensionality |
| `ngram_range` | (1,2) | captures phrases like "ignore previous" |
| `stop_words` | english | removes noise words |
| `min_df` | 2 (optional) | drops ultra-rare tokens |

**Auxiliary features (research-backed, cheap):**
- `prompt_length` (chars) — add as an extra column.
- `keyword_flags` — binary columns for known jailbreak markers (`dan`, `ignore`, `no restrictions`, `jailbreak`, `roleplay`).
- Combine with TF-IDF via `scipy.sparse.hstack` + `ColumnTransformer`.

**Stretch goal (Galinkin & Sablotny 2024):** sentence-embedding features (e.g., `sentence-transformers/all-MiniLM-L6-v2`) instead of / alongside TF-IDF. Research shows embeddings + RF beats pure TF-IDF.

---

### Phase 5 — Model Training & Comparison

**Goal:** train 3 classic classifiers, compare honestly.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         MultinomialNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=200, n_jobs=-1),
}
```

**Evaluation protocol:**
- Train on the provided 1,044 split; evaluate on the held-out 262 test rows (no random split — use the official split for comparability with published scores).
- Metrics: **accuracy, precision, recall, F1-score** (macro + per-class), plus confusion matrix.
- **Safety lens:** recall on `jailbreak` matters most (catching attacks > not annoying benign users). Report F1 as headline.

**Our measured baseline (official split, TF-IDF 5k, 1–2 grams; verified from `output/results/metrics.csv`):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 97.71% | 99.26% | 96.40% | 97.81% |
| Random Forest | 97.71% | 100.0% | 95.68% | 97.79% |
| Naive Bayes | 96.18% | 97.08% | 95.68% | 96.38% |

**Expected outcomes:** LR ≈ 97–98% F1; NB slightly lower; RF similar to LR but slower to train. All three beat rule-based baselines. Published SOTA on this dataset ≈ 99.6% (fine-tuned DeBERTa) — our gap is expected and explainable (classic features vs. deep learning).

**Hyperparameter tuning (optional):** `GridSearchCV` on `C` (LR), `n_estimators`/`max_depth` (RF) with 5-fold CV.

---

### Phase 6 — Model Selection & Error Analysis

**Goal:** pick the best model, understand where it fails.

1. Compare all metrics; choose best model (likely Logistic Regression — best F1, most interpretable).
2. **Confusion matrix** — quantify false negatives (missed jailbreaks) vs false positives (benign flagged).
3. **Error inspection:** print 10–15 misclassified test prompts. Categorize: short sneaky jailbreaks? benign prompts with jailbreak-y words?
4. **Feature importance (LR):** top positive-weight tokens = jailbreak signals; top negative = benign signals. Export for report.

**Expected finding:** misclassified cases are usually *novel* jailbreak phrasings the 2023-era training data never saw — direct evidence for the JailbreaksOverTime drift argument. Great material for the report's "limitations & future work."

---

### Phase 6.5 — Error Analysis & Interpretability (implemented)

`run_modeling.py` now exports the Logistic Regression misclassified test prompts and the full coefficient table, plus a top-signal visualization. These artifacts support the report's error analysis and interpretability claims.

- `output/results/misclassified_test_prompts.csv`
- `output/results/logistic_feature_coefficients.csv`
- `output/results/logistic_feature_importance.png`

### Phase 7 — Reporting & Deliverables

**Goal:** package results into the final submission.

**Deliverables:**
- `output/models/` — saved models (`joblib.dump`) + vectorizer.
- `output/results/` — metrics CSV, confusion-matrix PNG, feature-importance chart.
- `output/eda/` — all EDA plots.
- `REPORT.md` — full write-up: problem, dataset, methods, results table, insights, limitations.

**Report skeleton:**
1. Problem statement
2. Dataset (verified stats + limitations: small, 2023-era, artificial balance)
3. Methodology (cleaning → EDA → features → models)
4. Results (metrics table + confusion matrix)
5. Key insights (linguistic signals that predict jailbreaks)
6. Limitations & future work (drift, embeddings stretch goal, fine-tuned LLM comparison)

---

## 📈 Timeline & Milestones

| Phase | Deliverable | Estimated effort |
|---|---|---|
| 0 | Environment ready | 30 min |
| 1–2 | Cleaned dataset + inspection report | 1–2 h |
| 3 | EDA plots + insights | 2–3 h |
| 4 | Feature vectors + vectorizer saved | 1–2 h |
| 5–6 | Model comparison + best model + error analysis | 2–3 h |
| 7 | Report + final deliverables | 2 h |

**Total:** ~2–3 focused days.

---

## ✅ Success Criteria — Lecture 5 Data Preparation Audit

- [x] All 3 models trained on official split
- [x] Metrics table: accuracy, precision, recall, F1 per model
- [x] Best model ≥ 96% accuracy (verified: 97.71%)
- [x] Confusion matrix + error analysis documented
- [x] Top linguistic signals identified & visualized
- [x] All artifacts stored under `ds1/output/`
- [x] REPORT.md complete with limitations & future work
- [x] Train/test schema integration audit completed without merging official splits
- [x] Data aggregation assessed and correctly ruled out for independent prompt observations
- [x] Missing-value and duplicate analysis documented
- [x] Statistical profiling with mean/median/std/skewness/kurtosis completed
- [x] Correlation analysis completed
- [x] IQR and Z-score outlier analysis completed; valid extreme prompts retained
- [x] Log-transformation effect evaluated
- [x] SMOTE assessed and correctly not applied to balanced training data
- [x] Feature scaling, normalization, and standardization decisions documented; TF-IDF L2 normalization verified
- [x] Encoding, binning, and binarization documented/applied where relevant
- [x] Explicit tokenization/stopword/lowercasing audit completed
- [x] Stemming, lemmatization, POS, spell correction, NER, sentiment and punctuation stripping evaluated and intentionally not forced
- [x] Feature creation and chi-square feature-selection experiment completed
- [x] PCA/dimensionality-reduction decision documented
- [x] Lecture 5 technique applicability matrix generated at `output/data_preparation/technique_applicability.csv`

> **Metrics correction:** an older planning note reported 98.09% Logistic Regression accuracy. The current reproducible saved run in `output/results/metrics.csv` is 97.71%, and the report now uses the saved result as authoritative.

---

## 📚 References

1. Galinkin, E., Sablotny, M. (2024). *Improved Large Language Model Jailbreak Detection via Pretrained Embeddings.* arXiv:2412.01547.
2. Xie, Y. et al. (2024). *GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis.* ACL 2024.
3. Chen, G. et al. (2025). *LLM Jailbreak Detection for (Almost) Free!* Findings of EMNLP 2025.
4. Hawkins, J., Pramar, A., Beard, R., Chandra, R. (2025). *Machine Learning for Detection and Analysis of Novel LLM Jailbreaks.* arXiv:2510.01644.
5. Piet, J. et al. (2025). *JailbreaksOverTime: Detecting Jailbreak Attacks Under Distribution Shift.* arXiv:2504.19440.
6. Dataset: `jackhhao/jailbreak-classification` — https://huggingface.co/datasets/jackhhao/jailbreak-classification (Apache 2.0).
7. `traromal/AIccel_Jailbreak` — DeBERTa-v3-small fine-tuned on the same dataset (99.62% acc).
8. `madhurjindal/Jailbreak-Detector-2-XL` — Qwen2.5-0.5B + LoRA detector (99.48% acc).

### Phase 7.5 — Lecture 7 Inferential Statistics (implemented)

`inferential_statistics.py` implements and documents the applicable Lecture 7 methods using `cleaned_train.csv` only:

- point and interval estimation
- t-based 90%, 95% and 99% confidence intervals for means
- a clearly labeled z-formula demonstration using sample SD as a plug-in because population sigma is unavailable
- confidence-interval width comparison
- confidence intervals for the jailbreak proportion
- sample-size determination for a mean and proportion
- 5,000-resample percentile bootstrap intervals
- null/alternative hypotheses, alpha, test statistics, p-values and decisions
- one-tailed and two-tailed Welch independent t-tests
- one-tailed and two-tailed Mann–Whitney U tests
- explicit documentation that paired t-test and Wilcoxon signed-rank are not applicable to independent prompts
- one-way ANOVA and Kruskal–Wallis on punctuation ratio across three word-length groups
- inferential Pearson and Spearman correlation tests
- Chi-square test of independence for keyword presence × prompt class, with observed and expected tables
- simple OLS regression, best-fit line, prediction and residual plot
- multiple linear regression with coefficient standard errors, t statistics, p-values and confidence intervals
- Shapiro-Wilk normality and Levene variance-homogeneity diagnostics
- Lecture 7 applicability matrix and human-readable summary

Artifacts: `output/inferential_statistics/`.

### Phase 8 — Lecture 8 Machine Learning Compliance (implemented)

The Machine Learning lecture requirements are now explicitly implemented/documented without adding algorithms that are unrelated to the binary jailbreak-classification dataset.

- **Supervised-learning framing:** the labelled `type` target is explicitly documented as a binary classification task.
- **Official train/test split:** preserved exactly for final evaluation; the test set is not used for model selection.
- **Validation:** Stratified 5-Fold cross-validation was added on `cleaned_train.csv` only. The TF-IDF vectorizer is inside a scikit-learn Pipeline, so vocabulary fitting occurs independently within each fold.
- **Generalization diagnostics:** training vs held-out test accuracy and jailbreak F1 are exported for all three classifiers to assess overfitting/underfitting signals.
- **Bias assessment:** sampling, selection, measurement, labeling, historical, exclusion, evaluation, algorithmic, confirmation and observer bias are assessed using only evidence available in this dataset/project.
- **Data compliance/ethics:** privacy, provenance, fairness, transparency and safe-use limitations are documented. No unsupported demographic fairness analysis is fabricated because the dataset has no demographic attributes.
- **Bias-variance trade-off:** CV stability and train/test gaps are used alongside interpretability and test performance when selecting Logistic Regression.
- **Metric selection:** accuracy, precision, recall and F1 remain the classification metrics; jailbreak recall/F1 are emphasized because false negatives are more safety-critical for this project.
- **Confusion matrices:** already generated for every classifier.
- **Unsupervised learning:** marked not applicable because labels are available and the project objective is classification.
- **Regression metrics:** marked not applicable because the project target is not numerical.

Artifacts: `output/ml_audit/`.
