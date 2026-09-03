# Lecture 6 — Descriptive Statistics

Source: `cleaned_train.csv` (1,031 training prompts). The official test split is not used in these descriptive calculations.

## Central tendency
- Word count: mean=207.54, median=108.00, mode=12.
- Character count: mean=1220.79, median=671.00, mode=114.
- The mean is well above the median for both features, consistent with strong right-skew and a long upper tail.
- 10% trimmed means: word_count=160.06; char_count=951.36.
- 20% trimmed means: word_count=137.69; char_count=820.47.

## Dispersion
- Word count: range=1969, sample variance=61671.71, sample SD=248.34, CV=119.66%, IQR=274.00.
- Character count: range=11902, sample variance=2068331.44, sample SD=1438.17, CV=117.81%, IQR=1642.50.

## Position and z-scores
- Q1, Q2/median, Q3, P10, P90 and P95 are provided in `percentiles_quartiles_iqr.csv`.
- `zscore_summary.csv` reports the number and rate of observations beyond |Z|>2 and |Z|>3. The ±3 threshold is the lecture's typical outlier flag.

## Shape and distribution
- Word-count skewness=2.33, kurtosis=8.67.
- Character-count skewness=2.33, kurtosis=9.36.
- Both variables are strongly right-skewed and heavy-tailed rather than approximately normal.
- Modality is reported using exploratory KDE peak counting in `modality_kde_summary.csv`; it should be interpreted as visual/EDA evidence rather than a formal hypothesis test.

## Applicability decisions
- A line plot is not used for the dataset because it has no time/order variable for a meaningful trend; the lecture describes line plots as generally useful for trends over time.
- Mean/variance/SD use sample formulas (`ddof=1`) because the project treats the supplied training observations as a sample for statistical estimation. `pandas describe()` is also provided for reference.
