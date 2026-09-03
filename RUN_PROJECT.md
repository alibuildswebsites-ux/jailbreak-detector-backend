# Reproducible Project Runbook

## 1. Create environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Run the complete pipeline

```bash
.venv/bin/python prepare_data.py
.venv/bin/python data_preparation_audit.py
.venv/bin/python descriptive_statistics.py
.venv/bin/python eda_agent.py
.venv/bin/python run_modeling.py
.venv/bin/python inferential_statistics.py
.venv/bin/python ml_lecture_audit.py
```

## 3. Main outputs

- `output/data_preparation/` — Lecture 5 audit
- `output/descriptive_statistics/` — Lecture 6 descriptive statistics
- `output/eda/` — Lecture 6 visualization/EDA
- `output/models/` — saved classifiers and TF-IDF vectorizer
- `output/results/` — metrics, confusion matrices, feature importance and errors
- `output/inferential_statistics/` — Lecture 7 confidence intervals, bootstrapping, hypothesis tests, correlation tests, chi-square tests and regression
- `output/ml_audit/` — Lecture 8 validation, generalization, bias/compliance and applicability evidence

`test_prompt.py` can then be used for local inference.

## 4. Dataset discipline

The official train/test split remains separate. The vectorizer is fit only on cleaned training prompts. Descriptive statistics and EDA use the cleaned training split unless an artifact explicitly says otherwise.
