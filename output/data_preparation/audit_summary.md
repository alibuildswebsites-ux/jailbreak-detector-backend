# Data Preparation Audit — Lecture 5

All relevant techniques were evaluated; inappropriate transformations were documented rather than forced.

- Raw train/test: 1044/262 rows; cleaned: 1031/262.
- Missing values: train=0, test=0.
- Raw duplicate rows: train=11, test=0.
- TF-IDF matrix: 1031 documents × 5000 features; L2 normalization.
- Chi-square feature-selection audit: 5000 → 1000 features.

See CSV/PNG artifacts in `output/data_preparation/` for the complete calculations.