import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'output', 'ml_audit')
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(BASE, 'cleaned_train.csv'))
test = pd.read_csv(os.path.join(BASE, 'cleaned_test.csv'))
X_train, y_train = train['prompt'].fillna(''), train['type']
X_test, y_test = test['prompt'].fillna(''), test['type']

models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'naive_bayes': MultinomialNB(),
    'random_forest': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
}

def pipeline(model):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ('model', model),
    ])

# 1) Train/test generalization diagnostics. Test data remains untouched by CV/tuning.
rows = []
for name, estimator in models.items():
    pipe = pipeline(estimator)
    pipe.fit(X_train, y_train)
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    train_f1 = f1_score(y_train, train_pred, pos_label='jailbreak')
    test_f1 = f1_score(y_test, test_pred, pos_label='jailbreak')
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    rows.append({
        'model': name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'accuracy_gap_train_minus_test': train_acc - test_acc,
        'train_jailbreak_f1': train_f1,
        'test_jailbreak_f1': test_f1,
        'jailbreak_f1_gap_train_minus_test': train_f1 - test_f1,
        'diagnostic': ('possible_overfitting_signal' if train_f1 - test_f1 >= 0.05 else
                       'possible_underfitting_signal' if train_f1 < 0.90 else 'no_strong_signal'),
    })
train_test_df = pd.DataFrame(rows)
train_test_df.to_csv(os.path.join(OUT, 'train_test_generalization.csv'), index=False)

# 2) Stratified 5-fold CV on training data only. TF-IDF is fitted inside each fold.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, estimator in models.items():
    scores = cross_validate(
        pipeline(estimator), X_train, y_train, cv=skf,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
            'f1': 'f1_macro',
        }, return_train_score=False, n_jobs=1,
    )
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        vals = scores[f'test_{metric}']
        cv_rows.append({
            'model': name,
            'metric': metric,
            'cv_folds': 5,
            'mean': vals.mean(),
            'std': vals.std(ddof=1),
            'min': vals.min(),
            'max': vals.max(),
        })
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(os.path.join(OUT, 'stratified_5fold_cv.csv'), index=False)

# 3) Bias/data-compliance audit grounded only in fields actually present in the dataset.
# No demographic fairness metrics are fabricated because no demographic/group attributes exist.
class_counts = train['type'].value_counts()
class_props = train['type'].value_counts(normalize=True)
length = train['prompt'].str.split().str.len()
char_len = train['prompt'].str.len()

def keyword_hit(text, terms):
    s = str(text).lower()
    return any(t in s for t in terms)

keywords = {
    'dan': ['dan'],
    'ignore': ['ignore'],
    'previous_instructions': ['previous instructions'],
    'no_restrictions': ['no restrictions'],
    'roleplay': ['roleplay', 'role play'],
    'unfiltered': ['unfiltered'],
}

bias_rows = [
    {'bias_type': 'Sampling bias', 'status': 'Risk identified', 'evidence': 'Dataset is a curated benchmark of 1,306 prompts rather than a documented random sample of real-world traffic.', 'project_action': 'State limited population generalization; recommend external/temporal validation.'},
    {'bias_type': 'Selection bias', 'status': 'Risk identified', 'evidence': 'Train/test files are a supplied benchmark split with a near 50/50 class balance.', 'project_action': 'Preserve official split for comparability and explicitly report that deployment prevalence may differ.'},
    {'bias_type': 'Measurement bias', 'status': 'Not directly testable', 'evidence': 'The project has prompt text and labels but no independent measurement instrument or annotation metadata.', 'project_action': 'Do not claim measurement bias is absent; document provenance/annotation limitations.'},
    {'bias_type': 'Labeling bias', 'status': 'Risk cannot be quantified', 'evidence': 'No annotator identities, agreement scores, or labeling protocol are supplied with the modeling files.', 'project_action': 'Treat labels as benchmark labels and flag lack of annotation-agreement evidence as a limitation.'},
    {'bias_type': 'Historical bias', 'status': 'Risk identified', 'evidence': 'The benchmark reflects a particular collection period and known jailbreak styles; it may not represent current attacks.', 'project_action': 'Recommend temporal validation and newer attack data.'},
    {'bias_type': 'Exclusion bias', 'status': 'Risk identified', 'evidence': 'The two available fields are prompt text and binary class; no demographic, language, platform, or attack-family metadata are available.', 'project_action': 'Do not infer subgroup fairness; expand metadata/data coverage before subgroup claims.'},
    {'bias_type': 'Evaluation bias', 'status': 'Mitigated in protocol; residual risk remains', 'evidence': 'A fixed official test split is used once, while model-selection CV is confined to training data. The benchmark may still differ from deployment traffic.', 'project_action': 'Report test results separately and add temporal/external validation in future work.'},
    {'bias_type': 'Algorithmic/model bias', 'status': 'Risk assessed', 'evidence': 'TF-IDF relies on surface lexical patterns and may miss semantically obfuscated attacks.', 'project_action': 'Use error analysis and compare with semantic/embedding models in future work.'},
    {'bias_type': 'Confirmation bias', 'status': 'Mitigated by comparison', 'evidence': 'Three classifiers are evaluated with the same held-out test protocol and multiple metrics rather than selecting on a single preferred score.', 'project_action': 'Keep metric-selection rationale explicit in the report.'},
    {'bias_type': 'Observer bias', 'status': 'Not directly testable', 'evidence': 'No data-collection observer metadata are supplied.', 'project_action': 'Document this evidence gap rather than asserting absence.'},
]
for name, terms in keywords.items():
    flags = train['prompt'].map(lambda x: keyword_hit(x, terms))
    if flags.any():
        props = pd.crosstab(flags, train['type'], normalize='columns')
        jb = float(props.loc[True, 'jailbreak']) if True in props.index and 'jailbreak' in props.columns else 0.0
        bn = float(props.loc[True, 'benign']) if True in props.index and 'benign' in props.columns else 0.0
        bias_rows.append({'bias_type': f'Lexical confounding: {name}', 'status': 'Observed', 'evidence': f'Keyword/phrase presence rate: jailbreak={jb:.3f}, benign={bn:.3f}.', 'project_action': 'Interpret lexical features as predictive signals, not causal evidence of jailbreak intent.'})

bias_df = pd.DataFrame(bias_rows)
bias_df.to_csv(os.path.join(OUT, 'bias_and_compliance_audit.csv'), index=False)

# 4) Data compliance/ethics checklist, without claiming legal certification.
compliance = pd.DataFrame([
    {'topic': 'Privacy / personal data', 'status': 'Limited evidence', 'finding': 'Dataset fields are prompt text and class only; no dedicated personal-data fields are present in the project schema.', 'report_requirement': 'Avoid claiming that arbitrary prompt text can never contain personal data.'},
    {'topic': 'Permission / provenance', 'status': 'Documented at project level', 'finding': 'Project documentation records the benchmark source and Apache 2.0 license.', 'report_requirement': 'Retain source attribution and license information in the final report.'},
    {'topic': 'Fairness / representativeness', 'status': 'Limitation', 'finding': 'No demographic or deployment-population metadata are available for subgroup fairness testing.', 'report_requirement': 'Do not report demographic fairness conclusions unsupported by the dataset.'},
    {'topic': 'Transparency', 'status': 'Implemented', 'finding': 'Models, metrics, confusion matrices, errors, and feature coefficients are exported.', 'report_requirement': 'Include methodology and limitations alongside scores.'},
    {'topic': 'Security / safe use', 'status': 'Implemented as project framing', 'finding': 'Detector is treated as a first-line classifier, not a guarantee of safety.', 'report_requirement': 'State that false negatives remain possible and model performance can drift.'},
], columns=['topic','status','finding','report_requirement'])
compliance.to_csv(os.path.join(OUT, 'data_compliance_ethics_audit.csv'), index=False)

# 5) Simple visual for report: train vs test F1.
plot_df = train_test_df.set_index('model')[['train_jailbreak_f1', 'test_jailbreak_f1']]
ax = plot_df.plot(kind='bar', figsize=(8, 5))
ax.set_ylabel('Jailbreak F1')
ax.set_xlabel('Model')
ax.set_title('Train vs Test Jailbreak F1 — Generalization Diagnostic')
ax.legend(['Train', 'Test'])
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'train_vs_test_f1.png'), dpi=160)
plt.close(fig)

# 6) Applicability matrix for the ML lecture.
app = pd.DataFrame([
    ('Problem identification', 'Implemented', 'Binary jailbreak vs benign classification.'),
    ('Data/features/model/evaluation elements', 'Implemented', 'Prompt text -> TF-IDF features -> classifiers -> held-out metrics.'),
    ('Training/test split', 'Implemented', 'Official supplied split preserved; test set held out.'),
    ('Validation set', 'Not required for current untuned baseline', 'No hyperparameter tuning/model selection on the test set; 5-fold CV on training data provides validation evidence.'),
    ('Stratified K-Fold cross-validation', 'Implemented', '5-fold stratified CV on training data only; TF-IDF is fit within each fold.'),
    ('Supervised learning', 'Implemented', 'Both train and test have known binary labels.'),
    ('Classification metrics', 'Implemented', 'Accuracy, macro precision/recall/F1 and jailbreak precision/recall/F1.'),
    ('Confusion matrix', 'Implemented', 'One matrix per classifier.'),
    ('Metric choice', 'Implemented/documented', 'Jailbreak recall and F1 emphasized because missed attacks are more safety-critical.'),
    ('Bias assessment', 'Implemented', 'Dataset/model bias risks documented; unsupported subgroup claims avoided.'),
    ('Data compliance and ethics', 'Implemented/documented', 'Privacy, provenance, fairness, transparency and safe-use limitations recorded.'),
    ('Overfitting diagnostic', 'Implemented', 'Train-vs-test generalization gaps reported.'),
    ('Underfitting diagnostic', 'Implemented', 'Train performance and CV/test performance reviewed for weak-fit signals.'),
    ('Bias-variance trade-off', 'Implemented/documented', 'Model complexity and generalization are discussed using CV and train/test diagnostics.'),
    ('Unsupervised learning', 'Not applicable', 'The project target is explicitly labelled binary classification.'),
    ('Regression metrics', 'Not applicable', 'Primary task is classification; regression would not answer the project question.'),
], columns=['lecture_requirement','status','evidence'])
app.to_csv(os.path.join(OUT, 'ml_lecture_applicability_matrix.csv'), index=False)

summary = {
    'train_rows': int(len(train)), 'test_rows': int(len(test)),
    'train_class_counts': {str(k): int(v) for k,v in class_counts.items()},
    'train_class_proportions': {str(k): float(v) for k,v in class_props.items()},
    'random_state': RANDOM_STATE,
    'cv': 'StratifiedKFold(n_splits=5, shuffle=True, random_state=42)',
    'test_set_used_in_cv': False,
    'demographic_fairness_analysis_possible': False,
    'reason': 'No demographic/subgroup attributes are present in the dataset schema.'
}
with open(os.path.join(OUT, 'ml_audit_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(train_test_df.to_string(index=False))
print('\n5-fold CV:')
print(cv_df.to_string(index=False))
print('\nSaved ML lecture audit artifacts to output/ml_audit/')
