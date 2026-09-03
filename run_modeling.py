import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_MODELS = os.path.join(BASE, "output", "models")
OUT_RESULTS = os.path.join(BASE, "output", "results")
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

random_state = 42

train = pd.read_csv(os.path.join(BASE, "cleaned_train.csv"))
test = pd.read_csv(os.path.join(BASE, "cleaned_test.csv"))

X_train_raw, y_train = train["prompt"], train["type"]
X_test_raw, y_test = test["prompt"], test["type"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

joblib.dump(vectorizer, os.path.join(OUT_MODELS, "vectorizer.joblib"))

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=random_state
    ),
}

metric_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec_jb = precision_score(
        y_test, y_pred, pos_label="jailbreak", zero_division=0
    )
    rec_jb = recall_score(y_test, y_pred, pos_label="jailbreak", zero_division=0)
    f1_jb = f1_score(y_test, y_pred, pos_label="jailbreak", zero_division=0)

    metric_rows.append(
        {
            "model": name,
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_jailbreak": prec_jb,
            "recall_jailbreak": rec_jb,
            "f1_jailbreak": f1_jb,
        }
    )

    joblib.dump(model, os.path.join(OUT_MODELS, f"{name}.joblib"))

    cm = confusion_matrix(y_test, y_pred, labels=["benign", "jailbreak"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["benign", "jailbreak"])
    ax.set_yticks([0, 1], ["benign", "jailbreak"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_RESULTS, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close(fig)

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(os.path.join(OUT_RESULTS, "metrics.csv"), index=False)

# Error analysis + Logistic Regression interpretability artifacts.
lr = models["logistic_regression"]
lr_pred = lr.predict(X_test)
misclassified = test.loc[lr_pred != y_test].copy()
misclassified["predicted"] = lr_pred[lr_pred != y_test]
misclassified["correct_label"] = y_test[lr_pred != y_test].to_numpy()
misclassified[["prompt", "type", "predicted"]].to_csv(
    os.path.join(OUT_RESULTS, "misclassified_test_prompts.csv"), index=False
)

feature_names = vectorizer.get_feature_names_out()
coefficients = lr.coef_[0]
feature_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
feature_df["signal"] = np.where(feature_df["coefficient"] > 0, "jailbreak", "benign")
feature_df.sort_values("coefficient", ascending=False).to_csv(
    os.path.join(OUT_RESULTS, "logistic_feature_coefficients.csv"), index=False
)

fig, ax = plt.subplots(figsize=(9, 7))
top_pos = feature_df.nlargest(15, "coefficient").sort_values("coefficient")
top_neg = feature_df.nsmallest(15, "coefficient").sort_values("coefficient")
combined = pd.concat([top_neg, top_pos])
ax.barh(combined["feature"], combined["coefficient"])
ax.set_xlabel("Logistic Regression coefficient")
ax.set_ylabel("TF-IDF feature")
ax.set_title("Top Logistic Regression Signals")
ax.axvline(0, linewidth=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_RESULTS, "logistic_feature_importance.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {len(misclassified)} Logistic Regression misclassified test prompts.")
print("Saved Logistic Regression feature coefficients and importance plot.")

print("\n=== Final Metrics Table ===")
pd.set_option("display.width", 200)
print(metrics_df.to_string(index=False))
print("\n=== Classification Reports ===")
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
