# ============================================================
#  SENTIMENT ANALYSIS — Full ML Pipeline
#  Libraries: scikit-learn, nltk, pandas, matplotlib, seaborn
# ============================================================

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# ── Download NLTK assets ────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ============================================================
# 1.  SAMPLE DATASET  (replace with your own CSV or dataset)
# ============================================================
# Using a small hand-crafted dataset so the script runs
# without any downloads.  Swap this section for your own data.

SAMPLE_DATA = {
    "text": [
        # Positive
        "I absolutely loved this product! It works perfectly.",
        "Amazing quality and super fast shipping. Highly recommend!",
        "Best purchase I've made all year. Very happy.",
        "Fantastic experience from start to finish.",
        "The customer service was outstanding and very helpful.",
        "This exceeded all my expectations, truly brilliant.",
        "Great value for money, will buy again for sure.",
        "I'm very satisfied with this item, it's superb.",
        "Works exactly as described. Five stars all the way.",
        "Incredible product, my whole family loves it!",
        # Negative
        "Terrible product. Broke after one day of use.",
        "Very disappointed. Nothing like the description.",
        "Worst purchase I've ever made. Total waste of money.",
        "Poor quality and arrived completely damaged.",
        "Customer support was useless and rude.",
        "Do not buy this. It stopped working immediately.",
        "Awful experience. I want a full refund.",
        "The product smells bad and looks nothing like the photo.",
        "Complete garbage. I regret buying this.",
        "Horrible. Cheap materials and terrible finish.",
        # Neutral
        "The product arrived on time and is as described.",
        "It's okay, nothing special but does the job.",
        "Average quality. Not bad, not great.",
        "Packaging was fine. Product seems normal.",
        "Delivered quickly. Haven't tested it fully yet.",
        "It's a standard item. Works as expected.",
        "Decent product for the price. Nothing extraordinary.",
        "Pretty average. Might work for some people.",
        "Neither good nor bad. Just mediocre.",
        "It does what it says. No complaints, no praises.",
    ],
    "label": (
        ["positive"] * 10
        + ["negative"] * 10
        + ["neutral"] * 10
    ),
}

df = pd.DataFrame(SAMPLE_DATA)

print("=" * 60)
print("SENTIMENT ANALYSIS — ML Pipeline")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")


# ============================================================
# 2.  TEXT PREPROCESSING
# ============================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no", "never", "nor"}


def preprocess(text: str) -> str:
    """Clean and normalise a raw text string."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


df["clean_text"] = df["text"].apply(preprocess)
print("Sample after preprocessing:")
print(df[["text", "clean_text"]].head(3).to_string(index=False))
print()


# ============================================================
# 3.  TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"],
)

print(f"Train size : {len(X_train)}   Test size : {len(X_test)}\n")


# ============================================================
# 4.  MODEL DEFINITIONS  (each wrapped in a Pipeline)
# ============================================================

def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=10_000,
            sublinear_tf=True,
        )),
        ("clf", classifier),
    ])


models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    ),
    "Naive Bayes": make_pipeline(
        MultinomialNB(alpha=0.5)
    ),
    "Linear SVM": make_pipeline(
        LinearSVC(C=1.0, max_iter=2000, random_state=42)
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42)
    ),
    "Gradient Boosting": make_pipeline(
        GradientBoostingClassifier(n_estimators=100, random_state=42)
    ),
}


# ============================================================
# 5.  TRAINING & EVALUATION
# ============================================================

results = []

print("-" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'F1 (macro)':>12} {'CV F1':>10}")
print("-" * 60)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(
        pipeline, df["clean_text"], df["label"],
        cv=5, scoring="f1_macro"
    )
    cv_f1 = cv_scores.mean()

    results.append({"Model": name, "Accuracy": acc, "F1 Macro": f1, "CV F1": cv_f1})
    print(f"{name:<22} {acc:>10.4f} {f1:>12.4f} {cv_f1:>10.4f}")

print("-" * 60)

results_df = pd.DataFrame(results).sort_values("F1 Macro", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_pipeline   = models[best_model_name]

print(f"\n✅  Best model : {best_model_name}\n")


# ============================================================
# 6.  DETAILED REPORT FOR THE BEST MODEL
# ============================================================

y_pred_best = best_pipeline.predict(X_test)
print(f"Classification Report — {best_model_name}")
print(classification_report(y_test, y_pred_best, zero_division=0))


# ============================================================
# 7.  VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Sentiment Analysis — Results Dashboard", fontsize=14, fontweight="bold")

# --- 7a. Model Comparison Bar Chart ---
ax1 = axes[0]
bar_df = results_df.set_index("Model")[["Accuracy", "F1 Macro"]]
bar_df.plot(kind="bar", ax=ax1, colormap="coolwarm", edgecolor="black", width=0.6)
ax1.set_title("Model Comparison")
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.05)
ax1.tick_params(axis="x", rotation=30)
ax1.legend(loc="lower right")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# --- 7b. Confusion Matrix ---
ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=["positive", "neutral", "negative"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["positive", "neutral", "negative"])
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(f"Confusion Matrix\n({best_model_name})")

# --- 7c. Label Distribution ---
ax3 = axes[2]
label_counts = df["label"].value_counts()
colors = ["#4CAF50", "#F44336", "#2196F3"]
ax3.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax3.set_title("Label Distribution")

plt.tight_layout()
plt.savefig("sentiment_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊  Chart saved to  sentiment_results.png")


# ============================================================
# 8.  PREDICT ON NEW SENTENCES
# ============================================================

def predict_sentiment(texts: list[str], pipeline=best_pipeline) -> pd.DataFrame:
    """Predict sentiment for a list of raw strings."""
    cleaned = [preprocess(t) for t in texts]
    preds   = pipeline.predict(cleaned)
    # Probability support (not available for LinearSVC)
    try:
        probs = pipeline.predict_proba(cleaned)
        classes = pipeline.classes_
        prob_df = pd.DataFrame(probs, columns=classes)
    except AttributeError:
        prob_df = pd.DataFrame({"note": ["probabilities not available"] * len(texts)})

    out = pd.DataFrame({"text": texts, "prediction": preds})
    return pd.concat([out, prob_df], axis=1)


NEW_SENTENCES = [
    "I love this so much, absolutely fantastic!",
    "This is the worst thing I have ever bought.",
    "It arrived on time and works as expected.",
    "Not bad, but I expected a bit more for the price.",
    "Never buying from this store again. Disgusting quality.",
]

print("\n" + "=" * 60)
print("PREDICTIONS ON NEW SENTENCES")
print("=" * 60)
pred_df = predict_sentiment(NEW_SENTENCES)
print(pred_df.to_string(index=False))
print("\nDone! ✅")
