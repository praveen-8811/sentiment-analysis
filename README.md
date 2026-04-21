# 🎭 Sentiment Analysis — Machine Learning Pipeline

## Overview

Welcome to the **Sentiment Analysis Machine Learning Pipeline**! This project provides a robust, end-to-end natural language processing (NLP) workflow designed to classify text sentiment into **Positive**, **Negative**, or **Neutral** categories. Built entirely in Python using scikit-learn, it handles everything from raw text preprocessing to model training, evaluation, and visualization.

**What is this project?**
A modular text classification system using TF-IDF feature extraction and multiple machine learning models.

**Why was it built?**
To establish a clear, standardized, and easily extendable baseline for NLP sentiment analysis tasks and to serve as a comprehensive template for future ML projects.

**What does it do?**
It processes raw text data, trains five different ML algorithms, evaluates them via 5-fold cross-validation and standalone test sets, and outputs detailed metrics and comparative visual dashboards.

---

## ✨ Features

- **Advanced Text Preprocessing**: Automated lemmatization, stopword removal, URL stripping, and case normalization.
- **Robust Feature Engineering**: TF-IDF vectorization with unigrams and bigrams.
- **Multi-Model Comparison**: Trains and evaluates:
  - Logistic Regression
  - Linear Support Vector Machine (SVM)
  - Naive Bayes
  - Random Forest
  - Gradient Boosting
- **Comprehensive Evaluation**: Metrics include Accuracy, F1-Score (Macro), Confusion Matrices, and Classification Reports.
- **Visual Insights**: Automatically generates a high-quality dashboard (`sentiment_results.png`) featuring model comparisons, confusion matrices, and label distributions.
- **Extensible Prediction Pipeline**: Easily predict sentiment on novel input sentences.

---

## 🛠️ Tech Stack

- **Language**: ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
- **Machine Learning**: ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
- **NLP Processing**: ![NLTK](https://img.shields.io/badge/NLTK-3.8-green)
- **Data Manipulation**: ![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue?logo=pandas)
- **Data Visualization**: ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-red) ![Seaborn](https://img.shields.io/badge/Seaborn-0.12-blue?logo=pandas)

---

## 📁 Clean Folder Structure

```
sentiment-analysis/
├── src/                      # Main codebase
│   └── sentiment_analysis.py # Core ML pipeline script
├── docs/                     # Documentation files (if applicable)
├── assets/                   # Output visuals, images, and dashboards
│   └── sentiment_results.png # Generated results dashboard
├── tests/                    # Testing and validation scripts
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

---

## ⚙️ Setup / Installation Guide

Follow these steps to get the project running locally.

### Prerequisites
- Python 3.8+ installed
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install scikit-learn nltk pandas matplotlib seaborn
```

### 3. Run the Pipeline
Execute the main script from the `src` directory:
```bash
python src/sentiment_analysis.py
```

*Note: The script automatically downloads necessary NLTK datasets (like WordNet and stopwords) upon first run.*

---

## 💻 Usage

To use this model on your own data, simply open `src/sentiment_analysis.py` and replace the `SAMPLE_DATA` dictionary with a pandas dataframe loaded from your own CSV file:

```python
# Replace the sample data block with:
df = pd.read_csv("your_dataset.csv")
```

The script will handle the rest, re-training the models on your data and outputting new visualizations into the `assets/` directory.

---

## 🖼️ Output / Screenshots

Upon successful execution, the script generates a multi-panel visual dashboard:
1. **Model Comparison Bar Chart**: Side-by-side performance metrics.
2. **Confusion Matrix**: Visualizing true vs. predicted classifications for the best model.
3. **Label Distribution**: Pie chart breaking down the input dataset.

*(You can find the generated image at `assets/sentiment_results.png`)*

### Sample Predictions Output:
```text
============================================================
PREDICTIONS ON NEW SENTENCES
============================================================
text                                                prediction
I love this so much, absolutely fantastic!          positive
This is the worst thing I have ever bought.         negative
It arrived on time and works as expected.           neutral
```

---

## 🧠 Explanation of Working

1. **Data Ingestion**: Raw text strings and corresponding labels are loaded into a Pandas DataFrame.
2. **Text Normalization (NLP)**: The `preprocess()` function strips out noise (URLs, special characters), normalizes case, removes standard English stop-words, and lemmatizes words to their base forms using NLTK's `WordNetLemmatizer`.
3. **Feature Extraction**: Text is converted to numerical vectors using `TfidfVectorizer`, weighting terms by their frequency in the document offset by their frequency in the corpus.
4. **Modeling Pipeline**: Each algorithm is chained with the TF-IDF vectorizer using an `sklearn` Pipeline. This prevents data leakage during cross-validation.
5. **Evaluation**: Models are evaluated using hold-out testing and K-fold Cross Validation. The system programmatically selects the highest-scoring model (based on Macro F1 score) to produce detailed diagnostics.

---

## 📊 Architecture / System Design

```text
[ Raw Data ] --> [ Text Cleaning & Normalization ] --> [ TF-IDF Vectorization ]
                                                                 |
                                                                 v
[ K-Fold Cross Validation ] <------------------------- [ Model Training (5 Classifiers) ]
                                                                 |
                                                                 v
[ Visual Diagnostics & Results ] <-------------------- [ Best Model Selection ]
```

---

## 🐞 Issues / Known Problems

- **Class Imbalance**: Depending on the input dataset, highly imbalanced classes can skew the accuracy metric. (Mitigated partially by focusing on Macro F1-Score).
- **Contextual Nuance**: TF-IDF struggles with advanced contextual sarcasm or multi-word colloquialisms compared to modern transformer-based models (e.g., BERT).

---

## 🔮 Future Improvements

- **Deep Learning Integration**: Incorporate HuggingFace Transformers (e.g., RoBERTa) as a parallel pipeline option.
- **Hyperparameter Tuning**: Implement `GridSearchCV` or `RandomizedSearchCV` for automated parameter optimization.
- **Web Interface**: Build a simple FastAPI or Flask REST API wrapper around the `predict_sentiment()` function.
- **Interactive Dashboards**: Migrate static Matplotlib plots to interactive Plotly or Streamlit dashboards.

---

## 🤝 Contribution Guide

Contributions are always welcome! 

1. Fork the project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request.

---

## 📦 Releases / Versions

- **v1.0.0**: Initial Release — Standard ML Pipeline implementation with scikit-learn.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
