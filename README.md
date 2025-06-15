# 🎬 MovieReviewSentimentAnalyzer

A sentiment analysis project that classifies movie review snippets from Rotten Tomatoes into **positive** or **negative** sentiment using classic machine learning models.

---

## 📌 Project Description

This project implements a **binary sentiment classifier** trained on a dataset of movie review snippets. The objective is to predict whether a given sentence expresses a **positive (1)** or **negative (0)** sentiment. 

Three models were developed and evaluated:
- ✅ **Perceptron Classifier**
- ✅ **Logistic Regression Classifier**
- ✅ **Custom Feature-Enhanced Logistic Regression**

---

## 📂 Tasks Implemented

### ✅ Part 1: Perceptron Classifier (Unigram Features)
- Implemented the Perceptron training algorithm
- Used sparse bag-of-words unigram features
- Reached **~74%+ accuracy** on dev set

### ✅ Part 2: Logistic Regression (Unigram)
- Trained a logistic regression classifier with cross-entropy loss
- Implemented evaluation metrics and log-likelihood tracking
- Reached **~78% dev accuracy**

### ✅ Part 3: Feature Engineering
- Created two advanced feature extractors:
  - **BigramFeatureExtractor**: adds adjacent word pair features
  - **BetterFeatureExtractor**: combines unigrams + bigrams, removes stopwords (except negators like *not*, *never*)
- Achieved **~78.5% accuracy** with logistic regression and enhanced features

---

## 💡 Key Concepts

- **Bag-of-Words (BoW)**: Represents text as word count vectors
- **Unigram & Bigram Features**: Single words vs adjacent word pairs
- **Perceptron**: Mistake-driven linear classifier with online weight updates
- **Logistic Regression**: Probabilistic linear classifier trained with gradient descent
- **Stopword Removal**: Improves generalization by removing common, non-informative words

---

## 📈 Findings & Observations

- **Stopword filtering** improved generalization and reduced overfitting
- **Bigrams alone** performed worse than unigrams but enhanced overall performance when combined
- **Logistic Regression** outperformed Perceptron due to probabilistic learning and smoother updates
- **Learning rate and epoch tuning** were critical in achieving target accuracy

---

## 🚧 Challenges Faced

- Designing sparse feature vectors for bigrams efficiently
- Preventing overfitting with feature-rich representations
- Achieving required dev accuracy (≥74% for perceptron, ≥77% for logistic regression)
- Debugging weight update logic and convergence behavior

---

## 🛠 Technologies Used

- Python 3.8+
- NumPy
- NLTK (for stopword filtering)
- Matplotlib (for training curve visualization)

---

## 📊 Final Accuracy

| Model                     | Dev Accuracy |
|--------------------------|--------------|
| Perceptron (Unigram)     | 74.3%        |
| Logistic Regression (Unigram) | 78.4%    |
| Logistic + Better Features | **78.5%** ✅ |

---

## 👩‍💻 Author

**Ruchitha Reddy Kudumula**  
*Graduate Student in Computer Science*  
[GitHub](https://github.com/ruchitha007)

---

## 📁 How to Run

```bash
# Train with Perceptron and Unigrams
python sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM

# Train with Logistic Regression and Better Features
python sentiment_classifier.py --model LR --feats BETTER --no_run_on_test
