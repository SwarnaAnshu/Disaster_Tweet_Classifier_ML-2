# DisasterAlert: Real vs. Metaphorical Disaster Tweet Classifier



An end-to-end NLP and Machine Learning application that automatically classifies tweets as referring to a **real disaster** or **not**, helping emergency response teams and news organizations cut through social-media noise and prioritize genuine incidents in real time, deployed through an interactive Streamlit web app.

---

# 🚀 Project Overview

During emergencies such as earthquakes, floods, landslides, and cyclones, social media platforms are flooded with posts. Many tweets use disaster-related words metaphorically (e.g. *"this concert was on fire"*) rather than reporting an actual emergency, making manual monitoring slow and unreliable.

This project analyzes tweet text, cleans and vectorizes it using NLP techniques, and predicts whether it describes a genuine disaster — enabling automated, real-time triage of incoming social media posts.

---

# ✨ Key Features

### 🐦 Disaster Tweet Classification

* Binary classification: Real Disaster vs. Not a Disaster
* NLP-based text preprocessing
* TF-IDF feature extraction (unigrams + bigrams)
* Multinomial Naive Bayes classification
* Confidence score estimation

### 📂 Input Support

* Direct Text / Tweet Input
* Streamlit Text Area

### 🎯 Model Robustness Enhancements

* Vocabulary gap fixes for underrepresented disaster types (landslide, tsunami, avalanche)
* Hard-negative examples (weather chat, sports slang) to reduce false alarms
* Leakage-free hyperparameter tuning via imbalanced-learn Pipeline + GridSearchCV

### 🖥️ Interactive Web App

* Streamlit-based Web Application
* Dark Theme User Interface
* Sidebar with App Info, Model Details, and Tech Stack
* Real-Time Tweet Analysis
* Colored Result Box (🚨 Real Disaster / ✅ Not a Disaster)
* Confidence Percentage Displayed Directly on the Progress Bar
* Clear Button to Reset Input

---

# 📊 Dataset Overview

| Attribute      | Value                                    |
| -------------- | ----------------------------------------- |
| Dataset Source | Kaggle "NLP with Disaster Tweets"         |
| Total Records  | 11,370 Tweets                             |
| Classes        | 2 (Real Disaster / Not a Disaster)        |
| Target Column  | target                                    |
| Problem Type   | Binary Text Classification                |

### Features Used

```text
text
```

### Target Variable

```text
target
```

---

# 🛠️ Technology Stack

| Category                    | Technology                        |
| ---------------------------- | ---------------------------------- |
| Programming Language         | Python                             |
| Machine Learning             | Scikit-Learn (pinned to 1.8.0)      |
| Natural Language Processing  | NLTK                                |
| Feature Engineering          | TF-IDF Vectorization                |
| Classification Algorithm     | Multinomial Naive Bayes             |
| Imbalanced Data Handling     | imbalanced-learn (RandomOverSampler)|
| Web Framework                | Streamlit                           |
| Model Serialization          | Joblib                              |
| Deployment                   | Streamlit Community Cloud           |

---

# 🔄 Project Workflow

```text
Tweet Input
      ↓
Text Cleaning (lowercase, URL/HTML/emoji/number/punctuation removal)
      ↓
Tokenization, Stopword Removal, Lemmatization
      ↓
TF-IDF Vectorization
      ↓
Multinomial Naive Bayes Classification
      ↓
Disaster / Not-Disaster Prediction + Confidence Score
      ↓
Streamlit Dashboard
```

---

# 🧠 Machine Learning Pipeline

The final model was selected via leakage-free hyperparameter tuning, combining TF-IDF feature extraction with a tuned Multinomial Naive Bayes classifier.

```python
pipe_nb = ImbPipeline([
    ("ros", RandomOverSampler(random_state=42)),
    ("clf", MultinomialNB())
])

param_grid_nb = {"clf__alpha": [0.1, 0.5, 1.0, 2.0]}

grid_nb = GridSearchCV(pipe_nb, param_grid_nb, cv=5, scoring='f1', n_jobs=-1)
grid_nb.fit(X_train_vec, y_train)
```

### Saved Model Files

```text
trained_model.pkl
vectorizer.pkl
```

### Why TF-IDF?

TF-IDF transforms tweet text into numerical vectors by assigning greater importance to meaningful, distinctive terms (e.g. *landslide, volcano*) while reducing the impact of frequently occurring, generic words.

### Why Multinomial Naive Bayes?

Naive Bayes achieved the best F1-Score and Recall on the minority "Real Disaster" class — the metric that matters most here, since missing a real disaster (false negative) is more costly than a false alarm. KNN had higher raw Accuracy but much lower Recall, making it a worse fit despite the higher headline number.

### Why oversample inside a Pipeline?

Applying oversampling to the full training set *before* cross-validation causes duplicated minority-class rows to leak across folds, artificially inflating scores. Wrapping `RandomOverSampler` and the classifier in an `imblearn.pipeline.Pipeline` ensures oversampling happens fresh inside each fold — giving honest, reproducible tuning results.

---

# 🖥️ Streamlit Application

The project includes a fully interactive Streamlit application that allows users to paste a tweet and receive an instant prediction.

### Available Features

✅ Disaster Tweet Classification

✅ Direct Text Input

✅ Confidence Score Shown Directly on the Progress Bar

✅ Real-Time Prediction

✅ Sidebar with App Info, Model, Tech Stack, and Dataset Details

✅ Clear Button

✅ Dark Theme UI

---

# 🏗️ Project Architecture

```text
+---------------------------------------------------+
|                 Tweet Input Module                |
|              (Streamlit Text Area)                 |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|            NLP Text Preprocessing Layer            |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|          TF-IDF Feature Vectorization              |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|       Multinomial Naive Bayes Classification        |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|         Disaster / Not-Disaster Prediction          |
+---------------------------------------------------+
                         |
                         v
+---------------------------------------------------+
|              Streamlit Dashboard UI                |
+---------------------------------------------------+
```

---

# ⚙️ Installation Guide

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DisasterAlert.git
cd DisasterAlert
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `scikit-learn` is pinned to `1.8.0` in `requirements.txt` to exactly match the version the model was trained and pickled with. Using a different version can still work, but scikit-learn will print an `InconsistentVersionWarning` (harmless, but pinning avoids it entirely).

### Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Run the Application

```bash
streamlit run app.py
```

---

# 🚀 Usage

1. Launch the Streamlit application.
2. Paste a tweet into the text box.
3. Click **Predict**.
4. The system cleans and preprocesses the tweet text.
5. The trained machine learning model predicts Real Disaster / Not a Disaster.
6. A confidence percentage is displayed directly on the colored progress bar.
7. Use **Clear** to reset the input and try another tweet.

---

# 📁 Project Structure

```text
DisasterAlert/
│
├── disaster_tweet_capstone.ipynb
├── app.py
├── trained_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── Project_Report.pdf
│
└── dataset/
    └── tweets.csv
```

---

# 📈 Results

The developed system successfully demonstrates:

* Automated Disaster Tweet Classification
* NLP-Based Tweet Processing
* Leakage-Free Hyperparameter Tuning
* Real-World Vocabulary Gap Correction
* Real-Time Tweet Assessment
* Interactive Dashboard-Based Deployment

| Metric | Score |
|---|---|
| Accuracy | 84.9% |
| Precision (Disaster class) | 57% |
| Recall (Disaster class) | 78% |
| F1-Score | 65.9% |
| ROC-AUC | 0.904 |

The project combines machine learning and practical disaster-triage capabilities within a single application.

---

# 🔮 Future Scope

* Transformer Models (BERT, RoBERTa) for deeper contextual understanding
* Word Embeddings (Word2Vec, GloVe)
* Deep Learning Models (LSTM, GRU)
* Ensemble Methods
* Multi-Language Tweet Analysis
* Live Twitter/X API Integration for real-time streaming classification
* Cloud Deployment at Scale

---

# 🎓 Key Learning Outcomes

* Natural Language Processing (NLP)
* Text Cleaning and Preprocessing
* TF-IDF Feature Engineering
* Handling Imbalanced Datasets Correctly (leakage-free oversampling)
* Multinomial Naive Bayes Classification
* Binary Text Classification
* Hyperparameter Tuning with GridSearchCV
* Model Evaluation Beyond Accuracy (Precision, Recall, F1, ROC-AUC)
* Streamlit Application Development
* Model Deployment and Serialization

---

# 👨‍💻 Author

**Anshu Swarna**

Text-Based Machine Learning Capstone Project

Machine Learning | Data Science | NLP Enthusiast

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and enhance this project for learning and research purposes.
