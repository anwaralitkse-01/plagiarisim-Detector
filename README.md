# 🔍 AI Plagiarism Detection System

An NLP-based **Plagiarism Detection System** that uses **TF-IDF** for text feature extraction and a **Support Vector Machine (SVM)** classifier to determine whether a given text pair is classified as plagiarism or non-plagiarism.

The project also includes an interactive **Streamlit web application** with a modern UI, allowing users to enter an original/source text and a suspected text and receive a classification result.

---

## 📌 Project Overview

Plagiarism detection is an important NLP application used in education, content creation, research, and document verification.

This project takes two pieces of text:

* **Source / Original Text**
* **Student / Suspected Text**

The text is cleaned and preprocessed using NLP techniques. TF-IDF is then used to convert the text into numerical features, and an SVM model performs the final classification.

### Workflow

```text
                User Input
                    │
                    ▼
          ┌───────────────────┐
          │   Source Text     │
          │   Student Text    │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Text Preprocessing│
          │ • Lowercasing     │
          │ • Punctuation     │
          │ • Stopwords       │
          │ • Cleaning        │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │     TF-IDF        │
          │ Feature Extraction│
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │       SVM         │
          │    Classifier     │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Prediction Result │
          │                   │
          │ Plagiarism /      │
          │ No Plagiarism     │
          └───────────────────┘
```

---

## 🚀 Features

* 🔍 NLP-based plagiarism classification
* 🤖 Support Vector Machine (SVM)
* 📊 TF-IDF feature extraction
* 🧹 Text preprocessing
* 🚀 Interactive Streamlit web application
* 🎨 Custom CSS-based modern UI
* 📄 Source and suspected text comparison
* 📈 Text statistics
* ⚡ Fast prediction using saved `.pkl` models

---

## 🧠 Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core programming language |
| Pandas           | Data manipulation         |
| NumPy            | Numerical operations      |
| NLTK             | NLP preprocessing         |
| Scikit-learn     | TF-IDF and SVM            |
| Streamlit        | Web application           |
| Pickle           | Model serialization       |
| HTML/CSS         | UI customization          |
| Jupyter Notebook | Model development         |

---

## 📂 Project Structure

```text
Plagiarism_Detector/
│
├── app.py
├── plagiarism_detection.ipynb
├── README.md
├── requirements.txt
├── svm.pkl
└── tfidf_vectorizer.pkl
``` 

### File Description

**`app.py`**

Streamlit application responsible for:

* Accepting user input
* Preprocessing text
* Loading the trained model
* Transforming text using TF-IDF
* Generating predictions
* Displaying results

**`svm.pkl`**

Saved trained Support Vector Machine classification model.

**`vectorizer.pkl`**

Saved TF-IDF vectorizer used to transform text into numerical features.

**`plagiarism_detection.ipynb`**

Jupyter Notebook containing the data analysis, preprocessing, feature extraction, model training, and evaluation workflow.

**`requirements.txt`**

Contains the Python dependencies required to run the project.

---

# 📊 Dataset

The dataset contains pairs of text representing source/original content and suspected or plagiarized content.

The important columns used in the project are:

```text
source_text
plagiarized_text
```

The target column represents whether the text pair belongs to the plagiarism class.

> Replace the dataset description above with the exact dataset name and source link if you are publishing the dataset details on GitHub.

---

# 🔎 Exploratory Data Analysis

Before training the model, Exploratory Data Analysis was performed to understand the dataset.

The EDA included:

* Dataset shape
* Column information
* Missing values
* Duplicate records
* Class distribution
* Text length analysis
* Word count analysis
* Word frequency analysis
* Text similarity analysis
* Distribution of text lengths across classes

Example analysis:

```python
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()
df["label"].value_counts()
```

---

# 🧹 Text Preprocessing

The text was processed using NLP preprocessing techniques.

### Preprocessing steps

1. Handle missing values
2. Convert text to lowercase
3. Remove URLs
4. Remove punctuation
5. Remove extra spaces
6. Remove English stopwords

Example:

```python
def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(
        word
        for word in text.split()
        if word not in stop_words
    )

    return text
```

---

# 📚 TF-IDF Feature Extraction

After preprocessing, the text was converted into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

TF-IDF gives higher importance to words that are important in a document but less common across the overall corpus.

Conceptually:

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

The trained vectorizer is saved as:

```text
vectorizer.pkl
```

This allows the Streamlit application to use the same feature representation that was used during model training.

---

# 🤖 Machine Learning Model

Several machine learning models can be evaluated during experimentation. In this project, **Support Vector Machine (SVM)** was selected based on the evaluation performed during model development.

SVM is a supervised machine learning algorithm that finds a decision boundary separating different classes.

For text classification, linear SVM is commonly effective because text data represented with TF-IDF produces high-dimensional sparse feature vectors.

The trained model is saved as:

```text
svm.pkl
```

---

# 📈 Model Evaluation

The model was evaluated using standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
```

> SVM accuracy:0.878, precision:0.89, recall:0.89, and F1-score:0.87.

---

# 🌐 Streamlit Application

The project includes a web interface built using Streamlit.

The user provides:

### Source Text

The original/reference text.

### Student/Suspected Text

The text that needs to be checked.

The application then:

```text
Input
  ↓
Preprocessing
  ↓
TF-IDF Transformation
  ↓
SVM Model
  ↓
Prediction
```

The result is displayed as either:

```text
⚠️ PLAGIARISM DETECTED
```

or:

```text
✅ NO PLAGIARISM DETECTED
```

---

# 💻 Installation

##  Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd Plagiarism_Detector
```

---


---

##  Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🖥️ Application Interface

The application provides:

* Source text input
* Suspected text input
* Word count
* Plagiarism classification
* Model information
* Text statistics
* Responsive Streamlit interface

---

# 📦 Model Files

The application requires these trained model files:

```text
svm.pkl
tfidf_vectorizer.pkl
```

Make sure both files are located in the same directory as `app.py`.

```text
Plagiarism_Detector/
│
├── app.py
├── svm.pkl
└── tfidf_vectorizer.pkl
```

---

# ⚠️ Important Note

This system performs **machine-learning-based plagiarism classification**.

A prediction of:

```text
PLAGIARISM
```

means that the trained SVM model classified the input according to patterns learned from the training dataset.

It should not be interpreted as definitive proof of academic plagiarism.

The model's performance depends on:

* Dataset quality
* Training data distribution
* Text preprocessing
* TF-IDF configuration
* SVM hyperparameters
* Similarity between real-world text and training data

---

# 🔮 Future Improvements

Possible improvements include:

* 🔹 Semantic similarity using Sentence Transformers
* 🔹 BERT-based plagiarism detection
* 🔹 Paraphrase detection
* 🔹 Document/PDF upload
* 🔹 Multiple document comparison
* 🔹 Highlighting potentially copied sentences
* 🔹 Similarity percentage
* 🔹 Database of previously submitted documents
* 🔹 Advanced NLP similarity features
* 🔹 Transformer-based models
* 🔹 Deployment using Streamlit Cloud

---

# 🎯 Learning Outcomes

Through this project, the following concepts were implemented:

* Exploratory Data Analysis
* NLP text preprocessing
* Stopword removal
* TF-IDF
* Text classification
* Support Vector Machines
* Model evaluation
* Model serialization
* Streamlit development
* Custom CSS
* Machine Learning deployment

---

# 👨‍💻 Author

**Anwar Ali**

Software Engineering Student
AI / Machine Learning / NLP Enthusiast

---

# ⭐ Acknowledgements

This project was developed as part of my journey in **Artificial Intelligence, Machine Learning, and Natural Language Processing**.

If you find this project useful, consider giving the repository a ⭐.

---
