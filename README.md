# Amazon-Review-Sentiment-Classifier - Review Satisfaction Prediction Using NLP
A Machine Learning & Natural Language Processing Project
 ___________________________________________________________________________________________________
## 📌 Project Overview
This project focuses on building a text-based sentiment classification system to predict whether a customer review is satisfied or unsatisfied.
Using Natural Language Processing (NLP) techniques and machine learning models, the goal is to analyze customer feedback and automatically determine the satisfaction level.

The project follows a complete end-to-end data science workflow:

- Data loading and exploratory analysis
- Text preprocessing & cleaning
- Vectorization using TF-IDF
- Model building (Logistic Regression, Random Forest)
- Model evaluation using multiple metrics
- Confusion matrix & ROC-AUC analysis
- Saving the trained model & vectorizer
- Final business recommendations
This project demonstrates proficiency in NLP, ML modeling, evaluation, and deployment-ready workflows.
_____________________________________________________________________________________________________

## 🧠 Problem Statement
Organizations receive thousands of free-text user reviews daily, making manual sorting impossible.
The objective of this project is to:

### Automatically classify reviews into “Satisfied” or “Not Satisfied” using ML & NLP.

This helps businesses:
- Monitor customer experience
- Prioritize dissatisfaction cases
- Improve service quality
- Generate real-time customer satisfaction dashboards
______________________________________________________________________________________________________

## 🗂️ Project Structure
<pre>
Review_Satisfaction_Prediction/
│
├── data/
│   └── reviews.csv
│
├── notebooks/
│   └── Review_Satisfaction_Prediction.ipynb
│
├── models/
│   ├── best_model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt
</pre>
_______________________________________________________________________________________________________ 

## 🛠️ Tech Stack & Libraries
### Languages & Tools
- Python
- Jupyter Notebook
- Git & GitHub

### Python Libraries
- pandas, numpy – Data handling
- matplotlib, seaborn – Visualization
- scikit-learn – Modeling & evaluation
- nltk – NLP preprocessing
- pickle / joblib – Model serialization
________________________________________________________________________________________________________

## 🧹 Data Preprocessing & NLP Pipeline
### 1. Text Cleaning
- Remove HTML tags
- Lowercasing
- Remove special characters, digits
- Remove stopwords
- Tokenization
- Lemmatization
### 2. Feature Engineering
- TF-IDF Vectorization
    - ngram_range=(1,2)
### 3. Train/Test Split
- 80/20 split
- Stratified sampling
- random_state=42 for reproducibility
__________________________________________________________________________________________________________

## 🤖 Models Implemented
### 1. Logistic Regression (Final Selected Model)
- Works well with TF-IDF
- Fast, interpretable, high performance

### 2. Random Forest Classifier
- Tried as a non-linear alternative
- Used for benchmarking
___________________________________________________________________________________________________________

## 📊 Model Evaluation Metrics
Evaluation includes:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
The ROC-AUC curve is plotted to visualize model discrimination.
___________________________________________________________________________________________________________

## 🧪 Results Summary
- Logistic Regression performed the best overall.
- High precision & recall for satisfied/unsatisfied classes.
- Strong ROC-AUC score indicating effective separation between classes.
- Random Forest performed reasonably but was slower and less interpretable.
___________________________________________________________________________________________________________

## 💼 Business Recommendations
Based on the analysis:
1. Automate satisfaction tagging to reduce manual review time.
2. Use model outputs to trigger alerts for negative reviews.
3. Create dashboards to track satisfaction trends weekly/monthly.
4. Perform root-cause analysis on frequently negative terms (n-grams).
5. Integrate model into customer support pipelines to prioritize cases.
