# Spam Email Classifier

A machine learning project that classifies text messages as **spam** or **ham** (not spam) using TF-IDF features with two classifiers: **Naive Bayes** and **Linear SVM**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayush-Raipure/spam-email-classifier/blob/main/spam_email_classifier.ipynb)

## Overview

The notebook loads a labelled SMS dataset, converts the text into numerical features, trains two models, compares them, and lets you test your own messages.

**Pipeline:**

1. Load the SMS Spam Collection dataset
2. Split into train (80%) and test (20%) sets, stratified by label
3. Convert text to features with `TfidfVectorizer`
4. Train `MultinomialNB` and `LinearSVC`
5. Evaluate with accuracy, classification report, and confusion matrices
6. Predict on custom messages

## Dataset

SMS Spam Collection: about 5,572 messages labelled `ham` or `spam`. The data is imbalanced (spam is roughly 13% of messages), so the train/test split is stratified and recall on the spam class is worth checking alongside accuracy.

Source used in the notebook:
`https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv`

## Tech Stack

- Python 3
- pandas
- scikit-learn
- matplotlib

## Getting Started

### Run in Google Colab

Click the **Open in Colab** badge above and run all cells.

### Run locally

```bash
git clone https://github.com/Ayush-Raipure/spam-email-classifier.git
cd spam-email-classifier

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install pandas matplotlib scikit-learn jupyter
jupyter notebook
```

Then open `spam_email_classifier.ipynb` and run all cells.

## Usage

After training, test any message with:

```python
predict_spam("Congratulations! You've won a prize. Click here to claim!", model_name="SVM")
# -> "SPAM"
```

Available models: `"Naive Bayes"` and `"SVM"`.

## Results

Run the notebook to see accuracy, precision, recall, and confusion matrices for both models. Add your numbers here:

| Model       | Accuracy | Spam Precision | Spam Recall |
|-------------|----------|----------------|-------------|
| Naive Bayes | -        | -              | -           |
| SVM         | -        | -              | -           |

## Possible Improvements

- Use unigrams + bigrams (`ngram_range=(1, 2)`) and keep stop words
- Try `class_weight="balanced"` for the SVM
- Use cross-validation for a more reliable estimate
- Test other models such as Logistic Regression or Random Forest
- Note: this dataset is UK SMS data from around 2008, so modern phishing-style messages may be misclassified

## Project Structure

```
spam-email-classifier/
├── spam_email_classifier.ipynb
└── README.md
```

## Author

**Ayush Raipure**
GitHub: [@Ayush-Raipure](https://github.com/Ayush-Raipure)

## License

This project is open source and available for learning and personal use.
