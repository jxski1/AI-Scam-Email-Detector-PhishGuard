"""Train script for PhishGuard.
Usage:
    python train.py --data data/email_db_reduced.csv --model_out models/phish_model.joblib --vec_out models/tfidf.joblib
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from utils import clean_text

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'spam' not in df.columns:
        raise ValueError('CSV must contain text and spam columns')
    df = df.dropna(subset=['text','spam'])
    df['text_clean'] = df['text'].apply(clean_text)
    return df

def main(args):
    df = load_data(args.data)
    X = df['text_clean']
    y = df['spam']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    param_grid = [
        {'clf': [LogisticRegression(max_iter=1000)], 'clf__C': [0.1, 1.0, 10.0]},
        {'clf': [RandomForestClassifier()], 'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 10]}
    ]

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print('Best params:', grid.best_params_)
    print('Validation results:')
    preds = grid.predict(X_test)
    print(classification_report(y_test, preds, target_names=['legitimate','phishing']))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, preds))

    model_out = args.model_out or 'models/phish_model.joblib'
    vec_out = args.vec_out or 'models/tfidf.joblib'
    joblib.dump(grid.best_estimator_, model_out)
    print(f'Saved trained pipeline to {model_out}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
    '--data',
    type=str,
    default='data/email_db_reduced.csv',
    help='Path to CSV data'
    )
    parser.add_argument(
        '--model_out', 
        type=str, 
        default='models/phish_model.joblib', 
        help='Output path for trained model pipeline'
    )
    
    parser.add_argument(
        '--vec_out', 
        type=str, 
        default='models/tfidf.joblib', 
        help='(Not used)'
    )
    
    args = parser.parse_args()
    main(args)
