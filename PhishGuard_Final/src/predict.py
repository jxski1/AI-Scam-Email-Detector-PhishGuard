"""Simple prediction script that uses the trained pipeline to predict a single email text.
Usage:
    python predict.py --model models/phish_model.joblib --text "Your account..."
"""

import argparse
import joblib
from utils import clean_text

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    pipeline = joblib.load(args.model)
    text = args.text
    text_clean = clean_text(text)
    prob = pipeline.predict_proba([text_clean])[0]
    pred = pipeline.predict([text_clean])[0]
    label = 'phishing' if pred==1 else 'legitimate'
    print('Input:', text)
    print('Cleaned:', text_clean)
    print(f'Prediction: {label} (phishing probability={prob[1]:.3f})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/phish_model.joblib')
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    main(args)
