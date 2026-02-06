"""Streamlit app for PhishGuard.
Run with: streamlit run src/app.py
"""

import streamlit as st
import joblib
from utils import clean_text

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title='PhishGuard', layout='centered')
st.title('PhishGuard — AI Scam Email Detection System')
st.write('Paste an email (or subject + body) below to check whether it is likely a scam.')

model_path = os.path.join(BASE_DIR, 'models', 'phish_model.joblib')
pipeline = None  

if not os.path.exists(model_path):
    st.warning(
        'Model not found. Please train the model first by running '
        '`python src/train.py` and place the saved pipeline at '
        '`models/phish_model.joblib`.'
    )
else:
    pipeline = joblib.load(model_path)

user_input = st.text_area('Email text', height=200)

if st.button('Check email'):
    if pipeline is None:
        st.error('Model is not loaded. Please train the model first.')
    elif not user_input.strip():
        st.error('Please paste some email text to analyze.')
    else:
        clean = clean_text(user_input)
        pred = pipeline.predict([clean])[0]
        prob = pipeline.predict_proba([clean])[0][1]

        label = '⚠️ Likely Phishing' if pred == 1 else '☑ Likely Legitimate'
        st.markdown(f'### {label}')
        st.write(f'Phishing probability: **{prob:.3f}**')
        st.subheader('Cleaned text used for prediction:')
        st.write(clean)