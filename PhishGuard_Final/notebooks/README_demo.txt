# Quick demo (run in an environment where dependencies are installed)
# 1) Train:
#    python src/train.py --data data/sample_emails.csv --model_out models/phish_model.joblib
# 2) Predict:
#    python src/predict.py --model models/phish_model.joblib --text "Click here to reset your password http://fake.example"
# 3) Run UI:
#    streamlit run src/app.py
