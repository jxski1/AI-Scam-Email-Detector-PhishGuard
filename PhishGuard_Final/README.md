# AI Scam Email Detector

PhishGuard is a simple machine learning project that detects whether an email is spam/phishing or legitimate.

## Dataset

This project uses a real Kaggle spam email dataset reduced to 1,000 emails (500 spam, 500 legitimate).

File:
data/emails_reduced_1000.csv

## Run the project

pip install -r requirements.txt
python src/train.py --data data/emails_reduced_1000.csv
streamlit run src/app.py

# Docker utilization

This project can also be run inside a Docker container
If you choose to run it through docker, please download Docker Desktop

## License

MIT
