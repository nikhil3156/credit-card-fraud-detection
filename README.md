🔍 Project Overview

This project detects financial fraud by combining:

1️⃣ Supervised Model (Optional)

A credit-card fraud model that gives a fraud probability.

2️⃣ Unsupervised Model

An IsolationForest anomaly detector that catches unusual transaction patterns.

Hybrid Score = (Supervised Score + Anomaly Score)

This final score tells how risky a transaction is.

You can upload your transaction file in the Streamlit App, and it shows:

Fraud / risk predictions

High-risk flagged transactions

Anomaly score

Dashboard with visual charts

Downloadable results file

🧠 Why This Project Is Useful

Banks and companies deal with millions of transactions.
Most frauds do NOT look the same every time, so a hybrid approach works better.

This project helps:

✔ Detect hidden fraud patterns
✔ Highlight high-risk transactions
✔ Visualize fraud trends
✔ Create dashboards for business reporting

💡 Features

Upload any transaction CSV

Upload your own models (iso_model.pkl, credit_card_model.pkl)

Automatic feature engineering

Hybrid fraud scoring

Interactive Streamlit UI

Charts: score distribution, risk breakdown, anomalies

Export results to CSV

Ready for Power BI / Tableau dashboards

🚀 How to Run the App
pip install -r requirements.txt
streamlit run app.py

📁 Project Structure
project/
│── app.py                  → Streamlit dashboard
│── models/
│     ├── iso_model.pkl     → IsolationForest model
│     ├── credit_model.pkl  → (optional)
│── data/
│     └── raw/              → Your raw transaction files
│── outputs/
│     └── hybrid_scores.csv → App-generated results
│── assets/                 → Images
│── README.md
│── requirements.txt

📦 Input Data Format

Your CSV should have at least:

amount

step (time step or timestamp)

oldbalanceOrg

newbalanceOrig

oldbalanceDest

newbalanceDest

isFraud (only if you want accuracy evaluation)

The app can also work on unlabeled data.

🧪 Model Saving Format (Important)

Save your IsolationForest like this:

joblib.dump({
    "iso": iso,
    "scaler": scaler,
    "features": features
}, "iso_model.pkl")


This allows the Streamlit app to load it properly.
