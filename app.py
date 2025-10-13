import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# -------------------------------
# 1️⃣ Load dataset safely
# -------------------------------
st.title("💳 Credit Card Fraud Detection Model")

csv_path = os.path.join(os.path.dirname(__file__), "creditcard_sample.csv")

try:
    # Ensure proper comma separation
    data = pd.read_csv(csv_path, sep=",")
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found! Please upload 'creditcard_sample.csv' to your repository.")
    st.stop()
except pd.errors.ParserError:
    st.error("⚠️ Error reading CSV file. Ensure it's comma-separated and properly formatted.")
    st.stop()

# -------------------------------
# 2️⃣ Data preparation
# -------------------------------
st.write("### Dataset Overview")
st.write(data.head())

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Handle case if fraud data is very small
if len(fraud) == 0 or len(legit) == 0:
    st.error("⚠️ Dataset must contain both fraud and legitimate transactions.")
    st.stop()

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=min(len(fraud), len(legit)), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# -------------------------------
# 3️⃣ Train/test split & model
# -------------------------------
X = data.drop(columns="Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# -------------------------------
# 4️⃣ Display results
# -------------------------------
st.subheader("📊 Model Performance")
st.write(f"**Training Accuracy:** {train_acc:.2f}")
st.write(f"**Testing Accuracy:** {test_acc:.2f}")

# -------------------------------
# 5️⃣ User input for prediction
# -------------------------------
st.subheader("🔍 Predict a Transaction")
st.write("Enter all feature values separated by commas:")

input_text = st.text_input("Example: 0.1, -1.2, 2.3, ..., 0.05")
submit = st.button("Predict")

if submit:
    try:
        # Convert input string to numpy array
        features = np.array(input_text.split(","), dtype=np.float64)
        
        # Check if feature count matches training data
        if features.shape[0] != X.shape[1]:
            st.error(f"❌ Expected {X.shape[1]} features, but got {features.shape[0]}.")
        else:
            prediction = model.predict(features.reshape(1, -1))
            if prediction[0] == 0:
                st.success("✅ Legitimate Transaction")
            else:
                st.error("⚠️ Fraudulent Transaction Detected!")
    except ValueError:
        st.error("❌ Invalid input. Please enter only numbers separated by commas.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")




