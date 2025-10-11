import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# --- Safe file path handling ---
data = pd.read_csv("https://raw.githubusercontent.com/nikhil3156/credit-card-fraud-detection/main/creditcard_2023.csv")


# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit App
st.title("💳 Credit Card Fraud Detection Model")
st.write(f"**Training Accuracy:** {train_acc:.2f}")
st.write(f"**Testing Accuracy:** {test_acc:.2f}")
st.write("Enter the transaction features separated by commas:")

# Input for feature values
input_df = st.text_input('Example: 0.1, -1.2, 2.3, ..., 0.05')
submit = st.button("Predict")

if submit:
    try:
        features = np.array(input_df.split(','), dtype=np.float64)
        prediction = model.predict(features.reshape(1, -1))
        if prediction[0] == 0:
            st.success("✅ Legitimate transaction")
        else:
            st.error("⚠️ Fraudulent transaction")
    except Exception as e:
        st.error("Invalid input. Please enter numeric values separated by commas.")






