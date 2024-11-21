import math
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import requests

# Page Configuration
st.set_page_config(page_title="Health Monitoring App", layout="wide")

# Title and Description
st.title("Health Monitoring App")
st.markdown("""
This app demonstrates:
- **Real-time health monitoring predictions** using heart rate, SpO2, and pulse rate.
- **Model comparison** for disease classification.
- **Fine-tuning of Random Forest** models.
""")

# Sidebar Menu
menu = st.sidebar.radio("Navigation", ["Generate Data", "Train Model", "Model Comparison", "Real-Time Prediction"])

# Generate Synthetic Data
if menu == "Generate Data":
    st.header("Data")
    data_size = st.slider("Number of records ", 1000, 20000, 10000)
    np.random.seed(52)
    heart_rate = np.random.randint(60, 120, data_size)
    spo2 = np.random.uniform(85, 100, data_size)
    pulse_rate = np.random.randint(60, 120, data_size)

    def label_conditions(hr, sp, pr):
        if hr > 100 and sp < 90 and pr > 100:
            return 'Hypoxemia with Tachycardia'
        elif hr > 100 and pr > 100:
            return 'Tachycardia'
        elif sp < 90:
            return 'Hypoxemia'
        else:
            return 'Healthy'

    labels = [label_conditions(hr, sp, pr) for hr, sp, pr in zip(heart_rate, spo2, pulse_rate)]
    df = pd.DataFrame({
        'heart_rate': heart_rate,
        'SpO2': spo2,
        'pulse_rate': pulse_rate,
        'disease_status': labels
    })
    st.write("Dataset:")
    st.dataframe(df.head())
    
    # Save Dataset
    if st.button("Save Dataset"):
        df.to_csv("synthetic_oximeter_pulse_data.csv", index=False)
        st.success("Dataset saved as 'synthetic_oximeter_pulse_data.csv'")

# Train the Model
if menu == "Train Model":
    st.header("Train a Disease Prediction Model")
    
    # Load data
    df = pd.read_csv("synthetic_oximeter_pulse_data.csv")
    df['disease_status'] = df['disease_status'].map({
        'Healthy': 0,
        'Tachycardia': 1,
        'Hypoxemia': 2,
        'Hypoxemia with Tachycardia': 3
    })
    X = df[['heart_rate', 'SpO2', 'pulse_rate']]
    y = df['disease_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    dump(model, "disease_prediction_model.pkl")
    st.success("Model trained and saved as 'disease_prediction_model.pkl'")
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    st.subheader("Evaluation Metrics")
    st.text(classification_report(y_test, y_pred, target_names=[
        'Healthy', 'Tachycardia', 'Hypoxemia', 'Hypoxemia with Tachycardia'
    ]))

# Model Comparison
if menu == "Model Comparison":
    st.header("Compare Different Models")
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    df = pd.read_csv("synthetic_oximeter_pulse_data.csv")
    df['disease_status'] = df['disease_status'].map({
        'Healthy': 0,
        'Tachycardia': 1,
        'Hypoxemia': 2,
        'Hypoxemia with Tachycardia': 3
    })
    X = df[['heart_rate', 'SpO2', 'pulse_rate']]
    y = df['disease_status']
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = scores.mean()
    st.write("Model Comparison Results:")
    st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=["Accuracy"]))

# Real-Time Prediction
if menu == "Real-Time Prediction":
    st.header("Real-Time Prediction")
    
    model = load("disease_prediction_model.pkl")
    st.write("Enter Health Parameters:")
    heart_rate = st.number_input("Heart Rate", 60, 120, 80)
    spo2 = st.number_input("SpO2", 85, 100, 95)
    pulse_rate = st.number_input("Pulse Rate", 60, 120, 80)
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'heart_rate': [heart_rate],
            'SpO2': [spo2],
            'pulse_rate': [pulse_rate]
        })
        prediction = model.predict(input_data)
        disease_map = {
            0: 'Healthy',
            1: 'Tachycardia',
            2: 'Hypoxemia',
            3: 'Hypoxemia with Tachycardia'
        }
        st.success(f"Predicted Disease Status: {disease_map[prediction[0]]}")

def fetch_data(url):
    try:
        # Make a GET request to fetch the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request fails

        # Parse JSON response
        data = response.json()

        # Extract last_entry_id
        last_entry_id = data["channel"]["last_entry_id"]
        first_feed = next(feed for feed in data["feeds"] if feed["entry_id"] == 1)
        second_feed = next(feed for feed in data["feeds"] if feed["entry_id"] == 3)
        # Find the feed with last_entry_id
        last_feed = next(feed for feed in data["feeds"] if feed["entry_id"] == last_entry_id)
        # Extract field values
        heart_rate = first_feed["field1"]
        pulse = last_feed["field2"]
        spo2 = second_feed["field3"]
        return heart_rate, pulse, spo2

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None


# Function to predict health status
def predict_health(heart_rate, pulse, spo2):
    try:
        heart_rate = int(heart_rate)
        pulse = int(pulse)
        spo2 = int(spo2)

        if heart_rate < 60 or heart_rate > 100:
            hr_status = "Abnormal"
        else:
            hr_status = "Normal"

        if pulse < 50 or pulse > 120:
            pulse_status = "Abnormal"
        else:
            pulse_status = "Normal"

        if spo2 < 95:
            spo2_status = "Low"
        else:
            spo2_status = "Normal"

        return f"Heart Rate: {hr_status}, Pulse: {pulse_status}, SpO₂: {spo2_status}"

    except ValueError:
        return "Invalid inputs for prediction"


# Streamlit app interface
st.title("Health Monitoring System")

# Input section
st.header("Manual Input")
manual_heart_rate = st.text_input("Enter Heart Rate (bpm)", "")
manual_pulse = st.text_input("Enter Pulse", "")
manual_spo2 = st.text_input("Enter SpO₂ (%)", "")

if st.button("Predict from Manual Input"):
    prediction = predict_health(manual_heart_rate, manual_pulse, manual_spo2)
    st.success(f"Prediction: {prediction}")

# Fetch data and display section
st.header("Prediction from ThingSpeak data :")

# URL to fetch live data
url = "https://api.thingspeak.com/channels/2754538/feeds.json?api_key=FWDLYBFK7I9ETQ8Q&results=100"

if st.button("Predict from API"):
    heart_rate, pulse, spo2 = fetch_data(url)
    if heart_rate and pulse and spo2:
        st.write("Fetched Data:")
        st.write(f"- Heart Rate: {heart_rate}")
        st.write(f"- Pulse: {pulse}")
        st.write(f"- SpO₂: {spo2}")

        # Prediction from fetched data
        prediction = predict_health(heart_rate, pulse, spo2)
        st.success(f"Prediction: {prediction}")