import streamlit as st
import pandas as pd
import joblib

model = joblib.load("metaverse_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Metaverse Risk Predictor", page_icon="🔍")

st.title("Metaverse Transaction Risk Predictor")
st.write("Enter the transaction details to predict its risk category.")

with st.form("transaction_form"):
    hour_of_day = st.slider("Hour of Day", 0, 23, 12)
    amount = st.number_input("Transaction Amount", min_value=0.01, value=500.0)
    ip_prefix = st.number_input("IP Prefix", value=172.0)
    login_frequency = st.number_input(
        "Login Frequency", min_value=1, max_value=8, value=4
    )
    session_duration = st.number_input(
        "Session Duration", min_value=1, value=60
    )

    location_region = st.selectbox(
        "Location Region",
        ["Africa", "Asia", "Europe", "North America", "South America"]
    )

    purchase_pattern = st.selectbox(
        "Purchase Pattern",
        ["focused", "high_value", "random"]
    )

    age_group = st.selectbox(
        "Age Group",
        ["established", "veteran", "new"]
    )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_data = pd.DataFrame([{
        "hour_of_day": hour_of_day,
        "amount": amount,
        "ip_prefix": ip_prefix,
        "login_frequency": login_frequency,
        "session_duration": session_duration,
        "location_region": location_region,
        "purchase_pattern": purchase_pattern,
        "age_group": age_group
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "location_region",
            "purchase_pattern",
            "age_group"
        ],
        dtype=int
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.subheader("Prediction Result")

    if prediction == "low_risk":
        st.success(f"Predicted category: {prediction}")
    elif prediction == "moderate_risk":
        st.warning(f"Predicted category: {prediction}")
    else:
        st.error(f"Predicted category: {prediction}")

    st.caption(
        "This is a model prediction, not a guarantee that a transaction is safe or fraudulent."
    )
