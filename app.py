from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)


@st.cache_resource
def load_model():
    model_path = (
        Path(__file__).parent
        / "models"
        / "random_forest_churn_pipeline.joblib"
    )
    return joblib.load(model_path)


model = load_model()

st.title("📡 Telecom Customer Churn Predictor")
st.write(
    "Enter customer account details to estimate churn risk using the "
    "Random Forest model from this project."
)

st.caption(
    "Portfolio demonstration using the IBM Telco Customer Churn benchmark "
    "dataset. This is not a Zain Sudan production system."
)

with st.form("customer_input_form"):
    st.subheader("Customer profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        senior_citizen = int(st.checkbox("Senior citizen"))
    with col3:
        partner = st.selectbox("Has partner?", ["Yes", "No"])
    with col4:
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])

    st.subheader("Account and billing")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )
    with col2:
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=70.0,
            step=0.01
        )
    with col3:
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="Total charges accumulated across the customer's tenure."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
    with col2:
        paperless_billing = st.selectbox(
            "Paperless billing?",
            ["Yes", "No"]
        )
    with col3:
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.subheader("Phone and internet services")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone service?", ["Yes", "No"])

    with col2:
        if phone_service == "No":
            multiple_lines = "No phone service"
            st.text_input("Multiple lines", value=multiple_lines, disabled=True)
        else:
            multiple_lines = st.selectbox("Multiple lines?", ["Yes", "No"])

    with col3:
        internet_service = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"]
        )

    st.subheader("Internet add-on services")

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

        st.info("Internet add-on services are unavailable without internet service.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            online_security = st.selectbox("Online security?", ["Yes", "No"])
            online_backup = st.selectbox("Online backup?", ["Yes", "No"])

        with col2:
            device_protection = st.selectbox(
                "Device protection?",
                ["Yes", "No"]
            )
            tech_support = st.selectbox("Tech support?", ["Yes", "No"])

        with col3:
            streaming_tv = st.selectbox("Streaming TV?", ["Yes", "No"])
            streaming_movies = st.selectbox(
                "Streaming movies?",
                ["Yes", "No"]
            )

    submitted = st.form_submit_button("Predict churn risk")


if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Recreate the four engineered features used in model training.
    customer_data["HasInternet"] = int(internet_service != "No")
    customer_data["HasPhone"] = int(phone_service == "Yes")
    customer_data["AutoPayment"] = int(
        payment_method in [
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    add_on_services = [
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
    ]
    customer_data["ServiceCount"] = sum(
        service == "Yes" for service in add_on_services
    )

    input_data = pd.DataFrame([customer_data])

    # Preserve the exact feature order used when the pipeline was trained.
    input_data = input_data.reindex(columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]
    churn_index = list(model.classes_).index("Yes")
    churn_probability = model.predict_proba(input_data)[0][churn_index]

    st.divider()
    st.subheader("Prediction result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted churn probability", f"{churn_probability:.1%}")

    with col2:
        st.metric(
            "Model prediction",
            "Likely to churn" if prediction == "Yes" else "Likely to stay"
        )

    if prediction == "Yes":
        st.warning(
            "The model flags this customer as a potential churn risk. "
            "Consider a retention review."
        )

        recommendations = []

        if tenure <= 12:
            recommendations.append(
                "Prioritize onboarding or an early-lifecycle service check-in."
            )
        if contract == "Month-to-month":
            recommendations.append(
                "Test an offer that increases the perceived value of a longer contract."
            )
        if internet_service == "Fiber optic":
            recommendations.append(
                "Review service experience and price perception for this internet plan."
            )
        if tech_support == "No":
            recommendations.append(
                "Consider a targeted technical-support outreach or trial."
            )
        if payment_method == "Electronic check":
            recommendations.append(
                "Consider testing an optional automatic-payment incentive."
            )

        if recommendations:
            st.write("**Candidate retention actions to test:**")
            for recommendation in recommendations:
                st.write(f"- {recommendation}")
    else:
        st.success(
            "The model does not currently flag this customer as a likely churner."
        )

    st.caption(
        "Predictions are model estimates, not guarantees. "
        "Recommendations are candidate actions based on observed project patterns."
    )