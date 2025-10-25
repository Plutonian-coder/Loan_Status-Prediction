import streamlit as st
import joblib
import pandas as pd

# Load the trained model
# Make sure the model file 'loan_prediction_model.pkl' is in the same directory
try:
    model = joblib.load('loan_prediction_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'loan_prediction_model.pkl' not found. Please ensure the model is trained and saved correctly.")
    st.stop() # Stop the app if the model is not found

# --- Streamlit UI ---

# Add a sidebar
st.sidebar.header("About the Loan Prediction App")
st.sidebar.write("This application predicts the likelihood of a loan being approved based on the provided applicant details.")
st.sidebar.write("It uses a trained Support Vector Machine (SVM) model.")

# Main content with a faux bank header
st.markdown("""
    <style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #004d99; /* A shade of blue often associated with banks */
        margin-bottom: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #004d99;
        color: white;
    }
    </style>
    <div class="main-header">SecureBank Loan Application Predictor</div>
""", unsafe_allow_html=True)

st.write("Please enter the applicant's details below:")

# Mapping dictionaries for categorical features
gender_map = {'Male': 1, 'Female': 0}
married_map = {'No': 0, 'Yes': 1}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 4}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'No': 1, 'Yes': 0}
property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

# Organize inputs using columns and expander
with st.expander("Applicant Personal Details"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ('Male', 'Female'))
        married = st.selectbox("Married", ('No', 'Yes'))
        dependents = st.selectbox("Dependents", ('0', '1', '2', '3+'))
    with col2:
        education = st.selectbox("Education", ('Graduate', 'Not Graduate'))
        self_employed = st.selectbox("Self Employed", ('No', 'Yes'))
        property_area = st.selectbox("Property Area", ('Rural', 'Semiurban', 'Urban'))

with st.expander("Loan and Financial Details"):
    col3, col4 = st.columns(2)
    with col3:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
    with col4:
        loan_amount = st.number_input("Loan Amount", min_value=0.0)
        loan_amount_term = st.selectbox("Loan Amount Term (in days)", (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0))
        credit_history = st.selectbox("Credit History", (0.0, 1.0))


# Section to display input values before prediction
st.subheader("Review Your Input:")
input_summary = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self Employed": self_employed,
    "Property Area": property_area,
    "Applicant Income": applicant_income,
    "Coapplicant Income": coapplicant_income,
    "Loan Amount": loan_amount,
    "Loan Amount Term (in days)": loan_amount_term,
    "Credit History": credit_history
}
st.json(input_summary) # Display input as JSON for review


# Create a button to trigger prediction
if st.button("Predict Loan Status"):
    # Convert categorical inputs to numerical
    gender_val = gender_map[gender]
    married_val = married_map[married]
    dependents_val = dependents_map[dependents]
    education_val = education_map[education]
    self_employed_val = self_employed_map[self_employed]
    property_area_val = property_area_map[property_area]

    # Create a DataFrame with the input values, matching the training data column order
    # Ensure the column order matches the training data X
    input_data = pd.DataFrame([[gender_val, married_val, dependents_val, education_val,
                                self_employed_val, applicant_income, coapplicant_income,
                                loan_amount, loan_amount_term, credit_history, property_area_val]],
                              columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                                       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("Based on the provided details, the loan is likely to be APPROVED.")
    else:
        st.error("Based on the provided details, the loan is likely to be REJECTED.")
