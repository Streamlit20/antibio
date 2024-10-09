import re
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to extract numeric values from antibiotic strings (e.g., "R (>=16)" -> 16)
def extract_numeric(value):
    if isinstance(value, str):
        # Use regular expression to find numbers in the string
        match = re.search(r'\d+', value)
        if match:
            return float(match.group())
    return 0  # Return 0 for any 'None' or missing values

# Function to categorize antibiotic susceptibility based on numeric values
def categorize_susceptibility(value):
    if value >= 16:
        return "Resistant"
    elif 1 < value < 16:
        return "Intermediate"
    else:
        return "Susceptible"

# Load Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    data = pd.read_excel(uploaded_file)

    # Display basic info and preview
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Columns of interest
    relevant_columns = ['Dept', 'Isolate', 'Specimen']  # Only keep 'Dept', 'Isolate', and 'Specimen'
    antibiotic_columns = [col for col in data.columns if col not in relevant_columns + ['Patient Name', 'Req No', 'A / S', 'Ward', 'Admission/Reg Dt', 'OrderDate']]  # Select only antibiotic columns

    # Filter data for relevant columns
    data_cleaned = data[relevant_columns + antibiotic_columns]

    # Handle missing values
    data_cleaned.dropna(subset=relevant_columns, inplace=True)  # Drop rows with NaN in 'Dept', 'Isolate', or 'Specimen'
    data_cleaned.fillna("None", inplace=True)  # Fill missing antibiotic data with 'None'

    # Option to select an antibiotic for prediction
    selected_antibiotic = st.selectbox("Select Antibiotic for Prediction", antibiotic_columns)

    # Apply extraction and categorization functions to antibiotic data
    data_cleaned[selected_antibiotic] = data_cleaned[selected_antibiotic].apply(extract_numeric)
    data_cleaned[selected_antibiotic] = data_cleaned[selected_antibiotic].apply(categorize_susceptibility)

    # Filter relevant columns for modeling
    X = data_cleaned[relevant_columns]  # Features: 'Dept', 'Isolate', 'Specimen'
    y = data_cleaned[selected_antibiotic]  # Target: selected antibiotic (categorical)

    # One-hot encode categorical variables like 'Dept', 'Isolate', and 'Specimen'
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions and show accuracy
    y_pred = model.predict(X_test)
    st.write(f"Accuracy for {selected_antibiotic}: {accuracy_score(y_test, y_pred)}")

    # Prediction for new Dept/Isolate/Specimen combination
    st.write("Predict bacterial resistance for new combination:")

    # Input fields for Dept, Isolate, Specimen
    new_dept = st.text_input("Enter Department")
    new_isolate = st.text_input("Enter Isolate")
    new_specimen = st.text_input("Enter Specimen")

    # Create a DataFrame for the new input
    if st.button("Predict"):
        new_data = pd.DataFrame({
            'Dept': [new_dept],
            'Isolate': [new_isolate],
            'Specimen': [new_specimen]
        })

        # One-hot encode the new input to match the training data format
        new_data_encoded = pd.get_dummies(new_data).reindex(columns=X_encoded.columns, fill_value=0)

        # Predict susceptibility
        prediction = model.predict(new_data_encoded)
        st.write(f"Predicted Resistance for {selected_antibiotic}: {prediction[0]}")
