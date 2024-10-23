import re
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to extract numeric values from antibiotic strings (e.g., "R (>=16)" -> 16)
def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return float(match.group())
    return None

# Function to categorize antibiotic susceptibility based on symbols (R, I, S)
def categorize_susceptibility(value):
    if isinstance(value, str):
        if value.startswith("R"):
            return "Resistant"
        elif value.startswith("I"):
            return "Intermediate"
        elif value.startswith("S"):
            return "Susceptible"
    return None

# Cache data loading to prevent reloading on each interaction
@st.cache_data
def load_data(file):
    relevant_columns = ['Dept', 'Isolate', 'Specimen']
    columns_to_drop = ['OP/IP NO', 'Reg No', 'S No', 'Patient Name', 'Admission/Reg Dt', 'OrderDate', 'A / S', 'Ward']
    all_columns = pd.read_excel(file, nrows=0).columns
    antibiotic_columns = [col for col in all_columns if col not in relevant_columns + columns_to_drop]
    columns_to_load = relevant_columns + antibiotic_columns
    data = pd.read_excel(file, usecols=columns_to_load)
    return data, relevant_columns, antibiotic_columns
st.title("Antibiotics Prediction Application")
# Streamlit interface
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    with st.spinner("Analysing data..."):
        data, relevant_columns, antibiotic_columns = load_data(uploaded_file)

        # Filter and process data
        data_cleaned = data.dropna(subset=relevant_columns).copy()
        data_cleaned.loc[:, antibiotic_columns] = data_cleaned[antibiotic_columns].fillna("None")
        
        # Remove columns with all "None" values
        antibiotics_to_keep = [antibiotic for antibiotic in antibiotic_columns if data_cleaned[antibiotic].nunique() > 1]
        data_cleaned = data_cleaned[relevant_columns + antibiotics_to_keep]

        # Apply categorization and numeric extraction
        for antibiotic in antibiotics_to_keep:
            data_cleaned[f"{antibiotic}_category"] = data_cleaned[antibiotic].apply(categorize_susceptibility)
            data_cleaned[f"{antibiotic}_value"] = data[antibiotic].apply(extract_numeric)

    # Display the first 5 rows of the cleaned dataset
    st.write("Dataset Preview:")
    st.dataframe(data_cleaned.head(5))

    # One-hot encode categorical variables like 'Dept', 'Isolate', and 'Specimen'
    X = data_cleaned[relevant_columns]
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train models for each antibiotic based on the category
    models = {}
    for antibiotic in antibiotics_to_keep:
        y = data_cleaned[f"{antibiotic}_category"]
        X_filtered = X_encoded[y.notna()]
        y_filtered = y[y.notna()]

        if len(y_filtered) < 2:
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        models[antibiotic] = model

    # Get unique values for dropdown options
    dept_options = data_cleaned['Dept'].unique()
    isolate_options = data_cleaned['Isolate'].unique()
    specimen_options = data_cleaned['Specimen'].unique()

    # Dropdown fields for Dept, Isolate, Specimen
    new_dept = st.selectbox("Select Department", options=dept_options, key="dept")
    new_isolate = st.selectbox("Select Isolate", options=isolate_options, key="isolate")
    new_specimen = st.selectbox("Select Specimen", options=specimen_options, key="specimen")

    if st.button("Predict"):
        with st.spinner("Predicting antibiotics..."):
            if new_dept and new_isolate and new_specimen:
                new_data = pd.DataFrame({
                    'Dept': [new_dept],
                    'Isolate': [new_isolate],
                    'Specimen': [new_specimen]
                })

                # One-hot encode the new input to match the training data format
                new_data_encoded = pd.get_dummies(new_data).reindex(columns=X_encoded.columns, fill_value=0)

                # Predict resistance for each antibiotic
                resistant_antibiotics = []
                for antibiotic, model in models.items():
                    prediction = model.predict(new_data_encoded)
                    if prediction[0] == "Resistant":
                        resistant_antibiotics.append(antibiotic)

                # Display the antibiotics that can be given
                st.write("Getting the antibiotics...")
                if resistant_antibiotics:
                    for antibiotic in resistant_antibiotics:
                        st.write(f"**- {antibiotic}**")
                else:
                    st.write("**No suitable antibiotics found based on the input.**")
            else:
                st.warning("Please fill all the fields.")
