import re
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Function to extract numeric values from antibiotic strings (e.g., "R (>=16)" -> 16)
def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return float(match.group())
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

# Streamlit layout
st.set_page_config(layout="wide")
st.title("Antibiotics Prediction Application")
st.image("https://media.istockphoto.com/id/1468430468/photo/medical-technology-doctor-use-ai-robots-for-diagnosis-care-and-increasing-accuracy-patient.jpg?s=612x612&w=0&k=20&c=KqQbjGMVakHNTJOeh3LVeiqwZWF4Kt5j3taoJXY4x80=", use_column_width=True)

# Sidebar for file upload
st.sidebar.text("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    with st.spinner("Analysing data..."):
        data, relevant_columns, antibiotic_columns = load_data(uploaded_file)

        # Filter and process data
        data_cleaned = data.dropna(subset=relevant_columns).copy()
        data_cleaned.loc[:, antibiotic_columns] = data_cleaned[antibiotic_columns].fillna("None")
        
        # Remove columns with all "None" values
        antibiotics_to_keep = [antibiotic for antibiotic in antibiotic_columns if data_cleaned[antibiotic].nunique() > 1]
        data_cleaned = data_cleaned[relevant_columns + antibiotics_to_keep]

        # Extract numeric resistance values without converting them to categories
        for antibiotic in antibiotics_to_keep:
            data_cleaned[f"{antibiotic}_value"] = data_cleaned[antibiotic].apply(extract_numeric)

    # Display the first 5 rows of the cleaned dataset in an improved table format
    st.write("### Dataset Preview:")
    st.dataframe(data_cleaned.head(5), use_container_width=True)

    # Display help note
    st.info("Please select options from the sidebar and click 'Predict' to get the antibiotics.")

    # Get unique values for dropdown options
    dept_options = data_cleaned['Dept'].unique()
    isolate_options = data_cleaned['Isolate'].unique()
    specimen_options = data_cleaned['Specimen'].unique()

    # Sidebar dropdown fields for Dept, Isolate, Specimen
    new_dept = st.sidebar.selectbox("Select Department", options=dept_options, key="dept")
    new_isolate = st.sidebar.selectbox("Select Isolate", options=isolate_options, key="isolate")
    new_specimen = st.sidebar.selectbox("Select Specimen", options=specimen_options, key="specimen")

    # One-hot encode categorical variables like 'Dept', 'Isolate', and 'Specimen'
    X_encoded = pd.get_dummies(data_cleaned[relevant_columns], drop_first=True)

    # Train models for each antibiotic based on the numeric resistance values
    models = {}
    for antibiotic in antibiotics_to_keep:
        y = data_cleaned[f"{antibiotic}_value"]
        X_filtered = X_encoded.loc[y.notna()]
        y_filtered = y[y.notna()]

        # Ensure consistent length between X and y
        if len(X_filtered) != len(y_filtered):
            st.warning(f"Skipping {antibiotic} due to inconsistent data lengths.")
            continue

        if len(y_filtered) < 2:
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        models[antibiotic] = model

    # Show the predict button only after the file is uploaded and values are selected
    if new_dept and new_isolate and new_specimen:
        if st.sidebar.button("Predict"):
            with st.spinner("Predicting antibiotics..."):
                new_data = pd.DataFrame({
                    'Dept': [new_dept],
                    'Isolate': [new_isolate],
                    'Specimen': [new_specimen]
                })

                # One-hot encode the new input to match the training data format
                new_data_encoded = pd.get_dummies(new_data).reindex(columns=X_encoded.columns, fill_value=0)

                # Predict resistance for each antibiotic based on the numeric value threshold
                threshold = 32  # Define a threshold value for resistance
                resistant_antibiotics = {}
                for antibiotic, model in models.items():
                    prediction = model.predict(new_data_encoded)
                    if prediction[0] >= threshold:  # Only consider antibiotics above the threshold
                        resistant_value = prediction[0]
                        resistant_antibiotics[antibiotic] = resistant_value

                st.write("##### Predicted Antibiotics:")
                if resistant_antibiotics:
                    cols = st.columns(3)  # Creates a grid layout with 3 columns
                    per_col = len(resistant_antibiotics) // 3 + (len(resistant_antibiotics) % 3 > 0)
                    
                    # Distribute antibiotics among columns
                    for i, antibiotic in enumerate(resistant_antibiotics.keys()):
                        col = cols[i % 3]
                        with col:
                            st.markdown(
                                f"<ul style='padding-left: 20px; font-family: serif;'><li style='font-size:.9em;'>{antibiotic}</li></ul>", 
                                unsafe_allow_html=True
                            )

                    # Plot the line chart with pointers and hover text using Plotly
                    st.write("### Resistance Values Plot:")
                    df_plot = pd.DataFrame(list(resistant_antibiotics.items()), columns=['Antibiotic', 'Resistance Value'])
                    fig = px.line(
                        df_plot, 
                        x='Antibiotic', 
                        y='Resistance Value', 
                        markers=True,
                        title='Resistance Values per Antibiotic'
                    )
                    fig.update_traces(
                        mode='lines+markers',
                        hovertemplate='<b>%{x}</b>: %{y}'
                    )
                    fig.update_layout(
                        xaxis_title='Antibiotic',
                        yaxis_title='Resistance Value',
                        xaxis=dict(tickangle=-45),  # Rotate x-axis labels for better readability
                        showlegend=False
                    )

                    # Display both antibiotics and plot simultaneously
                    st.plotly_chart(fig, use_container_width=True)
                    

                else:
                    st.write("**No suitable antibiotics found based on the input.**")
