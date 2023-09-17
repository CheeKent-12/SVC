import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("Clean_dataset.csv")

# Create a sidebar
st.sidebar.title("Input Parameters")

# Define input fields for user
marital_status = st.sidebar.selectbox("Marital Status", df["Marital_status"].unique())
application_mode = st.sidebar.selectbox(
    "Application Mode", df["Application_mode"].unique()
)
application_order = st.sidebar.selectbox(
    "Application Order", df["Application_order"].unique()
)
course = st.sidebar.selectbox("Course", df["Course"].unique())
gender = st.sidebar.selectbox("Gender", df["Gender"].unique())
Daytime_evening_attendance = st.sidebar.selectbox(
    "Daytime_evening_attendance", df["Daytime_evening_attendance"].unique()
)
Previous_qualification = st.sidebar.selectbox(
    "Previous_qualification", df["Previous_qualification"].unique()
)
Mother_qualification = st.sidebar.selectbox(
    "Mother_qualification", df["Mother_qualification"].unique()
)
Father_qualification = st.sidebar.selectbox(
    "Father_qualification", df["Father_qualification"].unique()
)
Mother_occupation = st.sidebar.selectbox(
    "Mother_occupation", df["Mother_occupation"].unique()
)
Father_occupation = st.sidebar.selectbox(
    "Father_occupation", df["Father_occupation"].unique()
)
Displaced = st.sidebar.selectbox("Displaced", df["Displaced"].unique())
Debtor = st.sidebar.selectbox("Debtor", df["Debtor"].unique())
Tuition_fees_up_to_date = st.sidebar.selectbox(
    "Tuition_fees_up_to_date", df["Tuition_fees_up_to_date"].unique()
)
Scholarship_holder = st.sidebar.selectbox(
    "Scholarship_holder", df["Scholarship_holder"].unique()
)
Age_at_enrollment = st.sidebar.selectbox(
    "Age_at_enrollment", df["Age_at_enrollment"].unique()
)
# --------------------------------------------
Curricular_units_1st_sem_credited = st.sidebar.selectbox(
    "Curricular_units_1st_sem_credited",
    df["Curricular_units_1st_sem_credited"].unique(),
)
Curricular_units_1st_sem_enrolled = st.sidebar.selectbox(
    "Curricular_units_1st_sem_enrolled",
    df["Curricular_units_1st_sem_enrolled"].unique(),
)
Curricular_units_1st_sem_evaluations = st.sidebar.selectbox(
    "Curricular_units_1st_sem_evaluations",
    df["Curricular_units_1st_sem_evaluations"].unique(),
)
Curricular_units_1st_sem_approved = st.sidebar.selectbox(
    "Curricular_units_1st_sem_approved",
    df["Curricular_units_1st_sem_approved"].unique(),
)
Curricular_units_1st_sem_grade = st.sidebar.selectbox(
    "Curricular_units_1st_sem_grade", df["Curricular_units_1st_sem_grade"].unique()
)
Curricular_units_1st_sem_without_evaluations = st.sidebar.selectbox(
    "Curricular_units_1st_sem_without_evaluations",
    df["Curricular_units_1st_sem_without_evaluations"].unique(),
)
# ---------------------------------
Curricular_units_2nd_sem_credited = st.sidebar.selectbox(
    "Curricular_units_2nd_sem_credited",
    df["Curricular_units_2nd_sem_credited"].unique(),
)
Curricular_units_2nd_sem_enrolled = st.sidebar.selectbox(
    "Curricular_units_2nd_sem_enrolled",
    df["Curricular_units_2nd_sem_enrolled"].unique(),
)
Curricular_units_2nd_sem_evaluations = st.sidebar.selectbox(
    "Curricular_units_2nd_sem_evaluations",
    df["Curricular_units_2nd_sem_evaluations"].unique(),
)
Curricular_units_2nd_sem_approved = st.sidebar.selectbox(
    "Curricular_units_2nd_sem_approved",
    df["Curricular_units_2nd_sem_approved"].unique(),
)
Curricular_units_2nd_sem_grade = st.sidebar.selectbox(
    "Curricular_units_2nd_sem_grade", df["Curricular_units_2nd_sem_grade"].unique()
)
Curricular_units_2nd_sem_without_evaluations = st.sidebar.selectbox(
    "Curricular_units_2nd_sem_without_evaluations",
    df["Curricular_units_2nd_sem_without_evaluations"].unique(),
)

# Prepare user inputs as a dictionary
user_input = {
    "Marital_status": marital_status,
    "Application_mode": application_mode,
    "Application_order": application_order,
    "Course": course,
    "Gender": gender,
    "Daytime_evening_attendance": Daytime_evening_attendance,
    "Previous_qualification": Previous_qualification,
    "Mother_qualification": Mother_qualification,
    "Father_qualification": Father_qualification,
    "Displaced": Displaced,
    "Debtor": Debtor,
    "Tuition_fees_up_to_date": Tuition_fees_up_to_date,
    "Scholarship_holder": Scholarship_holder,
    "Age_at_enrollment": Age_at_enrollment,
    "Curricular_units_1st_sem_credited": Curricular_units_1st_sem_credited,
    "Curricular_units_1st_sem_enrolled": Curricular_units_1st_sem_enrolled,
    "Curricular_units_1st_sem_evaluations": Curricular_units_1st_sem_evaluations,
    "Curricular_units_1st_sem_approved": Curricular_units_1st_sem_approved,
    "Curricular_units_1st_sem_grade": Curricular_units_1st_sem_grade,
    "Curricular_units_1st_sem_approved": Curricular_units_1st_sem_approved,
    "Curricular_units_1st_sem_without_evaluations": Curricular_units_1st_sem_without_evaluations,
    # -----------------------
    "Curricular_units_2nd_sem_credited": Curricular_units_2nd_sem_credited,
    "Curricular_units_2nd_sem_enrolled": Curricular_units_2nd_sem_enrolled,
    "Curricular_units_2nd_sem_evaluations": Curricular_units_2nd_sem_evaluations,
    "Curricular_units_2nd_sem_approved": Curricular_units_2nd_sem_approved,
    "Curricular_units_2nd_sem_grade": Curricular_units_2nd_sem_grade,
    "Curricular_units_2nd_sem_approved": Curricular_units_2nd_sem_approved,
    "Curricular_units_2nd_sem_without_evaluations": Curricular_units_2nd_sem_without_evaluations,
}


# Function to convert user inputs into a feature vector
def convert_inputs_to_features(user_input, df):
    feature_names = [
        "Marital_status",
        "Application_mode",
        "Application_order",
        "Course",
        "Gender",
        "Daytime_evening_attendance",
        "Previous_qualification",
        "Mother_qualification",
        "Father_qualification",
        "Displaced",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Scholarship_holder",
        "Age_at_enrollment",
        "Curricular_units_1st_sem_credited",
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_evaluations",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_without_evaluations",
        # -----------------------
        "Curricular_units_2nd_sem_credited",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_evaluations",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_without_evaluations",
        # Add more feature names here
    ]
    features = []
    for feature_name in feature_names:
        feature_value = user_input.get(feature_name)
        # Find the corresponding column in the dataframe and encode the value
        feature_column = df[feature_name]
        encoded_value = feature_column[feature_column == feature_value].index[0]
        features.append(encoded_value)
    return np.array(features).reshape(1, -1)


# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Convert user input to features
    user_features = convert_inputs_to_features(user_input, df)

    # Load the trained SVM model
    svm_classifier = SVC(kernel="linear", random_state=42, probability=True)

    # Fit the model to your entire dataset (if not already done)
    X = df.drop(columns=["Target"]).to_numpy()
    y = df["Target"].to_numpy()
    svm_classifier.fit(X, y)

    # Make a prediction
    prediction = svm_classifier.predict(user_features)

    # Display the prediction result
    st.title("Prediction Result")
    if prediction[0] == 1.0:
        st.write("Predicted Outcome: Graduate")
    else:
        st.write("Predicted Outcome: Dropout")

# You can add more explanations or visuals as needed
