import streamlit as st

st.set_page_config(
    page_title="Welcome page",
    page_icon="👋",
    layout="centered",
)

st.title("😴 Sleep Stage Prediction App! ✨")

st.markdown("""
This application allows you to choose data in EDF format and get predictions on your sleep stages using pre-trained Random Forest models.
""")

st.header("🚀 How to Use This App")
st.markdown("""
1.  **📂 Manage Recordings**: Head over to the **"Recordings"** page using the sidebar. Here you can:
    * Browse a catalog of existing EDF (European Data Format) and annotation files.
    * View detailed metadata for each recording, including patient information and study type.
    * Upload new EDF recording and hypnogram annotation file pairs to the system.

2.  **🤖 Run Predictions & Explore Models**: Navigate to the **"Models"** page to access the core functionality:
    * **Explore Pre-trained Models**: Explore various machine learning model configurations available (e.g., EEG-only, EEG+EMG+EOG).
    * **View Model Performance**: Access comprehensive performance metrics (accuracy, F1-score, confusion matrices) for each model, derived from training, validation, and test datasets.
    * **Run Sleep Stage Prediction**: Select any uploaded EDF recording and choose a model configuration to run the automated sleep stage prediction pipeline. Get a prediction, confidence scores, and a summary of sleep stage distribution for your recording.

---

""")

st.sidebar.success("Select a page above.")

