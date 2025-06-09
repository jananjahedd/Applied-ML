""" Streamlit page for the actual pipeline/pedictions"""

import streamlit as st
import requests

FASTAPI_BASE_URL = "http://127.0.0.1:8000"

PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/pipeline/predict-edf"
PERFORMANCE_ENDPOINT = f"{FASTAPI_BASE_URL}/pipeline/all-performance"
AVAILABLE_CONFIGS = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]

st.set_page_config(
    page_title="Sleep Stage Prediction Pipeline ✅",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_selected_files_from_api():
    """Fetches currently selected EDF files from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/files/selected")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI.")
        return None
    except requests.exceptions.RequestException as e:
        if e.response and e.response.status_code == 404:
            return {"selected_files": {}, "total_selected": 0}
        st.error(f"Error fetching selected files: {e}")
        return None


st.title("Sleep Stage Prediction Pipeline")
st.write(
    "This page allows you to run the sleep stage prediction pipeline on selected EDF files. "
    "You can choose the model configuration and view the performance metrics."
)

st.header(""" Select a configuration""")

config = st.selectbox(
    "Model Configuration",
    AVAILABLE_CONFIGS,
    index=AVAILABLE_CONFIGS.index("eeg_emg_eog"),
)

st.header("""Previously Selected EDF Files""")


selected_files = get_selected_files_from_api().get("selected_files", {})
if st.button("Refresh Selected Files"):
    if selected_files is None:
        st.error("Failed to get selected files.")
    elif not selected_files:
        st.info("No files selected. Please select files on the Files page.")
    else:
        st.write(f"Total selected files: {len(selected_files)}")
        st.write("Selected EDF files:")
        for file_id, file_name in selected_files.items():
            st.write(f"- **{file_name}** | **(ID: {file_id})**")

st.markdown(""" If you want to change your selection, please go to the Files page""")

st.header("""Run the pipeline""")

# if st.button("Run Pipeline"):
#     for file_id, file_name in selected_files.items():
#         st.write(f"Running pipeline for **{file_name}** (ID: {file_id})...")
#         try:
        