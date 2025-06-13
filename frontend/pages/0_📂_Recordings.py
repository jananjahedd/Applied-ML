"""Streamlit page for managing EDF files from a FastAPI backend."""

import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="EDF Recordings",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="📂",
)

st.title("📂 EDF Recordings")
st.markdown(
    """
Welcome to the Recordings Explorer. This page interfaces with the Sleep Stage Prediction API to manage and view
polysomnography (PSG) recordings. You can browse existing files, inspect their metadata,
and upload new recording/annotation pairs.
"""
)


def get_from_api(endpoint: str) -> Any:
    """Helper function to perform a GET request to the FastAPI backend."""
    full_url = f"{FASTAPI_BASE_URL}{endpoint}"
    print("Full URL for GET request:", full_url)  # Debugging line to check the URL
    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()  # Raises an exception for 4XX/5XX errors
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"🔴 Connection Error: Could not connect to the backend at `{full_url}`. "
            "Please ensure the FastAPI server is running.",
            icon="💔",
        )
        return None
    except requests.exceptions.HTTPError as e:
        st.error(
            f"🔴 HTTP Error: Received status code {e.response.status_code} from the API. "
            "Detail: `{e.response.text}`",
            icon="🔥",
        )
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🔴 An unexpected error occurred: {e}", icon="💥")
        return None


def post_to_api_for_upload(endpoint: str, edf_file: Any, hypno_file: Any) -> Any:
    """Helper function to perform a POST request for file uploads."""
    full_url = f"{FASTAPI_BASE_URL}{endpoint}"
    files = {
        "edf_file": (edf_file.name, edf_file.getvalue(), edf_file.type),
        "hypno_file": (hypno_file.name, hypno_file.getvalue(), hypno_file.type),
    }
    try:
        response = requests.post(full_url, files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"🔴 Connection Error: Could not connect to the backend at `{full_url}`.", icon="💔")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(
            f"🔴 Upload Failed (HTTP {e.response.status_code}): "
            f"{e.response.json().get('detail', 'No detail provided.')}",
            icon="🔥",
        )
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🔴 An unexpected error occurred during upload: {e}", icon="💥")
        return None


# Fetch all available recordings on page load
with st.spinner("Fetching recordings from the API..."):
    recordings_data = get_from_api("/recordings/")

if recordings_data:
    # Combine cassette and telemetry recordings for the selection dropdown
    all_recordings = {**recordings_data.get("cassette_files", {}), **recordings_data.get("telemetry_files", {})}

    if not all_recordings:
        st.warning("No recordings were found by the API. You can upload some using the form below.", icon="📂")
        # Initialize an empty dictionary to prevent errors later
        all_recordings_display = {}
    else:
        # Create a user-friendly dictionary for display and selection
        all_recordings_display = {
            f"ID {key}: {os.path.basename(rec['recording_path'])} ({rec['study_type']})": key
            for key, rec in all_recordings.items()
        }

    st.sidebar.header("Navigation")
    selected_recording_display = st.sidebar.selectbox(
        "Select a Recording to View Details", options=list(all_recordings_display.keys()), index=0
    )

    st.header("🔍 Recording Details")

    if selected_recording_display:
        selected_id = all_recordings_display[selected_recording_display]

        with st.spinner(f"Fetching details for Recording ID: {selected_id}..."):
            recording_details = get_from_api(f"/recordings/{selected_id}")

        if recording_details:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("File Information")
                st.markdown(
                    f"""
                - **Recording File**: `{os.path.basename(recording_details['recording_path'])}`
                - **Annotation File**: `{os.path.basename(recording_details['annotation_path'])}`
                """
                )

            with col2:
                st.subheader("Study Information")
                st.metric("Study Type", recording_details["study_type"])
                st.metric("Night Number", recording_details["night"])

            st.divider()

            st.subheader("Patient Details")
            patient_info = recording_details.get("patient", {})
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Patient Number", patient_info.get("number", "N/A"))
            p_col2.metric("Age", f"{patient_info.get('age', 'N/A')}")
            p_col3.metric("Sex", patient_info.get("sex", "N/A"))

    st.header("📚 Full Recordings Catalog")

    def create_recordings_df(data: Dict[str, Dict[str, str]], title: str) -> None:
        """Create and display a DataFrame of recordings."""
        if not data:
            st.info(f"No {title.lower()} recordings found.")
            return

        df_data = {
            "ID": list(data.keys()),
            "Recording File": [os.path.basename(rec["recording_path"]) for rec in data.values()],
            "Annotation File": [os.path.basename(rec["annotation_path"]) for rec in data.values()],
        }
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

    with st.expander("Sleep Cassette Recordings", expanded=True):
        create_recordings_df(recordings_data.get("cassette_files", {}), "Cassette")

    with st.expander("Sleep Telemetry Recordings", expanded=True):
        create_recordings_df(recordings_data.get("telemetry_files", {}), "Telemetry")


st.header("⬆️ Upload New Recording")
st.markdown(
    "Upload a new pair of EDF recording and Hypnogram annotation files. "
    "Ensure filenames follow the required format (`SC...` or `ST...`)."
)

with st.form("upload_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        edf_file = st.file_uploader("Select EDF Recording File (.edf)", type=["edf"], accept_multiple_files=False)
    with col2:
        hypno_file = st.file_uploader(
            "Select Hypnogram Annotation File (.edf)", type=["edf"], accept_multiple_files=False
        )

    submitted = st.form_submit_button("Upload Files")

    if submitted:
        if edf_file is not None and hypno_file is not None:
            with st.spinner("Uploading and processing files via API..."):
                upload_response = post_to_api_for_upload("/recordings/upload", edf_file, hypno_file)
                if upload_response:
                    st.success(f"✅ Success: {upload_response.get('message')}", icon="🎉")
                    st.info("Refreshing the page to show the new recording...", icon="🔄")
                    st.rerun()
        else:
            st.warning("Please select both an EDF recording and a Hypnogram file to upload.", icon="⚠️")
