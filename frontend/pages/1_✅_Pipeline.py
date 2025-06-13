"""Streamlit page for the actual pipeline/pedictions"""

import streamlit as st
import requests
import pandas as pd
import os

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

# Updated endpoints to reflect the /models API structure
# Note: PREDICT_ENDPOINT is no longer used directly as prediction uses a specific recording ID
PERFORMANCE_ENDPOINT = f"{FASTAPI_BASE_URL}/models/" # Will be appended with {config}/performance
PREDICT_FROM_RECORDING_ENDPOINT = f"{FASTAPI_BASE_URL}/models/{{model_id}}/predict/{{recording_id}}"


st.set_page_config(
    page_title="Sleep Stage Prediction Pipeline ✅",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_all_recordings_from_api():
    """Fetches all available EDF recordings from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/recordings/", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"🔴 Connection Error: Could not connect to the backend at `{FASTAPI_BASE_URL}`. Please ensure the FastAPI server is running.", icon="💔")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"🔴 HTTP Error: Received status code {e.response.status_code} from the API. Detail: `{e.response.text}`", icon="🔥")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🔴 An unexpected error occurred while fetching recordings: {e}", icon="💥")
        return None


def print_pipeline_summary(pipeline_results):
    """ Prints a summary of the pipeline results in a readable format."""
    if pipeline_results.get("processing_summary"):
        st.write("### Processing Summary")
        summary = pipeline_results["processing_summary"]
        st.write(
            f"**Total Recording Time:** {summary.get('total_recording_time_hours'):.2f} hours"
        )
        st.write(f"**Epoch Duration:** {summary.get('epoch_duration_seconds')} seconds")
        st.write(
            f"**Annotations from Hypnogram:** {summary.get('annotations_from_hypnogram')}"
        )
        st.write(
            f"**Features Extracted per Epoch:** {summary.get('features_extracted_per_epoch')}"
        )

        if summary.get("sleep_stage_distribution"):
            st.write("**Sleep Stage Distribution:**")
            st.json(summary["sleep_stage_distribution"])

        if summary.get("current_file_performance"):
            st.write(
                "### Performance Metrics on Current File (if ground truth available)"
            )
            perf_metrics = summary["current_file_performance"]
            st.write(f"**Note:** {perf_metrics.get('note')}")
            st.write(f"**Dataset Size:** {perf_metrics.get('dataset_size')}")
            st.write(
                f"**Overall Accuracy:** {perf_metrics.get('overall_metrics', {}).get('accuracy'):.4f}"
            )
            st.write(
                f"**Macro F1 Score:** {perf_metrics.get('overall_metrics', {}).get('macro_f1_score'):.4f}"
            )

            if perf_metrics.get("per_class_metrics"):
                st.write("**Per-Class Metrics:**")
                per_class_df = pd.DataFrame(perf_metrics["per_class_metrics"]).T
                st.dataframe(per_class_df)

            if perf_metrics.get("confusion_matrix"):
                st.write("**Confusion Matrix:**")
                cm_data = perf_metrics["confusion_matrix"]["matrix"]
                cm_labels = perf_metrics["confusion_matrix"]["labels"]
                cm_df = pd.DataFrame(cm_data, index=cm_labels, columns=cm_labels)
                st.dataframe(cm_df)


st.title("Sleep Stage Prediction Pipeline")
st.write(
    "This page allows you to run the sleep stage prediction pipeline on selected EDF files. "
    "You can choose the model configuration and view the performance metrics."
)

st.header(""" Select a configuration""")

# Fetch available configurations from the /models/ endpoint
try:
    available_configs_response = requests.get(f"{FASTAPI_BASE_URL}/models/")
    available_configs_response.raise_for_status()
    available_configs_data = available_configs_response.json()
    AVAILABLE_CONFIGS = [
        cfg for cfg, details in available_configs_data.get("available_configurations", {}).items()
        if details["available"]
    ]
    DEFAULT_CONFIG = available_configs_data.get("default_configuration", AVAILABLE_CONFIGS[0] if AVAILABLE_CONFIGS else "eeg_emg_eog")
except Exception as e:
    st.error(f"Failed to fetch available model configurations: {e}. Using default list.")
    AVAILABLE_CONFIGS = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"] # Fallback
    DEFAULT_CONFIG = "eeg_emg_eog"


config = st.selectbox(
    "Model Configuration",
    AVAILABLE_CONFIGS,
    index=AVAILABLE_CONFIGS.index(DEFAULT_CONFIG) if DEFAULT_CONFIG in AVAILABLE_CONFIGS else 0,
)

st.header("""Select an EDF File for Prediction""")

# Fetch all available recordings to allow selection
with st.spinner("Fetching available recordings..."):
    all_recordings_data = get_all_recordings_from_api()

combined_recordings = {}
if all_recordings_data:
    combined_recordings = {
        **all_recordings_data.get("cassette_files", {}),
        **all_recordings_data.get("telemetry_files", {})
    }

selected_recording_id = None
if not combined_recordings:
    st.warning("No recordings available. Please upload EDF files on the 'Recordings' page first.")
else:
    recording_options = {
        f"ID {r_id}: {os.path.basename(rec_info['recording_path'])} ({rec_info['study_type']})": r_id
        for r_id, rec_info in combined_recordings.items()
    }
    selected_recording_display = st.selectbox(
        "Select an EDF Recording for Prediction:",
        options=list(recording_options.keys()),
        help="Choose a recording from the list to apply the sleep stage prediction model."
    )
    selected_recording_id = recording_options[selected_recording_display]


st.markdown("""If you want to manage your recordings (upload new ones), please go to the Files page.""")

st.header("""Run the pipeline""")

if st.button("Run Prediction"):
    if not selected_recording_id:
        st.warning("Please select a recording to run the prediction.")
    else:
        # Use a session state to store and display results if the user navigates away and comes back
        if "running_pipeline_old_page" not in st.session_state:
            st.session_state.running_pipeline_old_page = {}

        st.subheader(f"Processing Recording ID: {selected_recording_id}")
        with st.spinner("Processing recording and predicting sleep stages... This may take a moment."):
            try:
                prediction_endpoint = PREDICT_FROM_RECORDING_ENDPOINT.format(
                    model_id=config,
                    recording_id=selected_recording_id
                )
                response = requests.post(prediction_endpoint, timeout=300) # Increased timeout
                response.raise_for_status()
                prediction_results = response.json()
                st.session_state.running_pipeline_old_page[selected_recording_id] = prediction_results

                st.success("Prediction pipeline completed successfully! 🎉")
                print_pipeline_summary(prediction_results)

            except requests.exceptions.RequestException as e:
                st.error(f"Error running pipeline for recording ID {selected_recording_id}: {e}")
                if e.response:
                    st.error(f"Server response: {e.response.text}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Display results if available in session state
if st.session_state.get("running_pipeline_old_page"):
    st.subheader("Pipeline Results (Latest Run)")
    st.divider()
    # Display the results of the most recent prediction
    latest_recording_id = list(st.session_state.running_pipeline_old_page.keys())[-1]
    latest_results = st.session_state.running_pipeline_old_page[latest_recording_id]
    st.markdown(f"For Recording ID: {latest_recording_id}")
    print_pipeline_summary(latest_results)


st.header("Model Performance Analysis")
st.markdown(
    "View the detailed performance metrics for the selected model configuration."
)

if st.button("Get Model Performance"):
    with st.spinner(f"Loading performance metrics for {config} configuration"):
        try:
            # Updated endpoint for model performance
            response = requests.get(f"{PERFORMANCE_ENDPOINT}{config}/performance", timeout=10)
            response.raise_for_status()
            performance_data = response.json()

            st.success(
                f"Performance metrics for the model with configuration **{config}** :"
            )

            st.subheader("Performance Summary")
            if performance_data.get("performance_summary"):
                summary_df = pd.DataFrame.from_dict(
                    performance_data["performance_summary"], orient="index"
                )
                st.dataframe(summary_df, use_container_width=True)

            st.subheader("Overfitting Analysis")
            if performance_data.get("overfitting_analysis"):
                overfitting = performance_data["overfitting_analysis"]
                st.write(f"**Accuracy Dropoff:** {overfitting.get('accuracy_dropoff', 'N/A')}")
                st.write(f"**F1 Score Dropoff:** {overfitting.get('f1_score_dropoff', 'N/A')}")
                st.write(f"**Overfitting Severity:** {overfitting.get('overfitting_severity', 'N/A')}")
                st.write(f"**Generalization Quality:** {overfitting.get('generalization_quality', 'N/A')}")
                
                vs_random = overfitting.get("vs_random_guessing")
                if vs_random:
                    st.write("**Overfitting vs Random Performance:**")
                    st.json(vs_random)

            st.header("Detailed Metrics")
            if performance_data.get("detailed_metrics"):
                for dataset_type, metrics in performance_data["detailed_metrics"].items():
                    if metrics:
                        st.subheader(f"{dataset_type.replace('_', ' ').title()} Set Metrics")
                        
                        if metrics.get("overall_metrics"):
                            st.write("**Overall Metrics:**")
                            overall_metrics_df = pd.DataFrame([metrics["overall_metrics"]]).T
                            overall_metrics_df.columns = ["Value"]
                            st.dataframe(overall_metrics_df, use_container_width=True)

                        if metrics.get("per_class_metrics"):
                            st.write("**Per-Class Metrics:**")
                            per_class_df = pd.DataFrame(metrics["per_class_metrics"]).T
                            st.dataframe(per_class_df, use_container_width=True)

                        if metrics.get("confusion_matrix"):
                            st.write("**Confusion Matrix:**")
                            cm_data = metrics["confusion_matrix"]["matrix"]
                            cm_labels = metrics["confusion_matrix"]["labels"]
                            cm_df = pd.DataFrame(cm_data, index=cm_labels, columns=cm_labels)
                            st.dataframe(cm_df, use_container_width=True)
                        
                        if metrics.get("class_distribution"):
                            st.write("**Class Distribution:**")
                            st.json(metrics["class_distribution"])

                    else:
                        st.info(f"No detailed metrics available for the {dataset_type.replace('_', ' ')} set.")

            st.subheader("Recommendations")
            if performance_data.get("recommendations"):
                st.json(performance_data["recommendations"])

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching performance metrics: {e}")
            if e.response:
                st.error(f"Server response: {e.response.text}")

