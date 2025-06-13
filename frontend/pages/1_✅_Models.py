"""Streamlit page for the actual pipeline/pedictions"""

import streamlit as st
import requests
import pandas as pd
import os

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

AVAILABLE_MODELS_ENDPOINT = f"{FASTAPI_BASE_URL}/models/"
MODEL_DETAILS_ENDPOINT = f"{FASTAPI_BASE_URL}/models/" # /{model_id}
MODEL_PERFORMANCE_ENDPOINT = f"{FASTAPI_BASE_URL}/models/" # /{model_id}/performance
PREDICT_FROM_RECORDING_ENDPOINT = f"{FASTAPI_BASE_URL}/models/{{model_id}}/predict/{{recording_id}}"


st.set_page_config(
    page_title="Sleep Stage Prediction Pipeline ✅",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("😴 Sleep Stage Prediction Pipeline")
st.markdown(
    "This page allows you to explore available machine learning models, view their "
    "performance, and run sleep stage predictions on your EDF recordings."
)

def get_from_api(endpoint: str) -> requests.Response:
    """Helper function to perform a GET request to the FastAPI backend with error handling."""
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.ConnectionError:
        st.error(f"🔴 Connection Error: Could not connect to the backend at `{FASTAPI_BASE_URL}`. Please ensure the FastAPI server is running.", icon="💔")
        st.stop() # Stop execution if no connection
    except requests.exceptions.HTTPError as e:
        st.error(f"🔴 HTTP Error: Received status code {e.response.status_code} from the API. Detail: `{e.response.text}`", icon="🔥")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"🔴 An unexpected error occurred: {e}", icon="💥")
        st.stop()


def display_model_performance(performance_data: dict):
    """Displays the detailed performance analysis of a model."""
    if not performance_data:
        st.info("No performance data available for this model.")
        return

    st.subheader("📊 Performance Summary")
    if performance_data.get("performance_summary"):
        summary_df = pd.DataFrame.from_dict(
            performance_data["performance_summary"], orient="index"
        )
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("📈 Overfitting Analysis")
    if performance_data.get("overfitting_analysis"):
        overfitting = performance_data["overfitting_analysis"]
        st.write(f"**Accuracy Dropoff:** {overfitting.get('accuracy_dropoff', 'N/A')}")
        st.write(f"**F1 Score Dropoff:** {overfitting.get('f1_score_dropoff', 'N/A')}")
        st.write(f"**Overfitting Severity:** {overfitting.get('overfitting_severity', 'N/A')}")
        st.write(f"**Generalization Quality:** {overfitting.get('generalization_quality', 'N/A')}")
        
        vs_random = overfitting.get("vs_random_guessing")
        if vs_random:
            st.write("**Performance vs. Random Guessing:**")
            st.json(vs_random)

    st.header("Detailed Metrics")
    if performance_data.get("detailed_metrics"):
        for dataset_type, metrics in performance_data["detailed_metrics"].items():
            if metrics:
                st.subheader(f"{dataset_type.replace('_', ' ').title()} Set Metrics")
                
                if metrics.get("overall_metrics"):
                    st.write("**Overall Metrics:**")
                    # Convert to DataFrame for better display of metrics
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
    
st.sidebar.header("Model Selection")
with st.spinner("Fetching available model configurations..."):
    try:
        models_response = get_from_api(AVAILABLE_MODELS_ENDPOINT)
        available_configs_data = models_response.json()
        available_configs = [
            cfg for cfg, details in available_configs_data.get("available_configurations", {}).items()
            if details["available"] # Only show truly available models
        ]
        default_config = available_configs_data.get("default_configuration", available_configs[0] if available_configs else None)

        if not available_configs:
            st.error("No trained models found. Please ensure models are in the 'results/' directory and the API is running correctly.")
            st.stop()

        selected_model_config = st.sidebar.selectbox(
            "Choose a Model Configuration:",
            options=available_configs,
            index=available_configs.index(default_config) if default_config in available_configs else 0,
            help="Select the combination of modalities (e.g., EEG, EMG, EOG) used for sleep stage prediction."
        )
    except Exception as e:
        st.error(f"Failed to load model configurations: {e}")
        st.stop()

st.header("✨ Model Details")
if selected_model_config:
    with st.spinner(f"Loading details for '{selected_model_config}' model..."):
        model_details_response = get_from_api(f"{MODEL_DETAILS_ENDPOINT}{selected_model_config}")
        model_details = model_details_response.json()

        if model_details:
            col_detail_1, col_detail_2, col_detail_3 = st.columns(3)
            with col_detail_1:
                st.metric("Configuration Name", model_details.get("config_name"))
                st.metric("Expected Features", model_details.get("expected_features_count"))
            with col_detail_2:
                st.write("**Modalities Used:**")
                for modality in model_details.get("modalities_used", []):
                    st.markdown(f"- {modality.upper()}")
            with col_detail_3:
                st.write("**Class Labels Legend:**")
                class_labels = model_details.get("class_labels_legend", {})
                for label_id, label_name in class_labels.items():
                    st.markdown(f"- `{label_id}`: {label_name}")
            
            performance_summary = model_details.get("performance_summary")
            if performance_summary:
                st.subheader("Quick Performance Snapshot")
                st.info(f"**Test Accuracy:** {performance_summary.get('test_accuracy', 'N/A'):.4f} | **Test Macro F1:** {performance_summary.get('test_macro_f1_score', 'N/A'):.4f}")
            else:
                st.info("No quick performance summary available for this model. Click 'Get Detailed Model Performance' below for full metrics.")


st.header("🚀 Model Performance Analysis")
st.markdown("Dive deep into the performance of the selected model configuration, including training, validation, and test metrics.")

if st.button(f"Get Detailed Model Performance for {selected_model_config}"):
    with st.spinner(f"Fetching performance metrics for '{selected_model_config}'..."):
        performance_response = get_from_api(f"{MODEL_PERFORMANCE_ENDPOINT}{selected_model_config}/performance")
        performance_data = performance_response.json()
        display_model_performance(performance_data)


st.header("🧠 Run Sleep Stage Prediction")
st.markdown("Select an EDF recording from the available files and run the sleep stage prediction pipeline using the chosen model configuration.")

# Fetch all available recordings for selection
st.markdown("First, let's load the available recordings from the 'Recordings' page.")
all_recordings_response = get_from_api(f"{FASTAPI_BASE_URL}/recordings/")
all_recordings_data = all_recordings_response.json()

combined_recordings = {
    **all_recordings_data.get("cassette_files", {}),
    **all_recordings_data.get("telemetry_files", {})
}

if not combined_recordings:
    st.warning("No recordings available. Please upload EDF files on the 'Recordings' page first.")
    selected_recording_id = None
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


if st.button("Run Prediction"):
    if not selected_recording_id:
        st.error("Please select a recording to run the prediction.")
    else:
        prediction_endpoint = PREDICT_FROM_RECORDING_ENDPOINT.format(
            model_id=selected_model_config,
            recording_id=selected_recording_id
        )
        st.info(f"Running prediction for Recording ID `{selected_recording_id}` with model `{selected_model_config}`...")
        with st.spinner("Processing recording and predicting sleep stages... This may take a moment."):
            try:
                predict_response = requests.post(prediction_endpoint, timeout=300) # Increased timeout for long processing
                predict_response.raise_for_status()
                prediction_results = predict_response.json()

                st.success("Prediction pipeline completed successfully! 🎉")
                st.subheader("Prediction Results")

                # Display general prediction info
                st.write(f"**Model Configuration Used:** `{prediction_results.get('model_configuration_used')}`")
                st.write(f"**Number of Segments Processed:** {prediction_results.get('num_segments_processed')}")
                st.write(f"**Total Recording Time:** {prediction_results.get('processing_summary', {}).get('total_recording_time_hours', 0):.2f} hours")
                
                st.write("**Sleep Stage Distribution:**")
                st.json(prediction_results.get('processing_summary', {}).get('sleep_stage_distribution', {}))

                # Display predictions table
                predictions_df = pd.DataFrame({
                    "Epoch Index": list(range(prediction_results.get('num_segments_processed', 0))),
                    "Predicted Stage ID": prediction_results.get('prediction_ids', []),
                    "Predicted Stage Label": prediction_results.get('predictions', [])
                })
                st.dataframe(predictions_df, use_container_width=True)

                if prediction_results.get('probabilities_per_segment'):
                    with st.expander("Show Detailed Probabilities per Segment"):
                        # Convert probabilities to DataFrame for better display
                        prob_df = pd.DataFrame(prediction_results['probabilities_per_segment'])
                        # Use class labels as column names if available
                        legend = prediction_results.get('class_labels_legend', {})
                        if legend:
                            # Sort column names by their corresponding label ID
                            sorted_labels = sorted(legend.items(), key=lambda item: int(item[0]))
                            prob_df.columns = [name for _, name in sorted_labels]
                        
                        st.dataframe(prob_df, use_container_width=True)

                current_file_performance = prediction_results.get('processing_summary', {}).get('current_file_performance')
                if current_file_performance:
                    st.subheader("Performance on Current Recording (if annotations available)")
                    st.info(current_file_performance.get("note", "No note provided."))
                    display_model_performance({"detailed_metrics": {"current_file": current_file_performance}})


            except requests.exceptions.RequestException as e:
                st.error(f"Error during prediction: {e}")
                if e.response:
                    st.error(f"Server response: {e.response.text}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
