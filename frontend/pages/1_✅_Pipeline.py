"""Streamlit page for the actual pipeline/pedictions"""

import streamlit as st
import requests
import pandas as pd
import os

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/pipeline/predict-edf"
PERFORMANCE_ENDPOINT = f"{FASTAPI_BASE_URL}/pipeline/all-performance"
AVAILABLE_CONFIGS = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]
PREDICT_SELECTED_ENDPOINT = f"{FASTAPI_BASE_URL}/pipeline/predict-selected-file"


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

if st.button("Run Pipeline"):
    if not selected_files:
        st.warning("No files selected to run the pipeline. Please select files first.")
    else:
        st.session_state.running_pipeline = {}
        for file_id, file_name in selected_files.items():
            st.subheader(f"Processing: {file_name}")
            with st.spinner(f"Running pipeline for {file_name}..."):
                try:
                    response = requests.post(
                        f"{PREDICT_SELECTED_ENDPOINT}/{file_id}",
                        params={"config": config},
                    )
                    response.raise_for_status()
                    prediction_results = response.json()
                    st.session_state.running_pipeline[file_id] = prediction_results

                    st.success(f"Pipeline finished for {file_name}!")
                    # print_pipeline_summary(prediction_results)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error running pipeline for {file_name}: {e}")
                    if e.response:
                        st.error(f"Server response: {e.response.text}")


if st.session_state.get("running_pipeline"):
    st.subheader(" Pipeline Results")
    st.divider()
    for file_id, results in st.session_state.running_pipeline.items():
        st.markdown(f"For File ID: {file_id}")
        print_pipeline_summary(results)

st.header("Model Performance Analysis")
st.markdown(
    "View the detailed performance metrics for the selected model configuration."
)

if st.button("Get Model Performance"):
    with st.spinner(f"Loading performance metrics for {config} configuration"):
        try:
            response = requests.get(f"{PERFORMANCE_ENDPOINT}/{config}")
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
                st.dataframe(summary_df)

            st.subheader("Overfitting Analysis")
            if performance_data.get("overfitting_analysis"):
                accuracy = performance_data["overfitting_analysis"].get("accuracy_drop")
                if accuracy:
                    st.write(f"**Accuracy drop:** {accuracy:.4f}")
                f1_drop = performance_data["overfitting_analysis"].get("f1_drop")
                if f1_drop:
                    st.write(f"**F1 Score Drop:** {f1_drop:.4f}")
                severity = performance_data["overfitting_analysis"].get(
                    "overfitting_severity"
                )
                if severity:
                    st.write(f"**Overfitting Severity:** {severity}")
                quality = performance_data["overfitting_analysis"].get(
                    "generalization_quality"
                )
                if quality:
                    st.write(f"**Generalization Quality:** {quality}")
                vs_random = performance_data["overfitting_analysis"].get(
                    "vs_random_guessing"
                )
                if vs_random:
                    st.write("**Overfitting vs Random Performance:**")
                    st.json(vs_random)

            st.header("Detailed Metrics")
            if performance_data.get("detailed_metrics"):
                for dataset_type, metrics in performance_data[
                    "detailed_metrics"
                ].items():
                    if metrics:
                        st.subheader(f"{dataset_type.capitalize()} Set Metrics")
                        st.write("**Overall Metrics:**")
                        st.json(metrics["overall_metrics"])

                        if metrics.get("per_class_metrics"):
                            st.write("**Per-Class Metrics:**")
                            per_class_df = pd.DataFrame(metrics["per_class_metrics"]).T
                            st.dataframe(per_class_df)

                        if metrics.get("confusion_matrix"):
                            st.write("**Confusion Matrix:**")
                            cm_data = metrics["confusion_matrix"]["matrix"]
                            cm_labels = metrics["confusion_matrix"]["labels"]
                            cm_df = pd.DataFrame(
                                cm_data, index=cm_labels, columns=cm_labels
                            )
                            st.dataframe(cm_df)
                    else:
                        st.info(
                            f"No detailed metrics available for the {dataset_type} set."
                        )

            st.subheader("Recommendations")
            if performance_data.get("recommendations"):
                st.json(performance_data["recommendations"])

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching performance metrics: {e}")
            if e.response:
                st.error(f"Server response: {e.response.text}")
