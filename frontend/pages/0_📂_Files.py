""" Streamlit page for managing EDF files from a FastAPI backend."""
import streamlit as st
import requests

FASTAPI_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="EDF File Selector 📂",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="📂",
)

st.title("EDF File Selector")


def get_available_files_from_api():
    """Fetches available EDF files from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/files/available")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching available files: {e}")
        return None


def select_files_via_api(file_ids: list):
    """Sends selected file IDs to the FastAPI backend."""
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/files/select",
            json={"file_ids": file_ids}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error selecting files: {e}")
        return None


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


def deselect_files_via_api(file_ids: list):
    """Sends file IDs to deselect to the FastAPI backend."""
    try:
        response = requests.delete(
            f"{FASTAPI_BASE_URL}/files/deselect",
            json={"file_ids": file_ids}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error deselecting files: {e}")
        return None


# Section 1: Display available EDF files

st.markdown("""
Below you can see the available files you can select from for processing.
""")

available_files = get_available_files_from_api()

st.header("Available EDF Files on Server")
if available_files:
    st.subheader("Cassette Files:")
    if available_files["cassette_files"]:
        for file_id, filename in available_files["cassette_files"].items():
            st.write(f"**ID {file_id}:** {filename}")
    else:
        st.info("No cassette files found.")

    st.subheader("Telemetry Files:")
    if available_files["telemetry_files"]:
        for file_id, filename in available_files["telemetry_files"].items():
            st.write(f"**ID {file_id}:** {filename}")
    else:
        st.info("No telemetry files found.")
else:
    st.warning("Could not retrieve available files. Check FastAPI server.")


st.divider()

# Section 2: Select files

st.header("Select Files for Processing")

st.markdown("""
Select the files you want to process by choosinge their IDs.
""")

cassete_ids = available_files.get("cassette_files", {}).keys() if available_files else []
telemetry_ids = available_files.get("telemetry_files", {}).keys() if available_files else []


selected_ids = st.multiselect(
    "Select File IDs to Process",
    options=list(cassete_ids) + list(telemetry_ids),
)

# print(f"Selected IDs: {selected_ids}")

if st.button("Select Files"):
    if selected_ids:
        int_selected_ids = [int(id) for id in selected_ids]
        # print(f"Integer Selected IDs: {int_selected_ids}")
        response = select_files_via_api(int_selected_ids)
        if response:
            st.success("Files selected successfully.")
        else:
            st.error("Failed to select files")
    else:
        st.warning("Please select at least one file ID.")

# Section 3: Display currently selected files

st.divider()
st.header("Currently Selected Files")

st.markdown("""
To doublecheck your selection, you can use the "View Selected Files" button.
""")

selected_files = get_selected_files_from_api()

if st.button("View Selected Files"):
    if selected_files:
        for file_id, filename in selected_files.get("selected_files", {}).items():
            st.write(f"**ID {file_id}:** {filename}")
        st.write(f"Total Selected Files: {selected_files.get('total_selected', 0)}")
    else:
        st.warning("No files currently selected or could not retrieve selected files.")


# Section 4: Deselect files

st.divider()

st.header("Deselect Files")

st.markdown(""" If you want to remove files from your selection, you can do so by choosing their IDs below.""")

if selected_files:
    selected_file_ids = list(selected_files["selected_files"].keys())
    deselect_ids = st.multiselect(
        "Select File IDs to Deselect",
        options=selected_file_ids,
    )
    if st.button("Deselect Files"):
        if deselect_ids:
            int_deselect_ids = [int(id) for id in deselect_ids]
            response = deselect_files_via_api(int_deselect_ids)
            if response:
                st.success("Deselection complete")
            else:
                st.error("There was an error deselecting files.")
        else:
            st.warning("Please select at least one file ID to proceed.")
else:
    st.warning("No files are currently selected ")
