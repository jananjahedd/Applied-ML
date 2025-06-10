import streamlit as st

st.set_page_config(
    page_title="Welcome page",
    page_icon="👋",
    layout="centered",
)

st.title("Welcome to our Sleep Stage Prediction App! 👋")

st.markdown("""
This application allows you to choose data in EDF format and get predictions on your sleep stages using pre-trained SVM models.
""")

st.header("How to use this app")
st.markdown("""
1. **Select EDF Files**: Navigate to the "Files" page to manage EDF files.
2. **Run the Pipeline**: Go to the "Pipeline" page to run the sleep stage prediction pipeline on your selected files.
You can choose from various model configurations. And view the performance of the pretrained model when tested on the chosen file.
On the same page you can check the performance of the pretrained model on the chosen configuration.
""")

st.sidebar.success("Select a page above.")