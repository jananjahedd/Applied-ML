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

st.sidebar.success("Select a page above.")