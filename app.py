"""Entry point for the Wavelet Lab Streamlit application."""

import streamlit as st

st.set_page_config(
    page_title="Wavelet Lab | Medical Image Enhancement",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard import run_dashboard


run_dashboard()
