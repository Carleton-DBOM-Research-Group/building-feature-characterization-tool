import pandas as pd
import streamlit as st


@st.cache_data
def load_weather():
    try:
        weather = pd.read_csv(".\Data/ON_OTTAWA-INTL-ONT_716280_19EPW.csv", skiprows=18, encoding='latin1')
    except FileNotFoundError:
        st.error('Weather data file not found. Please ensure the file exists in the correct directory.')
        return None
    return weather