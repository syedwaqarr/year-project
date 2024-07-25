import streamlit as st
from home import app as home_app
from about import app as about_app
from predict import app as predict_app
from streamlit_option_menu import option_menu

# Create a dictionary to hold the pages
pages = {
    "Home": home_app,
    "About": about_app,
    "Predict": predict_app
}

# Create sidebar navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "About", "Predict"],
    icons=["house-fill", "info-circle-fill", "cloud-arrow-up-fill"], 
    default_index=0,
    orientation="horizontal",
)

# Run the selected page
pages[selected]()
