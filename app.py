import streamlit as st
from streamlit_navigation_bar import st_navbar
from pages import jewelry
from pages import beauty
from pages import glasses
from pages import watches

import streamlit as st
from streamlit_navigation_bar import st_navbar
from camera_input_live import camera_input_live
def load_page(page_name):
    if page_name == "Home":
        st.title("Home Page")
        st.write("Welcome to the Home Page!")
    elif page_name == "Jewelry":
        exec(open("pages/jewelry.py").read())
    elif page_name == "Beauty":
        exec(open("pages/beauty.py").read())
    elif page_name == "Glasses":
        exec(open("pages/glasses.py").read())
    elif page_name == "Watches":
        exec(open("pages/watches.py").read())

# Create navigation bar
page = st_navbar(["Home", "Jewelry", "Beauty", "Glasses", "Watches"])


# Load selected page
load_page(page)
