import os
import sys
import streamlit as st
from pathlib import Path
import Image_detection
import video_detection
from streamlit_option_menu import option_menu
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'



st.set_page_config(
    page_title="YOLO",
    layout="wide"
)


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run(self):
        # st.sidebar.image('logo_.jpg', width=220)
        with st.sidebar:
            app = option_menu(
                menu_title="page_names",
                options=['IMAGE', 'VIDEO', 'URL LINKS'],
                icons=['cloud-upload-fill', 'compass-fill', 'binoculars-fill'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important", "background-color": '#ff6666'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "black"},
                }
            )
        for app_info in self.apps:
            if app == app_info["title"]:
                app_info["function"]()
                break

app = MultiApp()
app.add_app("IMAGE Detection", Image_detection.app())
app.add_app("VIDEO Detection", video_detection.app())
