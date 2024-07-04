import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes, RTCConfiguration
import av
import cv2
#TOKEN

import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables from .env file
load_dotenv()

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)

token = client.tokens.create()
print(token.ice_servers)

#TOKEN
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.overlay_function = None
        self.selected_image = None

    def update_overlay(self, overlay_function, selected_image):
        self.overlay_function = overlay_function
        self.selected_image = selected_image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.overlay_function and self.selected_image:
            img = self.overlay_function(img, self.selected_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(img, format="rgb24")

def camera_input_live(key="default"):
    # Define the RTC configuration with Twilio ICE servers
    rtc_config = RTCConfiguration(
        {
            "iceServers": token.ice_servers
        }
    )

    webrtc_ctx = webrtc_streamer(
        key=key,
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True, controls=True, style={"width": "100%"},
        ),
    )
    return webrtc_ctx

def main():
    st.title("Live Camera Feed with Streamlit WebRTC")
    camera_input_live()

if __name__ == "__main__":
    main()
