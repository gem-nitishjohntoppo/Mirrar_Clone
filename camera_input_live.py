import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes, RTCConfiguration
import av
import cv2

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
    # Define the RTC configuration with STUN server
    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}  # STUN server
            ]
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
