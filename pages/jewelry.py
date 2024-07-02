import streamlit as st
from camera_input_live import camera_input_live
from Jewelry.bracelet_overlay import add_bracelet_overlay
from Jewelry.necklace_overlay import overlay_jewelry

st.title("Jewelry")
import streamlit as st

tab1, tab2, tab3 , tab4 = st.tabs(["Necklace","Ring","Ear_Ring","Bracelet"])

with tab1:
   st.header("Necklace")
   start_button = st.button("Start")

   if start_button:
       st.write("Starting camera feed...")
       frame_generator = camera_input_live()
       stop_button = st.button("Stop")
       if stop_button:
           stop_requested = True  # Set flag to stop the camera feed

       if frame_generator is not None:
           frame_placeholder = st.empty()  # Placeholder for updating frames

           try:
               for frame in frame_generator:
                   if frame is not None:
                       # Add bracelet overlay to the frame
                       frame_with_bracelet = overlay_jewelry(frame)
                       frame_placeholder.image(frame_with_bracelet, channels="RGB")
           except StopIteration:
               st.write("Camera feed stopped.")
       else:
           st.write("Failed to start camera feed.")




