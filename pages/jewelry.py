import cv2
from Jewelry.bracelet_overlay import add_bracelet_overlay
from Jewelry.ring_overlay import overlay_ring_on_hand
from Jewelry.earring_overlay import overlay_earring
import streamlit as st
from camera_input_live import camera_input_live
from Jewelry.necklace_overlay import overlay_jewelry
from streamlit_image_select import image_select

# Define available images
necklace_images = ["Jewelry/assets/Necklace/1.png", "Jewelry/assets/Necklace/2.png", "Jewelry/assets/Necklace/3.png", "Jewelry/assets/Necklace/4.png"]
ring_images = ["Jewelry/assets/Rings/1.png", "Jewelry/assets/Rings/2.png", "Jewelry/assets/Rings/3.png", "Jewelry/assets/Rings/4.png"]
ear_rings_images = ["Jewelry/assets/Ear-Rings/1.png", "Jewelry/assets/Ear-Rings/2.png", "Jewelry/assets/Ear-Rings/3.png", "Jewelry/assets/Ear-Rings/4.png"]
bracelet_images = ["Jewelry/assets/Bracelets/1.png", "Jewelry/assets/Bracelets/2.png", "Jewelry/assets/Bracelets/3.png"]
alt_ear_images = ["Jewelry/assets/1.png","Jewelry/assets/2.png","Jewelry/assets/3.png","Jewelry/assets/4.png"]
# Create tabs for different types of jewelry
tab1, tab2, tab3, tab4 = st.tabs(["Necklace", "Ring", "Earring", "Bracelet"])

# Initialize variables
stop_requested = st.session_state.get('stop_requested', False)
frame_generator = st.session_state.get('frame_generator', None)
frame_placeholder = st.empty()  # Placeholder for the camera feed

# Function to handle camera feed and overlay
def handle_camera_feed(selected_image, overlay_function, key_prefix):
    global stop_requested, frame_generator
    stop_requested = st.session_state.get(f'{key_prefix}_stop_requested', False)
    frame_generator = st.session_state.get(f'{key_prefix}_frame_generator', None)
    frame_placeholder = st.empty()  # Placeholder for the camera feed

    # Start button to start the camera feed
    if st.button("TRY ON", key=f"start_{key_prefix}"):
        stop_requested = False
        st.session_state[f'{key_prefix}_stop_requested'] = stop_requested
        frame_generator = camera_input_live()
        st.session_state[f'{key_prefix}_frame_generator'] = frame_generator
        st.write("Starting camera feed...")

    # Stop button to stop the camera feed
    if st.button("STOP", key=f"stop_{key_prefix}"):
        if frame_generator:
            stop_requested = True
            st.session_state[f'{key_prefix}_stop_requested'] = stop_requested
            st.write("Camera feed stopped.")

            # Clean up: Stop the frame generator
            try:
                frame_generator.close()  # Close the generator to stop the camera feed
            except Exception as e:
                st.write(f"Error stopping camera feed: {e}")

            # Clear frame placeholder
            frame_placeholder.empty()

            # Rerun Streamlit app to prevent page reload
            st.experimental_rerun()

    # Continuously update the frame with the selected jewelry
    if frame_generator and not stop_requested:
        try:
            for frame in frame_generator:
                if frame is not None:
                     # Flip the frame horizontally to create a mirror effect
                    frame_with_jewelry = overlay_function(frame, selected_image)
                    frame_placeholder.image(frame_with_jewelry, channels="RGB", use_column_width=True)
                if stop_requested:
                    break
        except StopIteration:
            print("Camera feed stopped.")
    else:
        print("Failed to start camera feed.")

with tab1:
    st.header("Necklace")
    selected_image_necklace = image_select("Select a Necklace", necklace_images, key="selected_image_necklace")
    handle_camera_feed(selected_image_necklace, overlay_jewelry, "necklace")

with tab2:
    st.header("Ring")
    selected_image_ring = image_select("Select a Ring", ring_images, key="selected_image_ring")
    handle_camera_feed(selected_image_ring, overlay_ring_on_hand, "ring")

with tab3:
    st.header("Earring")
    # selected_image_ear_ring = image_select("Select an Earring", ear_rings_images, key="selected_image_earring")
    # handle_camera_feed(selected_image_ear_ring, overlay_earring, "earring")
    selected_image_ear_ring = image_select("Select an Earring", ear_rings_images, key="selected_image_earring")

    # Find the index of the selected image in the ear_rings_images list
    index = ear_rings_images.index(selected_image_ear_ring)

    # Use the corresponding image from alt_ear_images
    selected_image_ear_ring = alt_ear_images[index]

    # Proceed with handling the camera feed
    handle_camera_feed(selected_image_ear_ring, overlay_earring, "earring")

with tab4:
    st.header("Bracelet")
    selected_image_bracelet = image_select("Select a Bracelet", bracelet_images, key="selected_image_bracelet")
    handle_camera_feed(selected_image_bracelet, add_bracelet_overlay, "bracelet")
