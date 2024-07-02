import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Load the earring image with transparency (e.g., a PNG file)
earring_image = cv2.imread('Lovepik_com-401703983-earring.png', cv2.IMREAD_UNCHANGED)


# Function to overlay earring image on the face
def overlay_earring(frame):
    # Process the frame to find face landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define ear landmark (e.g., landmark 234)
            ear_landmark = 234

            # Define earring image
            earring_img = cv2.imread('Lovepik_com-401703983-earring.png', cv2.IMREAD_UNCHANGED)

            # Define offsets
            offset_x = 0
            offset_y = 0

            # Calculate internal parameters
            x_ear, y_ear = int(face_landmarks.landmark[ear_landmark].x * frame.shape[1]), int(face_landmarks.landmark[ear_landmark].y * frame.shape[0])
            x_ear += offset_x  # Add an offset to fine-tune the position horizontally
            y_ear += offset_y  # Add an offset to fine-tune the position vertically

            # Calculate the distance to determine the size of the earring
            distance = frame.shape[1] * 0.15  # Adjust the size based on your image and preference

            # Resize the earring image based on the distance
            scale = distance / earring_img.shape[1]
            new_w = int(earring_img.shape[1] * scale)
            new_h = int(earring_img.shape[0] * scale)

            # Determine if the earring is for the left ear (flip horizontally)
            if face_landmarks.landmark[ear_landmark].x < 0.5:
                overlay = cv2.flip(earring_img, 1)  # Flip horizontally for left ear
            else:
                overlay = earring_img.copy()  # Right ear, use as is

            overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate position for overlay
            x_center = x_ear
            y_center = y_ear + int(distance * 0.5)  # Position slightly below the ear landmark
            h, w = frame.shape[:2]
            x_start = max(0, x_center - new_w // 2)
            y_start = max(0, y_center - new_h // 2)
            x_end = min(w, x_start + new_w)
            y_end = min(h, y_start + new_h)

            overlay_resized = overlay[:(y_end - y_start), :(x_end - x_start)]

            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Apply overlay to the frame
            for c in range(0, 3):
                frame[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
                                                          alpha_l * frame[y_start:y_end, x_start:x_end, c])

    return frame
