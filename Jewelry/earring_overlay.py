import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to overlay earring image on the face
def overlay_earring(frame, images):
    # Process the frame to find face landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define ear landmarks (e.g., landmarks 234 for left ear and 454 for right ear)
            ear_landmarks = [234, 454]

            # Load the earring image with transparency (e.g., a PNG file)
            earring_img = cv2.imread(images, cv2.IMREAD_UNCHANGED)

            # Get z-coordinates to determine visibility
            z_ear_left = face_landmarks.landmark[234].z
            z_ear_right = face_landmarks.landmark[454].z

            for ear_landmark in ear_landmarks:
                # Skip overlay if the ear is likely hidden (based on z-coordinate comparison)
                if ear_landmark == 234 and z_ear_left > z_ear_right + 0.1:
                    continue
                if ear_landmark == 454 and z_ear_right > z_ear_left + 0.1:
                    continue

                # Calculate internal parameters
                x_ear, y_ear = int(face_landmarks.landmark[ear_landmark].x * frame.shape[1]), int(face_landmarks.landmark[ear_landmark].y * frame.shape[0])

                # Calculate the distance to determine the size of the earring
                distance = frame.shape[1] * 0.15  # Adjust the size based on your image and preference

                # Resize the earring image based on the distance
                scale = distance / earring_img.shape[1]
                new_w = int(earring_img.shape[1] * scale)
                new_h = int(earring_img.shape[0] * scale)

                # Determine if the earring is for the left ear (flip horizontally and adjust position)
                if ear_landmark == 234:  # Left ear
                    overlay = cv2.flip(earring_img, 1)  # Flip horizontally for left ear
                    x_ear -= 20  # Move left earring slightly to the left
                else:  # Right ear
                    overlay = earring_img.copy()  # Right ear, use as is
                    x_ear += 20  # Move right earring slightly to the right

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
    frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
