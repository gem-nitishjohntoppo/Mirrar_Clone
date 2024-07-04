import cv2
import mediapipe as mp
import numpy as np
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Load the jewelry image with transparency (e.g., a PNG file)



def overlay_jewelry(frame,selected_image_necklace):
    jewelry_img = cv2.imread(selected_image_necklace, cv2.IMREAD_UNCHANGED)
    # Offset for positioning the jewelry below the chin
    offset_y = 30

    # Process the frame to find face landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame1 = frame.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = face_landmarks.landmark

            # Use landmarks 152 (lower chin) and 9 (neck area)
            x_chin, y_chin = int(landmark_points[152].x * frame.shape[1]), int(landmark_points[152].y * frame.shape[0])
            x_neck, y_neck = int(landmark_points[9].x * frame.shape[1]), int(landmark_points[9].y * frame.shape[0])
            y_chin += offset_y  # Add an offset to position the jewelry slightly below the chin

            # Calculate the distance between the landmarks
            distance = np.sqrt((x_neck - x_chin) ** 2 + (y_neck - y_chin) ** 2)

            # Resize the jewelry image based on the distance
            scale = distance / jewelry_img.shape[1]
            new_w = int(jewelry_img.shape[1] * scale)
            new_h = int(jewelry_img.shape[0] * scale)
            overlay = cv2.resize(jewelry_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate position for overlay
            x_center = x_chin
            y_center = y_chin + int(distance / 3)  # Position slightly above the chest
            h, w = frame.shape[:2]
            x_start = max(0, x_center - new_w // 2)
            y_start = max(0, y_center - new_h // 2)
            x_end = min(w, x_start + overlay.shape[1])
            y_end = min(h, y_start + overlay.shape[0])

            overlay_resized = overlay[:(y_end - y_start), :(x_end - x_start)]

            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Apply overlay to the frame
            for c in range(0, 3):
                frame1[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_resized[:, :, c] +
                                                           alpha_l * frame[y_start:y_end, x_start:x_end, c])
    frame1 = cv2.flip(frame1, 1)
    # main_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    return frame1
