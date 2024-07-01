import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Load the jewelry image with transparency (e.g., a PNG file)
jewelry_image = cv2.imread('assets/jewellery/3.png', cv2.IMREAD_UNCHANGED)

# Function to overlay jewelry image on the face
def overlay_jewelry(frame, chin_landmark, neck_landmark, jewelry_img, offset_y=30):
    x_chin, y_chin = int(chin_landmark.x * frame.shape[1]), int(chin_landmark.y * frame.shape[0])
    x_neck, y_neck = int(neck_landmark.x * frame.shape[1]), int(neck_landmark.y * frame.shape[0])
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

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = face_landmarks.landmark

            # Use landmarks 152 (lower chin) and 9 (neck area)
            jewelry_frame = overlay_jewelry(frame, landmark_points[152], landmark_points[9], jewelry_image)

        cv2.imshow('Jewelry Try-On', jewelry_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
