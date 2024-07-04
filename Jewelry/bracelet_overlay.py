import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define a reference size for the bracelet (you can adjust this based on your application)
reference_size = 100  # Example size in pixels
scaling_factor = 1.2  # Scale bracelet size by 20%

def add_bracelet_overlay(frame, image):
    frame1 = frame.copy()
    bracelet_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process using Mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Overlay bracelet image near the wrist area (landmark 0)
            if len(hand_landmarks.landmark) >= 2:  # Ensure there are enough landmarks
                # Get the coordinates of landmark 0 (wrist coordinates)
                x1 = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y1 = int(hand_landmarks.landmark[0].y * frame.shape[0])

                # Calculate size of the bracelet image based on a reference size
                size_factor = (reference_size / bracelet_img.shape[1]) * scaling_factor

                # Resize bracelet image
                bracelet_resized = cv2.resize(bracelet_img, None, fx=size_factor, fy=size_factor)

                # Adjust offsets to move the bracelet closer to landmark 0
                x_offset = x1 - bracelet_resized.shape[1] // 2
                y_offset = y1 - bracelet_resized.shape[0] // 2 + 30  # Move it 30 pixels below the landmark 0

                # Ensure overlay and bracelet_resized have the same shape
                if 0 <= y_offset < frame.shape[0] - bracelet_resized.shape[0] and 0 <= x_offset < frame.shape[1] - bracelet_resized.shape[1]:
                    overlay = frame[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]]
                    if overlay.shape[0] == bracelet_resized.shape[0] and overlay.shape[1] == bracelet_resized.shape[1]:
                        # Overlay the resized bracelet image onto the frame
                        alpha_s = bracelet_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            overlay[:, :, c] = (alpha_s * bracelet_resized[:, :, c] +
                                                alpha_l * overlay[:, :, c])

                        # Update the frame with the overlay
                        frame1[y_offset:y_offset + bracelet_resized.shape[0], x_offset:x_offset + bracelet_resized.shape[1]] = overlay
    frame1 = cv2.flip(frame1, 1)
    return cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
