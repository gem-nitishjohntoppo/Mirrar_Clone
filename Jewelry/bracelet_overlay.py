import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the bracelet image (replace with your own bracelet image path)
bracelet_img = cv2.imread('assets/SDC14597.png', cv2.IMREAD_UNCHANGED)

# Define a reference size for the bracelet (you can adjust this based on your application)
reference_size = 100  # Example size in pixels
scaling_factor = 1.2  # Scale bracelet size by 20%

def add_bracelet_overlay(frame):
    frame1 = frame.copy()

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process using Mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Overlay bracelet image between landmarks 0 and 1 (wrist coordinates)
            if len(hand_landmarks.landmark) >= 2:  # Ensure there are enough landmarks
                # Get the coordinates of landmarks 0 and 1
                x1 = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y1 = int(hand_landmarks.landmark[0].y * frame.shape[0])
                x2 = int(hand_landmarks.landmark[1].x * frame.shape[1])
                y2 = int(hand_landmarks.landmark[1].y * frame.shape[0])

                # Calculate midpoint between landmarks 0 and 1
                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2

                # Calculate size of the bracelet image based on a reference size
                size_factor = (reference_size / bracelet_img.shape[1]) * scaling_factor

                # Resize bracelet image
                bracelet_resized = cv2.resize(bracelet_img, None, fx=size_factor, fy=size_factor)

                # Calculate position to place the bracelet image centered on the midpoint
                x_offset = x_mid - bracelet_resized.shape[1] // 2
                y_offset = y1  # Adjust y_offset to stick the top corners to y1

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
                        frame1[y_offset:y_offset + bracelet_resized.shape[0],
                               x_offset:x_offset + bracelet_resized.shape[1]] = overlay

    return frame1
