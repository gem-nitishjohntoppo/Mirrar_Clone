import cv2
import mediapipe as mp


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the ring image (replace with your own ring image path)
ring_img = cv2.imread('assets/PngItem_3503026.png', cv2.IMREAD_UNCHANGED)

# Define a reference size for the ring (you can adjust this based on your application)
reference_size = 50  # Example size in pixels

def overlay_ring_on_hand(frame):
    frame1 = frame.copy()

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process using Mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Overlay ring image between landmarks 13 and 14 (assuming landmark indices)
            if len(hand_landmarks.landmark) >= 14:  # Ensure there are enough landmarks
                # Get the coordinates of landmarks 13 and 14
                x1 = int(hand_landmarks.landmark[13].x * frame.shape[1])
                y1 = int(hand_landmarks.landmark[13].y * frame.shape[0])
                x2 = int(hand_landmarks.landmark[14].x * frame.shape[1])
                y2 = int(hand_landmarks.landmark[14].y * frame.shape[0])

                # Calculate midpoint between landmarks 13 and 14
                x_mid = (x1 + x2) // 2
                y_mid = (y1 + y2) // 2

                # Calculate size of the ring image based on a reference size
                size_factor = reference_size / ring_img.shape[1]  # Assuming ring_img is square

                # Resize ring image
                ring_resized = cv2.resize(ring_img, None, fx=size_factor, fy=size_factor)

                # Calculate position to place the ring image centered on the midpoint
                x_offset = x_mid - ring_resized.shape[1] // 2
                y_offset = y_mid - ring_resized.shape[0] // 2

                # Ensure overlay and ring_resized have the same shape
                overlay = frame1[y_offset:y_offset + ring_resized.shape[0], x_offset:x_offset + ring_resized.shape[1]]
                if overlay.shape[0] == ring_resized.shape[0] and overlay.shape[1] == ring_resized.shape[1]:
                    # Overlay the resized ring image onto the frame
                    alpha_s = ring_resized[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        overlay[:, :, c] = (alpha_s * ring_resized[:, :, c] +
                                            alpha_l * overlay[:, :, c])

                    # Update the frame with the overlay
                    frame1[y_offset:y_offset + ring_resized.shape[0],
                    x_offset:x_offset + ring_resized.shape[1]] = overlay

        # Draw hand landmarks on the frame outside the overlay logic
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame1, (x, y), 5, (255, 0, 0), -1)

    return frame1