import cv2
import mediapipe as mp
import numpy as np
from skimage import color

# Function to apply blush color
def apply_blush_color(image, intensity=0.5, r=223., g=91., b=111.):
    height, width, _ = image.shape

    # Convert image to LAB color space
    val = color.rgb2lab((image / 255.)).reshape(width * height, 3)
    L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])

    # Target LAB color for blush
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )

    # Adjust LAB values based on intensity
    ll, aa, bb = (L1 - L) * intensity, (A1 - A) * intensity, (B1 - B) * intensity
    val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)
    val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)
    val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)

    # Convert back to RGB and scale to 0-255 range
    image_lab = color.lab2rgb(val.reshape(height, width, 3)) * 255
    return image_lab.astype(np.uint8)

# Function to smoothen blush effect with circular mask
def smoothen_blush(image, center_x, center_y, radius):
    height, width, _ = image.shape

    # Create circular mask
    imgBase = np.zeros((height, width))
    cv2.circle(imgBase, (center_x, center_y), radius, 1, -1)

    # Blur the mask
    imgMask = cv2.GaussianBlur(imgBase, (51, 51), 0)

    # Create 3-channel mask
    imgBlur3D = np.zeros_like(image, dtype='float')
    imgBlur3D[:, :, 0] = imgMask
    imgBlur3D[:, :, 1] = imgMask
    imgBlur3D[:, :, 2] = imgMask

    # Apply blush effect by blending original image with blurred image
    return (imgBlur3D * image + (1 - imgBlur3D) * image_org).astype(np.uint8)

# Function to apply blush to the left cheek
def apply_blush_to_left_cheek(image, face_landmarks, left_cheek_indices):
    # Extract landmarks for left cheek
    left_cheek_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for idx, landmark in enumerate(face_landmarks.landmark) if idx in left_cheek_indices]
    left_x, left_y = zip(*left_cheek_points)

    # Determine center and radius for left cheek blush (example values, adjust as needed)
    left_center_x = int(np.mean(left_x))
    left_center_y = int(np.mean(left_y))
    left_radius = int(min(max(np.mean(np.abs(np.diff(left_x))), np.mean(np.abs(np.diff(left_y)))), 20))

    # Apply blush color and smoothening to left cheek
    image_blushed = apply_blush_color(image.copy())
    image_blushed = smoothen_blush(image_blushed, left_center_x, left_center_y, left_radius)

    return image_blushed

# Function to apply blush to the right cheek
def apply_blush_to_right_cheek(image, face_landmarks, right_cheek_indices):
    # Extract landmarks for right cheek
    right_cheek_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for idx, landmark in enumerate(face_landmarks.landmark) if idx in right_cheek_indices]
    right_x, right_y = zip(*right_cheek_points)

    # Determine center and radius for right cheek blush (example values, adjust as needed)
    right_center_x = int(np.mean(right_x))
    right_center_y = int(np.mean(right_y))
    right_radius = int(min(max(np.mean(np.abs(np.diff(right_x))), np.mean(np.abs(np.diff(right_y)))), 20))

    # Apply blush color and smoothening to right cheek
    image_blushed = apply_blush_color(image.copy())
    image_blushed = smoothen_blush(image_blushed, right_center_x, right_center_y, right_radius)

    return image_blushed

# Load the image
image_path = 'imgs/Input.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (512, 512))  # Resize image to 512x512
image_org = image.copy()

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Convert image to RGB and process
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image_rgb)

# Define cheek landmarks (adjust indices based on Mediapipe's landmark map)
left_cheek_indices = [10, 152, 145, 163, 166, 172, 70]
right_cheek_indices = [284, 442, 435, 453, 456, 462]

# Process each detected face
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Apply blush to left cheek
        image_blushed_left = apply_blush_to_left_cheek(image.copy(), face_landmarks, left_cheek_indices)

        # Apply blush to right cheek
        image_blushed_right = apply_blush_to_right_cheek(image.copy(), face_landmarks, right_cheek_indices)

        blended_image = cv2.addWeighted(image_blushed_left, 0.5, image_blushed_right, 0.5, 0)

        # Display the blended image with both cheeks colored
        cv2.imshow('Blended Cheek Colored Image', blended_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()