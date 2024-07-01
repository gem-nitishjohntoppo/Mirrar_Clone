import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import gaussian

# Define the segmentation parts and their corresponding colors
table = {
    'hair': 17,
    'upper_lip': 11,
    'lower_lip': 12
}

colors = {
    'hair': [230, 50, 20],  # Reddish color for hair
    'upper_lip': [20, 70, 180],  # Blue color for upper lips
    'lower_lip': [20, 70, 180]   # Blue color for lower lips
}

# Convenience expression for automatically determining device
device = (
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, channel_axis=-1)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


# Load models
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

# Define the image path
image_path = 'imgs/6.jpg'
image = cv2.imread(image_path)

# Run inference on image
inputs = image_processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits  # Shape (batch_size, num_labels, ~height/4, ~width/4)

# Resize output to match input image dimensions
upsampled_logits = nn.functional.interpolate(logits,
                                             size=(image.shape[0], image.shape[1]),  # H x W
                                             mode='bilinear',
                                             align_corners=False)

# Get label masks
labels = upsampled_logits.argmax(dim=1)[0]

# Convert to numpy array
labels_viz = labels.cpu().numpy()

# Process each part for coloring
for part_name, part_id in table.items():
    color = colors[part_name]
    if part_name == 'hair':
        image = hair(image, labels_viz, part_id, color)
    elif part_name == 'upper_lip' or part_name == 'lower_lip':
        mask = labels_viz == part_id
        # Blend color with image using opacity
        alpha = 0.5
        blended_color = np.array(color) * alpha + (1 - alpha) * image[mask]
        image[mask] = np.clip(blended_color, 0, 255).astype(np.uint8)

# Convert back to PIL Image for display
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Show the original and modified images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image with Colored Parts")
plt.imshow(image_pil)
plt.axis('off')

plt.show()
