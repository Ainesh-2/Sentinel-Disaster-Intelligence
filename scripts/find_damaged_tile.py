import os
import cv2
import numpy as np

mask_dir = "xbd-tiles-256/val/masks"
image_dir = "xbd-tiles-256/val/images"

for mask_file in os.listdir(mask_dir):

    image_file = mask_file.replace(".png", ".tif")

    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)

    # Skip if image does NOT exist
    if not os.path.exists(image_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    damage_pixels = np.sum((mask == 2) | (mask == 3) |
                           (mask == 4) | (mask == 255))

    if damage_pixels > 3000:  # strong visible damage
        print("Strong damaged tile found:")
        print("Image:", image_file)
        print("Damage pixels:", damage_pixels)
        break
