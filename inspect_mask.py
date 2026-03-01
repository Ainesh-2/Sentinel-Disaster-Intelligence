import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

mask_dir = "xbd-tiles-256/train/masks"

for file in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    unique_values = np.unique(mask)
    if len(unique_values) > 6:
        print("Found damage mask:", file)
        print("Unique pixel values in the mask:", unique_values)
        break
print("All images have been checked.")
