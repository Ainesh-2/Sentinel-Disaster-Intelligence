import cv2
import numpy as np


def generate_heatmap_overlay(image_path, heatmap):
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image.")
        return

    # Normalize heatmap safely
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap

    heatmap_uint8 = np.uint8(255 * heatmap_norm)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Ensure correct types
    colored = colored.astype(np.uint8)
    image = image.astype(np.uint8)

    overlay = cv2.addWeighted(image, 0.7, colored, 0.3, 0)

    cv2.imshow("Damage Heatmap Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
