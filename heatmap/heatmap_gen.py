import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_confidence_overlay(image_path, confidence_heatmap):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    heatmap = confidence_heatmap.copy()
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_norm = cv2.normalize(
        heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_norm[heatmap_norm < 15] = 0
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title('Damage Heatmap Overlay')
    plt.axis('off')
    plt.show()
