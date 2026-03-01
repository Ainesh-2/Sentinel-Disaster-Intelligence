from ultralytics import YOLO
import numpy as np
import cv2


class YOLOSegmentation:
    def __init__(self):
        print("Using trained disaster model...")
        self.model = YOLO("runs/segment/train/weights/best.pt")

    def segment_image(self, image_path, conf_threshold=0.03):
        results = self.model(image_path, conf=conf_threshold)

        result = results[0]

        if result.masks is None:
            return None, 0.0, None

        masks = result.masks.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        original = cv2.imread(image_path)
        if original is None:
            print("Image loading failed.")
            return None, 0.0, None

        h, w, _ = original.shape

        heatmap = np.zeros((h, w), dtype=np.float32)

        for mask, conf in zip(masks, confidences):
            resized_mask = cv2.resize(mask, (w, h))

            # Convert to binary mask
            binary_mask = (resized_mask > 0.5).astype(np.float32)

            # Ignore very tiny regions (noise)
            if np.sum(binary_mask) < 200:
                continue

            heatmap += binary_mask * float(conf)

        if heatmap.max() == 0:
            return None, 0.0, (h, w)

        damage_percent = (np.sum(heatmap > 0) / (h * w)) * 100

        return heatmap, damage_percent, (h, w)
