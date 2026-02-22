from ultralytics import YOLO
import cv2
import numpy as np
import torch


class YOLOSegmentation:
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def segment_image(self, image_path):
        results = self.model(image_path, verbose=False)
        result = results[0]

        print("Boxes detected:", len(result.boxes))
        print("Masks present:", result.masks is not None)

        h, w = result.orig_shape

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_heatmap = np.zeros((h, w), dtype=np.float32)

        if result.masks is None or len(result.boxes) == 0:
            return combined_mask, 0.0, confidence_heatmap, (h, w)

        masks = result.masks.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for mask, conf in zip(masks, confidences):
            binary_mask = (mask > 0).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (w, h))
            combined_mask += binary_mask
            confidence_heatmap += binary_mask * float(conf)

        combined_mask = (combined_mask > 0).astype(np.uint8)
        damage_percentage = (combined_mask.sum() / combined_mask.size * 100)
        return combined_mask, round(damage_percentage, 2), confidence_heatmap, (h, w)
