from ultralytics import YOLO
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
        if results[0].masks is None:
            print("No masks found in the results.")
            return None, 0.0

        masks = results[0].masks.data.cpu().numpy()

        combined_mask = np.sum(masks, axis=0)
        combined_mask = (combined_mask > 0).astype(np.uint8)

        damage_percentage = (combined_mask.sum() / combined_mask.size) * 100

        return combined_mask, round(damage_percentage, 2), results[0].orig_shape
