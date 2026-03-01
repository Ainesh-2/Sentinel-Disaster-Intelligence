from models.yolo_segmentation import YOLOSegmentation
from heatmap.heatmap_gen import generate_heatmap_overlay

image_path = "xbd-tiles-256/val/images/000_000_hurricane-florence_00000407_post_disaster.tif"

model = YOLOSegmentation()

heatmap, damage_percent, _ = model.segment_image(image_path)

if heatmap is not None:
    print("Damage Percentage:", damage_percent)
    print("Max heat value:", heatmap.max())
    generate_heatmap_overlay(image_path, heatmap)
else:
    print("No damage detected.")
