import os
import cv2
import numpy as np

SOURCE_DIR = "xbd-tiles-256"
DEST_DIR = "data/disaster_dataset"

MAX_TRAIN_SAMPLES = 2000


def ensure_dirs():
    os.makedirs(f"{DEST_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{DEST_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{DEST_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{DEST_DIR}/labels/val", exist_ok=True)


def convert_split(split, max_samples=None):
    image_dir = f"{SOURCE_DIR}/{split}/images"
    mask_dir = f"{SOURCE_DIR}/{split}/masks"

    dest_image_dir = f"{DEST_DIR}/images/{split}"
    dest_label_dir = f"{DEST_DIR}/labels/{split}"

    count = 0

    for file in os.listdir(image_dir):

        if not file.endswith(".tif"):
            continue

        image_path = os.path.join(image_dir, file)
        mask_filename = file.replace(".tif", ".png")
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        h, w = mask.shape

        binary_mask = np.where(
            (mask == 2) | (mask == 3) | (mask == 4) | (mask == 255),
            1,
            0
        ).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            continue

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        yolo_lines = []

        for contour in contours:
            if len(contour) < 3:
                continue

            contour = contour.squeeze()

            if len(contour.shape) != 2:
                continue

            normalized = []
            for x, y in contour:
                normalized.append(x / w)
                normalized.append(y / h)

            yolo_line = "0 " + " ".join(map(str, normalized))
            yolo_lines.append(yolo_line)

        if len(yolo_lines) == 0:
            continue

        image_name = file.replace(".tif", ".png")

        cv2.imwrite(os.path.join(dest_image_dir, image_name), image)

        label_path = os.path.join(
            dest_label_dir,
            image_name.replace(".png", ".txt")
        )

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        count += 1

        if max_samples and count >= max_samples:
            break

    print(f"{split} conversion complete. Samples:", count)


if __name__ == "__main__":
    ensure_dirs()
    convert_split("train", max_samples=MAX_TRAIN_SAMPLES)
    convert_split("val")
    print("Dataset conversion finished.")
