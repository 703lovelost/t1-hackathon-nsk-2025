from pathlib import Path

import numpy as np
import cv2
import torch
from torchvision.transforms import ToPILImage

from segment_models import YoloModel


BASE_PATH = Path("Human-Segmentation-Dataset")

def hconcat_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Unequal heights!")
    shap = img1.shape
    width = int((img1.shape[1] + img2.shape[1]) / 40)
    shap = (shap[0], width, shap[2])
    delim = np.zeros(shape=shap, dtype=np.uint8)
    delim[:, 0:3, :] = 255
    delim[:, (width - 3):width, :] = 255
    return cv2.hconcat([img1, delim, img2])

def get_frame_mask(num):
    mask_path = BASE_PATH / "Ground_Truth" / f"{num}.png"
    image_path = BASE_PATH / "Training_Images" / f"{num}.jpg"
    return cv2.imread(mask_path), cv2.imread(image_path)

def main():
    model = YoloModel("yolo11l-seg.pt")
    to_pil = ToPILImage()
    mask, image = get_frame_mask(9)
    disp = hconcat_images(mask, image)
    boxes, masks = model.predict_frame(image)
    data = masks.data * 255
    pred_mask = np.array(to_pil(data))
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    # disp = hconcat_images(pred_mask, disp)
    while 1:
        cv2.imshow("Window", disp)
        cv2.imshow("Window2", pred_mask)
        key = cv2.waitKey(20)
        if (key == ord("q")) or (key == 27):
            break

if __name__ == "__main__":
    main()
