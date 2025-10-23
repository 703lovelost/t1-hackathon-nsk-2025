from time import time

import torch
from ultralytics import YOLO


class SegmentModel:
    def __init__(self, model):
        self.model = model
    
    def process_frame(self, frame):
        raise NotImplementedError


class YoloModel(SegmentModel):
    def __init__(self, weight_name: str = "yolo11s-seg.pt"):
        self.model = YOLO(weight_name)
        print("Initialized YoloModel", flush=True)
    
    def process_frame(self, frame):
        start = time()
        result = self.model.predict(
            source=frame,
            imgsz=(frame.shape[0], frame.shape[1]),
            verbose=False,
            save=False,
            classes=[0],
        )[0]
        res = result.masks.data * 255
        if res.shape[0] != 1:
            res = (res > 0).any(dim=0, keepdim=True).to(torch.uint8)

        return res, time() - start
