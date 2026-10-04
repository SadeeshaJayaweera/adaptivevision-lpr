import numpy as np
from typing import List
from ultralytics import YOLO

from backend.app.core.interfaces import IVehicleDetector, IPlateDetector
from backend.app.models.domain import BoundingBox

class YOLOVehicleDetector(IVehicleDetector):
    def __init__(self, model_path: str = "yolov8n.pt"):
        # We use standard YOLOv8 nano for baseline vehicle detection
        self.model = YOLO(model_path)
        # COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        # Ultralytics inference
        results = self.model(frame, classes=self.vehicle_classes, verbose=False)
        bboxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                bboxes.append(BoundingBox(
                    x_min=float(x1), y_min=float(y1), 
                    x_max=float(x2), y_max=float(y2), 
                    confidence=conf
                ))
        return bboxes

class YOLOPlateDetector(IPlateDetector):
    def __init__(self, model_path: str = "best_plate.pt"):
        # In a real environment, this would be a custom trained YOLO plate model
        try:
            self.model = YOLO(model_path)
        except Exception:
            # Fallback to a placeholder behavior if model is not trained yet
            self.model = None

    def detect(self, frame: np.ndarray, vehicle_bbox: BoundingBox) -> List[BoundingBox]:
        if not self.model:
            # Mock behavior: return a bounding box in the middle lower half of the vehicle
            w = vehicle_bbox.x_max - vehicle_bbox.x_min
            h = vehicle_bbox.y_max - vehicle_bbox.y_min
            return [BoundingBox(
                x_min=vehicle_bbox.x_min + w * 0.3,
                y_min=vehicle_bbox.y_min + h * 0.7,
                x_max=vehicle_bbox.x_min + w * 0.7,
                y_max=vehicle_bbox.y_min + h * 0.9,
                confidence=0.85
            )]
            
        # Real inference on the cropped vehicle
        x1, y1, x2, y2 = int(vehicle_bbox.x_min), int(vehicle_bbox.y_min), int(vehicle_bbox.x_max), int(vehicle_bbox.y_max)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
            
        results = self.model(crop, verbose=False)
        bboxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                # Translate back to global coordinates
                bboxes.append(BoundingBox(
                    x_min=float(px1 + x1), y_min=float(py1 + y1), 
                    x_max=float(px2 + x1), y_max=float(py2 + y1), 
                    confidence=conf
                ))
        return bboxes
