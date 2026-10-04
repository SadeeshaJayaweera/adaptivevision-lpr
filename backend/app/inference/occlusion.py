import cv2
import numpy as np
from typing import Dict, Any, Tuple

from backend.app.core.interfaces import IVisibilityAnalyzer, IRecoverabilityEstimator
from backend.app.models.domain import BoundingBox, CompoundCondition, RecoverabilityClass

class PlateOcclusionAnalyzer(IVisibilityAnalyzer):
    def __init__(self):
        pass

    def analyze(self, frame: np.ndarray, plate_bbox: BoundingBox) -> Dict[str, Any]:
        """
        Estimates what percentage of the plate is visible vs occluded.
        In a full implementation, this uses a semantic segmentation model.
        Here we use heuristic edge density and contrast maps as a proxy.
        """
        x1, y1 = int(plate_bbox.x_min), int(plate_bbox.y_min)
        x2, y2 = int(plate_bbox.x_max), int(plate_bbox.y_max)
        
        # Ensure bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "visibility": 0.0,
                "occlusion": 1.0,
                "occlusion_type": "OUT_OF_FRAME",
                "visible_character_regions": 0.0
            }

        # Grayscale and edge detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density. A clean plate has high edge density around characters.
        # Massive occlusions (like mud or shadows) typically destroy edge features.
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize edge density to a theoretical max of 0.15 for a plate
        expected_density = 0.15
        visibility = min(1.0, edge_density / expected_density)
        
        # Very rough heuristic: if visibility is very low, assume dirt/mud occlusion
        occl_type = "NONE" if visibility > 0.8 else "PHYSICAL_OBSTRUCTION"
        
        return {
            "visibility": float(visibility),
            "occlusion": float(1.0 - visibility),
            "occlusion_type": occl_type,
            "visible_character_regions": float(visibility * 0.9) # roughly approx
        }


class EvidenceRecoverabilityEstimator(IRecoverabilityEstimator):
    def __init__(self):
        pass

    def estimate(self, visibility_data: Dict[str, Any], condition: CompoundCondition) -> Tuple[RecoverabilityClass, float]:
        visibility = visibility_data.get("visibility", 1.0)
        occlusion = visibility_data.get("occlusion", 0.0)
        
        # If the plate is completely covered, it's non-recoverable
        if occlusion > 0.8:
            return RecoverabilityClass.NON_RECOVERABLE, 0.95
            
        # If it's mostly visible, it's fully recoverable
        if visibility > 0.85:
            return RecoverabilityClass.FULLY_RECOVERABLE, 0.90
            
        # If it's partially occluded, check if environmental conditions make it worse
        if 0.4 < occlusion <= 0.8:
            if condition.weather.name in ["HEAVY_RAIN", "FOG", "SMOKE"]:
                # Compound issue: occluded + bad weather = low info
                return RecoverabilityClass.LOW_INFORMATION, 0.75
            else:
                return RecoverabilityClass.PARTIALLY_RECOVERABLE, 0.80
                
        return RecoverabilityClass.PARTIALLY_RECOVERABLE, 0.50
