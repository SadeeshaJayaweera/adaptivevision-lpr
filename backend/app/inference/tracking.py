from typing import List, Dict
import numpy as np

from backend.app.models.domain import BoundingBox

class ObjectTracker:
    """
    Abstract placeholder for vehicle and plate tracking.
    In production, this wraps ByteTrack or BoT-SORT.
    """
    def __init__(self):
        # Dictionary to store active tracks: track_id -> BoundingBox
        self.active_tracks: Dict[str, BoundingBox] = {}
        self.next_id = 1
        
    def update(self, frame: np.ndarray, detections: List[BoundingBox]) -> Dict[str, BoundingBox]:
        """
        Updates the tracks based on new detections.
        Mock implementation: assigns a new ID to every detection.
        """
        tracked_objects = {}
        for det in detections:
            track_id = f"trk_{self.next_id}"
            self.active_tracks[track_id] = det
            tracked_objects[track_id] = det
            self.next_id += 1
            
        return tracked_objects
