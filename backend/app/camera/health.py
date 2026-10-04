import time
from typing import Dict, Any, Deque
from collections import deque
import numpy as np

from backend.app.core.interfaces import ICameraHealthAnalyzer
from backend.app.models.domain import CameraHealth

class CameraHealthMonitor(ICameraHealthAnalyzer):
    def __init__(self, history_size: int = 30):
        self.frame_timestamps: Deque[float] = deque(maxlen=history_size)
        self.dropped_frames_count = 0
        self.total_frames_count = 0
        
    def log_frame(self, success: bool):
        self.total_frames_count += 1
        if success:
            self.frame_timestamps.append(time.time())
        else:
            self.dropped_frames_count += 1

    def calculate_fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if time_diff <= 0:
            return 0.0
        return len(self.frame_timestamps) / time_diff

    def analyze(self, camera_id: str, frame: np.ndarray, telemetry: Dict[str, Any] = None) -> CameraHealth:
        fps = self.calculate_fps()
        
        drop_rate = 0.0
        if self.total_frames_count > 0:
            drop_rate = self.dropped_frames_count / self.total_frames_count
            
        # Stub logic for lens obstruction based on simple heuristics (e.g., constant static regions)
        lens_obstruction = telemetry.get("lens_obstruction", 0.0) if telemetry else 0.0
        network_quality = telemetry.get("network_quality", 1.0 - drop_rate) if telemetry else (1.0 - drop_rate)
        
        # Calculate overall health score (0.0 to 1.0)
        health_score = max(0.0, 1.0 - (drop_rate * 1.5) - (lens_obstruction * 0.5))
        
        return CameraHealth(
            camera_id=camera_id,
            health=health_score,
            fps=fps,
            frame_drop_rate=drop_rate,
            lens_obstruction=lens_obstruction,
            network_quality=network_quality
        )
