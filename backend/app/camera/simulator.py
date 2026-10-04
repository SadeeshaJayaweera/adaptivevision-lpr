import cv2
import numpy as np
import time
import random
from typing import Tuple
from backend.app.core.interfaces import ICameraSource

class SimulatedCameraSource(ICameraSource):
    """
    Simulates a camera source with optional network degradation, 
    frame drops, and simulated latency to test Disaster Mode capabilities.
    """
    def __init__(self, video_path: str, simulate_drops: bool = False, drop_rate: float = 0.1, simulate_latency: bool = False):
        self.video_path = video_path
        self.simulate_drops = simulate_drops
        self.drop_rate = drop_rate
        self.simulate_latency = simulate_latency
        self.cap = cv2.VideoCapture(video_path)
        self.is_open = self.cap.isOpened()
        
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        if not self.is_open:
            return False, np.array([])
            
        if self.simulate_latency:
            # Simulate RTSP network jitter
            time.sleep(random.uniform(0.01, 0.05))
            
        if self.simulate_drops and random.random() < self.drop_rate:
            # Simulate a dropped frame by reading but returning False (or just returning empty)
            self.cap.read() 
            return False, np.array([])

        ret, frame = self.cap.read()
        
        # Loop video for continuous simulation
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        return ret, frame

    def close(self):
        if self.cap:
            self.cap.release()
            self.is_open = False
