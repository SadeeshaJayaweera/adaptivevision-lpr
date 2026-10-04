from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Condition(str, Enum):
    NORMAL = "NORMAL"
    LOW_LIGHT = "LOW_LIGHT"
    EXTREME_LOW_LIGHT = "EXTREME_LOW_LIGHT"
    GLARE = "GLARE"
    OVEREXPOSURE = "OVEREXPOSURE"
    MOTION_BLUR = "MOTION_BLUR"
    DEFOCUS_BLUR = "DEFOCUS_BLUR"
    RAIN = "RAIN"
    FOG = "FOG"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    COMPRESSION = "COMPRESSION"
    PERSPECTIVE_DISTORTION = "PERSPECTIVE_DISTORTION"
    MULTI_CONDITION = "MULTI_CONDITION"
    UNKNOWN = "UNKNOWN"

class QualityReport(BaseModel):
    brightness: float = Field(..., ge=0.0, le=1.0)
    contrast: float = Field(..., ge=0.0, le=1.0)
    blur_score: float = Field(..., ge=0.0, le=1.0)
    noise: float = Field(..., ge=0.0, le=1.0)
    glare: float = Field(..., ge=0.0, le=1.0)
    plate_resolution: str
    condition: Condition
    quality_score: float = Field(..., ge=0.0, le=1.0)

class Point(BaseModel):
    x: float
    y: float

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = Field(..., ge=0.0, le=1.0)

class OCRResult(BaseModel):
    engine: str
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    character_confidences: List[float] = []

class PlateObservation(BaseModel):
    frame_id: str
    timestamp: datetime
    track_id: str
    plate_bbox: BoundingBox
    vehicle_bbox: BoundingBox
    quality: QualityReport
    ocr_results: List[OCRResult]
    applied_enhancements: List[str] = []

class DecisionState(str, Enum):
    ACCEPT = "ACCEPT"
    REPROCESS = "REPROCESS"
    UNKNOWN = "UNKNOWN"

class FinalPrediction(BaseModel):
    track_id: str
    plate: str
    confidence: float
    frames_used: int
    ocr_agreement: float
    temporal_agreement: float
    format_valid: bool
    decision: DecisionState
    timestamp: datetime = Field(default_factory=datetime.utcnow)
