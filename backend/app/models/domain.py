from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

# --- Enums for Condition Analysis ---

class IlluminationCondition(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"
    OVEREXPOSED = "OVEREXPOSED"
    BACKLIT = "BACKLIT"
    GLARE = "GLARE"

class WeatherCondition(str, Enum):
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"
    FOG = "FOG"
    MIST = "MIST"
    DUST = "DUST"
    SMOKE = "SMOKE"

class MotionCondition(str, Enum):
    STATIC = "STATIC"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class CameraDegradation(str, Enum):
    NORMAL = "NORMAL"
    BLUR = "BLUR"
    WATER_DROPLETS = "WATER_DROPLETS"
    DIRT = "DIRT"
    SHAKE = "SHAKE"
    LOW_BITRATE = "LOW_BITRATE"
    FRAME_DROP = "FRAME_DROP"
    SENSOR_NOISE = "SENSOR_NOISE"

class PhysicalObstruction(str, Enum):
    NONE = "NONE"
    PARTIAL = "PARTIAL"
    SEVERE = "SEVERE"
    COMPLETE = "COMPLETE"

class RecoverabilityClass(str, Enum):
    FULLY_RECOVERABLE = "FULLY_RECOVERABLE"
    PARTIALLY_RECOVERABLE = "PARTIALLY_RECOVERABLE"
    TEMPORARILY_OCCLUDED = "TEMPORARILY_OCCLUDED"
    LOW_INFORMATION = "LOW_INFORMATION"
    NON_RECOVERABLE = "NON_RECOVERABLE"

class DecisionState(str, Enum):
    ACCEPT = "ACCEPT"
    REPROCESS = "REPROCESS"
    UNKNOWN = "UNKNOWN"

class FailureReason(str, Enum):
    NO_PLATE_DETECTED = "NO_PLATE_DETECTED"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    LOW_LIGHT = "LOW_LIGHT"
    GLARE = "GLARE"
    MOTION_BLUR = "MOTION_BLUR"
    DEFOCUS = "DEFOCUS"
    RAIN = "RAIN"
    FOG = "FOG"
    SMOKE = "SMOKE"
    DUST = "DUST"
    PHYSICAL_OCCLUSION = "PHYSICAL_OCCLUSION"
    PLATE_DAMAGED = "PLATE_DAMAGED"
    PLATE_OUT_OF_FRAME = "PLATE_OUT_OF_FRAME"
    MULTIPLE_PLATES = "MULTIPLE_PLATES"
    OCR_DISAGREEMENT = "OCR_DISAGREEMENT"
    TEMPORAL_DISAGREEMENT = "TEMPORAL_DISAGREEMENT"
    FORMAT_INVALID = "FORMAT_INVALID"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CAMERA_FAILURE = "CAMERA_FAILURE"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    UNKNOWN = "UNKNOWN"

# --- Core Data Models ---

class CameraHealth(BaseModel):
    camera_id: str
    health: float = Field(..., ge=0.0, le=1.0)
    fps: float
    frame_drop_rate: float
    lens_obstruction: float
    network_quality: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CompoundCondition(BaseModel):
    illumination: IlluminationCondition
    weather: WeatherCondition
    motion: MotionCondition
    optics: CameraDegradation
    occlusion: PhysicalObstruction
    camera_health: str = "NORMAL"
    compound_condition: bool = True

class Point(BaseModel):
    x: float
    y: float

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = Field(..., ge=0.0, le=1.0)

class CharacterEvidence(BaseModel):
    candidate: str
    probability: float
    source: str
    frame_id: str
    engine: str
    condition: str

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
    quality: Dict[str, float]
    conditions: CompoundCondition
    ocr_results: List[OCRResult]
    visibility: float
    occlusion: float
    occlusion_type: str
    recoverability: float
    visible_character_regions: float
    applied_enhancements: List[str] = []

class EventModel(BaseModel):
    event_id: str
    vehicle_track_id: str
    camera_ids: List[str]
    first_seen: datetime
    last_seen: datetime
    plate: Optional[str]
    status: DecisionState
    failure_reason: Optional[FailureReason] = None
    confidence: float
    recoverability: float
    frames_used: int
    conditions: List[CompoundCondition]
    pipeline: List[str]
    ocr_engines: List[str]
    evidence: Dict[str, Any]
    model_versions: Dict[str, str]
