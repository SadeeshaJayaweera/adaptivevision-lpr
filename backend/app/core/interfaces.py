from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np

from backend.app.models.domain import (
    CameraHealth, 
    CompoundCondition, 
    BoundingBox,
    OCRResult,
    CharacterEvidence,
    RecoverabilityClass,
    PlateObservation
)

class ICameraSource(ABC):
    @abstractmethod
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        pass

class ICameraHealthAnalyzer(ABC):
    @abstractmethod
    def analyze(self, camera_id: str, frame: np.ndarray, telemetry: Dict[str, Any]) -> CameraHealth:
        pass

class IConditionAnalyzer(ABC):
    @abstractmethod
    def analyze_scene(self, frame: np.ndarray) -> CompoundCondition:
        pass
        
    @abstractmethod
    def get_quality_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        pass

class IVehicleDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        pass

class IPlateDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, vehicle_bbox: BoundingBox) -> List[BoundingBox]:
        pass

class IVisibilityAnalyzer(ABC):
    @abstractmethod
    def analyze(self, frame: np.ndarray, plate_bbox: BoundingBox) -> Dict[str, Any]:
        """Returns visibility, occlusion, occlusion_type, visible_character_regions"""
        pass

class IRecoverabilityEstimator(ABC):
    @abstractmethod
    def estimate(self, visibility_data: Dict[str, Any], condition: CompoundCondition) -> Tuple[RecoverabilityClass, float]:
        """Returns the recoverability class and a confidence score 0.0-1.0"""
        pass

class IAdaptiveRouter(ABC):
    @abstractmethod
    def determine_pipeline(self, condition: CompoundCondition, visibility_data: Dict[str, Any]) -> List[str]:
        pass
        
    @abstractmethod
    def execute_pipeline(self, frame: np.ndarray, pipeline: List[str]) -> Tuple[np.ndarray, List[str]]:
        pass

class IEnhancementModule(ABC):
    @abstractmethod
    def name(self) -> str:
        pass
        
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        pass

class IOCREngine(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        pass

class IFusionEngine(ABC):
    @abstractmethod
    def fuse_characters(self, character_evidence: List[CharacterEvidence]) -> str:
        pass
        
    @abstractmethod
    def fuse_temporal(self, observations: List[PlateObservation]) -> Tuple[str, float]:
        """Returns final fused plate string and agreement confidence"""
        pass

class IUncertaintyEstimator(ABC):
    @abstractmethod
    def estimate_uncertainty(self, observations: List[PlateObservation], fusion_confidence: float) -> Tuple[float, float]:
        """Returns (epistemic_uncertainty, aleatoric_uncertainty)"""
        pass

class IValidationEngine(ABC):
    @abstractmethod
    def validate_format(self, plate_text: str) -> Tuple[bool, str, str]:
        """Returns (is_valid, normalized_plate, failure_reason)"""
        pass
