from typing import List, Tuple
from backend.app.core.interfaces import IUncertaintyEstimator
from backend.app.models.domain import PlateObservation

class EpistemicAleatoricUncertainty(IUncertaintyEstimator):
    def __init__(self):
        pass

    def estimate_uncertainty(self, observations: List[PlateObservation], fusion_confidence: float) -> Tuple[float, float]:
        """
        Estimates uncertainty.
        Aleatoric Uncertainty: Data uncertainty (noise, weather, occlusion).
        Epistemic Uncertainty: Model uncertainty (disagreement among OCR engines, temporal variance).
        """
        if not observations:
            return 1.0, 1.0 # Max uncertainty
            
        # 1. Aleatoric (Data) Uncertainty
        # Average visibility and quality across frames
        avg_visibility = sum(obs.visibility for obs in observations) / len(observations)
        avg_blur = sum(obs.quality.get("blur", 0.5) for obs in observations) / len(observations)
        
        # High blur and low visibility means high aleatoric uncertainty
        aleatoric = ((1.0 - avg_visibility) + avg_blur) / 2.0
        aleatoric = min(1.0, max(0.0, aleatoric))
        
        # 2. Epistemic (Model) Uncertainty
        # Measure disagreement among OCR engines and temporal frames
        # If fusion_confidence is low despite good aleatoric conditions, epistemic uncertainty is high
        
        # Base epistemic on fusion confidence inverse
        epistemic = 1.0 - fusion_confidence
        
        # Penalize if we have very few observations (model hasn't seen enough)
        if len(observations) < 3:
            epistemic += 0.2
            
        epistemic = min(1.0, max(0.0, epistemic))
        
        return float(epistemic), float(aleatoric)
        
class DecisionEngine:
    def __init__(self, uncertainty_estimator: IUncertaintyEstimator):
        self.estimator = uncertainty_estimator
        
        # Thresholds
        self.ACCEPT_EPISTEMIC_MAX = 0.35
        self.ACCEPT_ALEATORIC_MAX = 0.60
        self.REPROCESS_ALEATORIC_MAX = 0.85

    def make_decision(self, observations: List[PlateObservation], fusion_conf: float, format_valid: bool) -> str:
        epistemic, aleatoric = self.estimator.estimate_uncertainty(observations, fusion_conf)
        
        # Rule 1: Unavailable information
        if aleatoric > self.REPROCESS_ALEATORIC_MAX:
            return "UNKNOWN"
            
        # Rule 2: Reprocess if uncertainty is borderline
        if epistemic > self.ACCEPT_EPISTEMIC_MAX or aleatoric > self.ACCEPT_ALEATORIC_MAX:
            return "REPROCESS"
            
        # Rule 3: Must match format
        if not format_valid:
            return "REPROCESS"
            
        # Rule 4: Accept
        return "ACCEPT"
