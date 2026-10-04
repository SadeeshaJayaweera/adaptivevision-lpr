from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

from backend.app.core.interfaces import IFusionEngine
from backend.app.models.domain import CharacterEvidence, PlateObservation

class TemporalBayesianFusion(IFusionEngine):
    def __init__(self):
        pass

    def fuse_characters(self, character_evidence: List[CharacterEvidence]) -> str:
        """
        Fuses character evidence at a specific position across multiple frames/engines.
        Uses a weighted voting / simple Bayesian update mechanism.
        """
        if not character_evidence:
            return "?"
            
        # Prior probabilities
        probs = defaultdict(float)
        
        for ev in character_evidence:
            # We scale the probability by engine trust and condition impact.
            # E.g., PaddleOCR might be weighted 1.2, EasyOCR 1.0
            engine_weight = 1.0
            
            # Simple additive probability log (proxy for Bayesian)
            probs[ev.candidate] += (ev.probability * engine_weight)
            
        if not probs:
            return "?"
            
        # Return the character with the highest accumulated probability mass
        best_candidate = max(probs.items(), key=lambda x: x[1])
        return best_candidate[0]

    def fuse_temporal(self, observations: List[PlateObservation]) -> Tuple[str, float]:
        """
        Aligns multiple plate observations over time and fuses them character by character.
        """
        if not observations:
            return "", 0.0
            
        # In a real system, you must align the strings because OCR engines might miss a character 
        # (e.g. "ABC1234" vs "AB1234"). We use a basic positional mapping here assuming fixed length 
        # or left-aligned strings for Sri Lankan plates (typically 7-8 chars).
        
        # Max length observed
        max_len = max(len(ocr.text) for obs in observations for ocr in obs.ocr_results if ocr.text)
        if max_len == 0:
            return "", 0.0
            
        # Position -> List of CharacterEvidence
        position_evidence: Dict[int, List[CharacterEvidence]] = defaultdict(list)
        
        total_ocr_confidence = 0.0
        ocr_count = 0
        
        for obs in observations:
            for ocr in obs.ocr_results:
                text = ocr.text.replace(" ", "")
                total_ocr_confidence += ocr.confidence
                ocr_count += 1
                
                # Align to right if it looks like numbers, left if letters...
                # For simplicity in this baseline, we just left-align.
                for i, char in enumerate(text):
                    char_conf = ocr.character_confidences[i] if i < len(ocr.character_confidences) else ocr.confidence
                    
                    position_evidence[i].append(CharacterEvidence(
                        candidate=char,
                        probability=char_conf,
                        source="temporal",
                        frame_id=obs.frame_id,
                        engine=ocr.engine,
                        condition=obs.conditions.weather.name # Using weather as proxy
                    ))
                    
        fused_string = ""
        for i in range(max_len):
            if i in position_evidence:
                fused_char = self.fuse_characters(position_evidence[i])
                fused_string += fused_char
                
        # Approximate agreement confidence
        agreement_conf = total_ocr_confidence / ocr_count if ocr_count > 0 else 0.0
        
        return fused_string, agreement_conf
