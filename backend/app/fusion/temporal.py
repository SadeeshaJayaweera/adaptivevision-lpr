from typing import List
from collections import defaultdict
from backend.app.models.domain import OCRResult, FinalPrediction, DecisionState, PlateObservation
from backend.app.validation.sri_lanka import SriLankanPlateValidator

class TemporalFusionEngine:
    def __init__(self, confidence_threshold=0.9, reprocess_threshold=0.55):
        self.validator = SriLankanPlateValidator()
        self.confidence_threshold = confidence_threshold
        self.reprocess_threshold = reprocess_threshold

    def fuse_track(self, track_id: str, observations: List[PlateObservation]) -> FinalPrediction:
        """
        Takes a list of observations for a single vehicle track and performs character-level voting.
        """
        if not observations:
            raise ValueError("No observations provided for fusion")
            
        # Group by character position
        position_votes = defaultdict(lambda: defaultdict(float))
        
        total_ocr_results = 0
        for obs in observations:
            for ocr in obs.ocr_results:
                total_ocr_results += 1
                text = self.validator.normalize(ocr.text)
                weight = ocr.confidence * obs.quality.quality_score
                for i, char in enumerate(text):
                    # We might have character_confidences, but if not we use the overall confidence
                    char_conf = ocr.character_confidences[i] if i < len(ocr.character_confidences) else ocr.confidence
                    position_votes[i][char] += (char_conf * weight)
                    
        if total_ocr_results == 0:
             return FinalPrediction(
                track_id=track_id,
                plate="",
                confidence=0.0,
                frames_used=len(observations),
                ocr_agreement=0.0,
                temporal_agreement=0.0,
                format_valid=False,
                decision=DecisionState.UNKNOWN
            )

        # Find the most common string length among all OCR results
        lengths = [len(self.validator.normalize(ocr.text)) for obs in observations for ocr in obs.ocr_results]
        target_length = max(set(lengths), key=lengths.count) if lengths else 0
        
        final_chars = []
        correct_votes_weight = 0.0
        total_votes_weight = 0.0
        
        for i in range(target_length):
            if i in position_votes and position_votes[i]:
                # Find the character with the maximum weight at this position
                best_char, best_weight = max(position_votes[i].items(), key=lambda x: x[1])
                final_chars.append(best_char)
                
                correct_votes_weight += best_weight
                total_votes_weight += sum(position_votes[i].values())
                
        raw_fused_plate = "".join(final_chars)
        
        # Calculate temporal agreement (percentage of weight supporting the winning characters)
        temporal_agreement = correct_votes_weight / total_votes_weight if total_votes_weight > 0 else 0.0
        
        # Validate format
        is_valid, formatted_plate, reason = self.validator.validate_and_format(raw_fused_plate)
        
        # Decision Logic
        if is_valid and temporal_agreement >= self.confidence_threshold:
            decision = DecisionState.ACCEPT
        elif temporal_agreement >= self.reprocess_threshold:
            decision = DecisionState.REPROCESS
        else:
            decision = DecisionState.UNKNOWN
            
        return FinalPrediction(
            track_id=track_id,
            plate=formatted_plate,
            confidence=temporal_agreement,
            frames_used=len(observations),
            ocr_agreement=0.0, # Simplification: could compare inter-engine agreement
            temporal_agreement=temporal_agreement,
            format_valid=is_valid,
            decision=decision
        )
