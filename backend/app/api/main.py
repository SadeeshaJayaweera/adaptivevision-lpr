from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import base64
import datetime
import uuid

# Domain and Inference modules
from backend.app.inference.condition import SceneConditionAnalyzer
from backend.app.inference.router import AdaptiveEnhancementRouter
from backend.app.inference.occlusion import PlateOcclusionAnalyzer, EvidenceRecoverabilityEstimator
from backend.app.inference.fusion import TemporalBayesianFusion
from backend.app.inference.validation import SriLankanPlateValidator
from backend.app.inference.uncertainty import EpistemicAleatoricUncertainty, DecisionEngine
from backend.app.inference.detection import YOLOVehicleDetector, YOLOPlateDetector, find_plate_contour
from backend.app.inference.ocr import EasyOCRBaseline

from backend.app.models.domain import BoundingBox, PlateObservation, CompoundCondition, DecisionState, EventModel
from backend.app.api.endpoints import router as api_router
from backend.app.database.database import engine
from backend.app.database.models import Base

# Ensure SQLite / Postgres tables are initialized
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AdaptiveVision-LPR API",
    description="Production-grade, uncertainty-aware AI vehicle license plate recognition platform.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# --- Global Component Initialization ---
condition_analyzer = SceneConditionAnalyzer()
occlusion_analyzer = PlateOcclusionAnalyzer()
recoverability_estimator = EvidenceRecoverabilityEstimator()
router = AdaptiveEnhancementRouter()
validation_engine = SriLankanPlateValidator()
uncertainty_estimator = EpistemicAleatoricUncertainty()
decision_engine = DecisionEngine(uncertainty_estimator)

# Vehicle detector, Plate detector & EasyOCR Engine
vehicle_detector = YOLOVehicleDetector()
ocr_engine = EasyOCRBaseline(use_gpu=False)
plate_detector = YOLOPlateDetector(ocr_engine=ocr_engine)

def encode_image_base64(img: np.ndarray) -> str:
    """Encodes an OpenCV image to base64 data URI."""
    if img is None or img.size == 0:
        return ""
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.get("/")
def read_root():
    return {
        "name": "AdaptiveVision-LPR API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.post("/api/inference/simulate")
async def simulate_inference(file: UploadFile = File(...)):
    """
    Executes real machine learning & computer vision license plate recognition:
    1. Plate & vehicle localization (YOLO + morphological contour analysis)
    2. Environmental condition assessment (glare, blur, low light)
    3. Occlusion and recoverability estimation
    4. Condition-adaptive ML enhancement (specular glare suppression, CLAHE, deblur, super-res)
    5. Optical character recognition (EasyOCR alphanumeric engine)
    6. Sri Lankan plate format verification and OCR confusion correction
    7. Epistemic and aleatoric uncertainty quantification & decision (ACCEPT, REPROCESS, UNKNOWN)
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
            
        h, w = frame.shape[:2]
        
        # Step 1: Vehicle & Plate Detection
        vehicles = vehicle_detector.detect(frame)
        v_box = vehicles[0] if vehicles else BoundingBox(x_min=0.0, y_min=0.0, x_max=float(w), y_max=float(h), confidence=0.85)
        
        plates = plate_detector.detect(frame, v_box)
        plate_box = plates[0] if plates else BoundingBox(
            x_min=float(w * 0.2), y_min=float(h * 0.35),
            x_max=float(w * 0.8), y_max=float(h * 0.75),
            confidence=0.85
        )
        
        # Extract initial raw plate crop
        px1 = max(0, int(plate_box.x_min))
        py1 = max(0, int(plate_box.y_min))
        px2 = min(w, int(plate_box.x_max))
        py2 = min(h, int(plate_box.y_max))
        
        # Add small margin if possible
        margin_x = int((px2 - px1) * 0.05)
        margin_y = int((py2 - py1) * 0.05)
        crop_x1 = max(0, px1 - margin_x)
        crop_y1 = max(0, py1 - margin_y)
        crop_x2 = min(w, px2 + margin_x)
        crop_y2 = min(h, py2 + margin_y)
        
        raw_plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2] if (crop_x2 > crop_x1 and crop_y2 > crop_y1) else frame
        
        # Step 2: Condition Analysis on frame and plate crop
        condition: CompoundCondition = condition_analyzer.analyze_scene(frame, plate_roi=raw_plate_crop)
        quality_metrics = condition_analyzer.get_quality_metrics(raw_plate_crop)
        
        # Step 3: Visibility and Occlusion Analysis
        visibility_data = occlusion_analyzer.analyze(frame, plate_box)
        
        # Step 4: Recoverability Estimation
        recov_class, recov_conf = recoverability_estimator.estimate(visibility_data, condition)
        
        # Step 5: Determine Adaptive Enhancement Pipeline
        pipeline = router.determine_pipeline(condition, visibility_data)
        
        # Step 6: Execute ML & CV Enhancement Operations
        enhanced_crop, executed_steps = router.execute_pipeline(raw_plate_crop, pipeline)
        
        # Step 7: OCR Inference with EasyOCR (Dual-pass evaluation)
        ocr_result_enh = ocr_engine.recognize(enhanced_crop)
        ocr_result_raw = ocr_engine.recognize(raw_plate_crop)
        
        # Validate both
        format_valid_enh, val_plate_enh, val_reason_enh = validation_engine.validate_format(ocr_result_enh.text)
        format_valid_raw, val_plate_raw, val_reason_raw = validation_engine.validate_format(ocr_result_raw.text)
        
        # Pick the best verified prediction
        if format_valid_enh and not format_valid_raw:
            ocr_result = ocr_result_enh
            format_valid, validated_plate, val_reason = format_valid_enh, val_plate_enh, val_reason_enh
        elif format_valid_raw and not format_valid_enh:
            ocr_result = ocr_result_raw
            format_valid, validated_plate, val_reason = format_valid_raw, val_plate_raw, val_reason_raw
        elif ocr_result_enh.confidence >= ocr_result_raw.confidence and ocr_result_enh.text:
            ocr_result = ocr_result_enh
            format_valid, validated_plate, val_reason = format_valid_enh, val_plate_enh, val_reason_enh
        else:
            ocr_result = ocr_result_raw
            format_valid, validated_plate, val_reason = format_valid_raw, val_plate_raw, val_reason_raw
            
        final_plate_text = validated_plate if format_valid else (ocr_result.text or "UNKNOWN")
        
        # Step 9: Epistemic & Aleatoric Uncertainty Estimation & Decision
        observation = PlateObservation(
            frame_id=str(uuid.uuid4())[:8],
            timestamp=datetime.datetime.utcnow(),
            track_id="trk-" + str(uuid.uuid4())[:6],
            plate_bbox=plate_box,
            vehicle_bbox=v_box,
            quality=quality_metrics,
            conditions=condition,
            ocr_results=[ocr_result],
            visibility=visibility_data["visibility"],
            occlusion=visibility_data["occlusion"],
            occlusion_type=visibility_data["occlusion_type"],
            recoverability=float(recov_conf),
            visible_character_regions=visibility_data.get("visible_character_regions", 0.8),
            applied_enhancements=executed_steps
        )
        
        epistemic_unc, aleatoric_unc = uncertainty_estimator.estimate_uncertainty(
            [observation], ocr_result.confidence
        )
        decision = decision_engine.make_decision(
            [observation], ocr_result.confidence, format_valid
        )
        
        # Step 10: Create Annotated Image with Plate Bounding Box
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (59, 130, 246), 3) # Blue box
        # Tag label background
        label_text = f"{final_plate_text} ({int(ocr_result.confidence * 100)}%)" if final_plate_text != "UNKNOWN" else "PLATE REGION"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_frame, (px1, max(0, py1 - lh - 12)), (px1 + lw + 10, py1), (59, 130, 246), -1)
        cv2.putText(annotated_frame, label_text, (px1 + 5, max(18, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode visual previews for production frontend
        raw_crop_b64 = encode_image_base64(raw_plate_crop)
        enhanced_crop_b64 = encode_image_base64(enhanced_crop)
        annotated_frame_b64 = encode_image_base64(annotated_frame)
        
        # Generate Sauvola local binarized pixel view for crisp character inspection
        gray_enh = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2GRAY)
        th_sauv = cv2.blur(gray_enh.astype(np.float32), (21, 21))
        sq_th = cv2.blur(gray_enh.astype(np.float32)**2, (21, 21))
        std_th = np.sqrt(np.maximum(0, sq_th - th_sauv**2))
        sauv_map = (gray_enh.astype(np.float32) > (th_sauv * (1.0 + 0.22 * ((std_th / 128.0) - 1.0)))) * 255
        sauvola_crop = sauv_map.astype(np.uint8)
        sauvola_b64 = encode_image_base64(cv2.cvtColor(sauvola_crop, cv2.COLOR_GRAY2BGR))
        
        # Active Condition Tags
        active_conditions = []
        if condition.illumination.name != "NORMAL":
            active_conditions.append(condition.illumination.name)
        if condition.weather.name != "CLEAR":
            active_conditions.append(condition.weather.name)
        if condition.motion.name not in ["STATIC", "LOW"]:
            active_conditions.append(f"{condition.motion.name}_MOTION")
        if not active_conditions:
            active_conditions.append("OPTIMAL_CONDITIONS")

        return {
            "event_id": observation.frame_id,
            "plate": final_plate_text,
            "raw_ocr": ocr_result.text,
            "status": decision,
            "confidence": round(float(ocr_result.confidence), 3),
            "format_valid": format_valid,
            "validation_reason": val_reason if not format_valid else "Sri Lankan Format Verified",
            "conditions": active_conditions,
            "condition_classification": {
                "illumination": condition.illumination.name,
                "weather": condition.weather.name,
                "motion": condition.motion.name,
                "optics": condition.optics.name
            },
            "quality_metrics": quality_metrics,
            "occlusion_analysis": visibility_data,
            "recoverability": {
                "class": recov_class.name,
                "confidence": round(float(recov_conf), 2)
            },
            "adaptive_routing": {
                "pipeline_selected": pipeline,
                "steps_executed": executed_steps
            },
            "uncertainty": {
                "epistemic": round(float(epistemic_unc), 3),
                "aleatoric": round(float(aleatoric_unc), 3)
            },
            "original_crop_preview": raw_crop_b64,
            "enhanced_plate_preview": enhanced_crop_b64,
            "sauvola_crop_preview": sauvola_b64,
            "annotated_frame_preview": annotated_frame_b64,
            "message": f"Adaptive pipeline executed. Decision: {decision}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.app.api.main:app", host="0.0.0.0", port=8000, reload=True)
