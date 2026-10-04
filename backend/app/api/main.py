from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np

# We import the abstract interfaces and the implementations we built
from backend.app.inference.condition import SceneConditionAnalyzer
from backend.app.inference.router import AdaptiveEnhancementRouter
from backend.app.inference.occlusion import PlateOcclusionAnalyzer, EvidenceRecoverabilityEstimator
from backend.app.inference.fusion import TemporalBayesianFusion
from backend.app.inference.validation import SriLankanPlateValidator
from backend.app.inference.uncertainty import EpistemicAleatoricUncertainty, DecisionEngine
from backend.app.inference.detection import YOLOVehicleDetector, YOLOPlateDetector
from backend.app.inference.ocr import EasyOCRBaseline

from backend.app.models.domain import PlateObservation, CompoundCondition, DecisionState, EventModel
import datetime
import uuid

from backend.app.api.endpoints import router as api_router

app = FastAPI(
    title="AdaptiveVision-LPR API",
    description="Research-grade, uncertainty-aware AI vehicle license plate recognition platform.",
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
# In production, use dependency injection or lifespan events.
condition_analyzer = SceneConditionAnalyzer()
occlusion_analyzer = PlateOcclusionAnalyzer()
recoverability_estimator = EvidenceRecoverabilityEstimator()
router = AdaptiveEnhancementRouter()
validation_engine = SriLankanPlateValidator()
uncertainty_estimator = EpistemicAleatoricUncertainty()
decision_engine = DecisionEngine(uncertainty_estimator)

# Placeholder models that would typically be loaded on startup
# vehicle_detector = YOLOVehicleDetector()
# plate_detector = YOLOPlateDetector()
# ocr_engine = EasyOCRBaseline()

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.post("/api/inference/simulate")
async def simulate_inference(file: UploadFile = File(...)):
    """
    Simulates the DAG Adaptive Routing pipeline on a single frame.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        # 1. Condition Analysis
        condition: CompoundCondition = condition_analyzer.analyze_scene(frame)
        
        # 2. Mock Detection (Skipping YOLO inference here for latency/memory on local dev)
        # We assume the plate is in the center for testing
        h, w = frame.shape[:2]
        plate_bbox_mock = type('obj', (object,), {'x_min': w*0.3, 'y_min': h*0.4, 'x_max': w*0.7, 'y_max': h*0.6, 'confidence': 0.99})
        
        # 3. Visibility and Occlusion
        visibility_data = occlusion_analyzer.analyze(frame, plate_bbox_mock)
        
        # 4. Recoverability Estimation
        recov_class, recov_conf = recoverability_estimator.estimate(visibility_data, condition)
        
        # 5. Adaptive Routing
        pipeline = router.determine_pipeline(condition, visibility_data)
        
        # 6. Apply enhancements
        enhanced_frame, executed_steps = router.execute_pipeline(frame, pipeline)
        
        # Return a rich JSON breakdown reflecting the research architecture
        return {
            "event_id": str(uuid.uuid4()),
            "condition_classification": {
                "illumination": condition.illumination.name,
                "weather": condition.weather.name,
                "motion": condition.motion.name,
                "optics": condition.optics.name
            },
            "occlusion_analysis": visibility_data,
            "recoverability": {
                "class": recov_class.name,
                "confidence": recov_conf
            },
            "adaptive_routing": {
                "pipeline_selected": pipeline,
                "steps_executed": executed_steps
            },
            "message": "Adaptive Processing Successful. Ready for Temporal Fusion."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.app.api.main:app", host="0.0.0.0", port=8000, reload=True)
