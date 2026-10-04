from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
from pydantic import BaseModel
from backend.app.websocket.manager import manager

router = APIRouter()

class InferenceRequest(BaseModel):
    image_base64: str
    camera_id: str

@router.post("/inference")
async def process_inference(request: InferenceRequest):
    # This will be connected to the AI pipeline queue
    # For now it's a stub
    return {"status": "queued", "camera_id": request.camera_id}

@router.get("/detections")
async def get_detections(limit: int = 50, offset: int = 0):
    # Stub for DB call
    return []

@router.get("/detections/{detection_id}")
async def get_detection(detection_id: str):
    # Stub for DB call
    return {"id": detection_id}

@router.get("/vehicles")
async def get_vehicles(limit: int = 50, offset: int = 0):
    # Stub for DB call
    return []

@router.get("/statistics")
async def get_statistics():
    # Stub for DB call
    return {"accuracy": 0.0, "latency": 0.0, "fps": 0.0}

@router.get("/health")
async def health_check():
    return {"status": "ok", "db": "ok", "redis": "ok"}

@router.get("/models")
async def get_models():
    return {"detection": "yolov8n", "ocr": ["easyocr", "paddleocr"]}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming commands if necessary
    except WebSocketDisconnect:
        manager.disconnect(websocket)
