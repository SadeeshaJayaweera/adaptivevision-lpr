# AdaptiveVision-LPR

Research-grade, production-oriented **AI-powered robust vehicle license plate recognition system**.

AdaptiveVision-LPR is an **Adaptive Multi-Condition Temporal Fusion License Plate Recognition System** capable of recognizing vehicle number plates under difficult visual conditions including daylight, night, low illumination, headlight glare, blur, rain, and perspective distortion.

## Core Architecture

The central intelligence of the system relies on an **Adaptive Routing Pipeline**:
> **Detect the visual condition → dynamically select the appropriate enhancement pipeline → detect and rectify the plate → perform OCR using multiple evidence sources → fuse results across multiple video frames → estimate uncertainty → accept, reject, or reprocess the prediction.**

## Features
- **Condition-Aware AI**: Automatically estimates image quality and visual conditions.
- **Adaptive Enhancement Router**: Routes frames to targeted enhancement pipelines (Retinex, HDR-style, Super-Resolution) rather than applying blanket operations.
- **Multi-OCR Ensemble**: Abstracted OCR layer evaluating PaddleOCR, EasyOCR, and Tesseract.
- **Temporal Evidence Fusion**: Character-position-level voting across multiple frames for robust vehicle tracking.
- **Sri Lankan License Plate Intelligence**: Localized plate format validation and intelligent OCR correction.
- **Uncertainty-Aware Decision Engine**: Intelligent accept/reprocess/reject mechanisms.
- **Real-Time Edge Dashboard**: WebSocket-enabled React dashboard for live tracking and analytics.

## Structure
- `backend/`: FastAPI application, databases, and AI pipelines.
- `frontend/`: React/Vite dashboard.
- `ml/`: Model training, baselines, and experiment tracking.

*Project setup in progress.*
