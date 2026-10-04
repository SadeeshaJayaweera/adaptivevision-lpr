# 🚔 AdaptiveVision-LPR

> **A research-grade, production-oriented, uncertainty-aware, and disaster-resilient AI vehicle license plate recognition platform.**

Unlike conventional `Camera → YOLO → OCR` pipelines that hallucinate characters under adverse conditions, **AdaptiveVision-LPR** acts as an intelligent perception platform. It actively diagnoses environmental conditions, assesses physical visibility, routes frames through targeted dynamic enhancements, and utilizes Temporal Bayesian Fusion to construct character evidence. 

If the information is physically destroyed by mud, heavy occlusion, or catastrophic disaster conditions, the system safely returns **`UNKNOWN`** rather than a high-confidence false positive.

---

## 🌟 Core Research Features

* **Condition-Aware Routing**: Evaluates live environmental noise (Low Light, Heavy Rain, Glare, Motion Blur) and conditionally triggers pre-processing (CLAHE, Unsharp Masking, Super Resolution).
* **Physical Occlusion & Recoverability AI**: Employs an explicit `IVisibilityAnalyzer` to measure structural plate degradation. It rejects unrecoverable frames before wasting computational cycles on OCR.
* **Temporal Bayesian Fusion**: OCR engines are highly volatile. This platform aligns strings across up to 15 frames, acting as a Bayesian voter *per character index*. Random OCR hallucinations rarely survive temporal consensus.
* **Uncertainty Estimation Layer**: Separates Aleatoric (Data Noise) and Epistemic (Model Disagreement) uncertainty to make mathematical `ACCEPT`, `REPROCESS`, or `UNKNOWN` decisions.
* **Sri Lankan Plate Intelligence**: Features a specialized `IValidationEngine` that cross-checks Sri Lankan province formats and autocorrects domain-specific OCR character confusions.
* **Disaster Mode Telemetry**: Real-time `CameraHealthMonitor` tracks network jitter and FPS drops to lower latency thresholds and heighten persistence during catastrophic events.

---

## 🏗️ Architecture Overview

The system transitions from a linear pipeline to a **Directed Acyclic Graph (DAG) Adaptive Router**.

```text
Camera / RTSP Stream
        ↓
Camera Health Analysis
        ↓
Condition Classifier (Illumination, Weather, Blur)
        ↓
Vehicle & Plate Detection (YOLO)
        ↓
Occlusion & Recoverability Estimator
        ↓
Adaptive Processing Router (e.g., Glare Suppression, Deblur)
        ↓
OCR Ensemble (EasyOCR / PaddleOCR)
        ↓
Temporal Evidence Fusion
        ↓
Uncertainty Estimation
        ↓
Decision Engine (ACCEPT / REPROCESS / UNKNOWN)
```

---

## 💻 Tech Stack

* **Machine Learning / Vision**: PyTorch, OpenCV, Ultralytics YOLOv8, EasyOCR
* **Backend Pipeline**: Python 3.11, FastAPI, SQLAlchemy, Uvicorn, AsyncIO
* **Frontend UI**: React 18, TypeScript, Vanilla CSS (Glassmorphism, Dark Mode)
* **Data & Persistence**: PostgreSQL, Redis (Queues)
* **Infrastructure**: Docker, Docker Compose

---

## 🚀 Getting Started

### Prerequisites
* Docker & Docker Compose
* *(Optional)* NVIDIA GPU with CUDA for real-time video stream throughput. Mac users will utilize CPU/MPS scaling.

### Run with Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SadeeshaJayaweera/adaptivevision-lpr.git
   cd adaptivevision-lpr
   ```

2. **Spin up the stack:**
   ```bash
   docker-compose up --build -d
   ```

3. **Access the platform:**
   * **Dashboard**: [http://localhost:3000](http://localhost:3000)
   * **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Repository Structure

```text
adaptivevision-lpr/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI Endpoints & WebSocket routers
│   │   ├── camera/         # Simulator and Health Telemetry
│   │   ├── core/           # Abstract Interfaces (DAG routing contracts)
│   │   ├── database/       # SQLAlchemy Persistence
│   │   ├── inference/      # Detection, Condition, Occlusion, Fusion, Uncertainty logic
│   │   └── models/         # Pydantic Schemas & Domain Enums
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Responsive React UI Dashboard
│   │   └── index.css       # Premium Design System (Glassmorphism)
│   └── Dockerfile
├── ml/                     # ML Model Weights & Notebooks
├── data/                   # Evaluative Dataset (Raw, Annotations, Conditions)
├── scripts/                # Dataset generators and tools
├── docs/                   # Research Methodology & Final Reports
└── docker-compose.yml
```

---

## 📄 License
This project is developed for specialized research into disaster-resilient LPR systems and uncertainty-aware deep learning.
