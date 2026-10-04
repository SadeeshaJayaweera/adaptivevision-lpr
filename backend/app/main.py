from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router as api_router

app = FastAPI(
    title="AdaptiveVision-LPR API",
    description="Adaptive Multi-Condition Temporal Fusion License Plate Recognition System",
    version="1.0.0"
)

# CORS configuration for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to AdaptiveVision-LPR"}
