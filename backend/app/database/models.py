from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
import datetime
import uuid

Base = declarative_base()

class DBEventModel(Base):
    __tablename__ = "detection_events"

    event_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    vehicle_track_id = Column(String, index=True)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    
    plate = Column(String, index=True, nullable=True)
    status = Column(String) # ACCEPT, REPROCESS, UNKNOWN
    failure_reason = Column(String, nullable=True)
    
    confidence = Column(Float)
    recoverability = Column(Float)
    frames_used = Column(Integer)
    
    # Store complex JSON payloads
    conditions = Column(JSON)
    pipeline = Column(JSON)
    ocr_engines = Column(JSON)
    evidence = Column(JSON)
    model_versions = Column(JSON)
    
    # Relationships
    observations = relationship("DBPlateObservation", back_populates="event")

class DBPlateObservation(Base):
    __tablename__ = "plate_observations"
    
    observation_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String, ForeignKey("detection_events.event_id"))
    frame_id = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    visibility = Column(Float)
    occlusion = Column(Float)
    occlusion_type = Column(String)
    
    applied_enhancements = Column(JSON)
    ocr_results = Column(JSON)
    
    event = relationship("DBEventModel", back_populates="observations")

class DBCameraHealth(Base):
    __tablename__ = "camera_health"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    health = Column(Float)
    fps = Column(Float)
    frame_drop_rate = Column(Float)
    lens_obstruction = Column(Float)
    network_quality = Column(Float)
