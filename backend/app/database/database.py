from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Uses a local SQLite database for development and testing.
# In production, swap this to PostgreSQL: postgresql+psycopg2://user:password@host/dbname
SQLALCHEMY_DATABASE_URL = "sqlite:///./adaptivevision.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
