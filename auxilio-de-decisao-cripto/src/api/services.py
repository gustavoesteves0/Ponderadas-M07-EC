from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

# Configuration for PostgreSQL database connection
DATABASE_URL = "postgresql://postgres23@postgres:5432/crypto_model_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to log API usage
def log_usage(crypto: str):
    # Here you could save the usage logs to a database
    # Example: save to a logging table or store as a log file
    print(f"Log: Previsão realizada para {crypto} em {datetime.now()}")