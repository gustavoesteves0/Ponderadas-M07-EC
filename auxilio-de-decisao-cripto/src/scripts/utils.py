from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Configuração para conexão com o banco de dados Postgres
DATABASE_URL = "postgresql://postgres:postgres23@postgres:5432/crypto_model_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def convert_timestamp_to_date(timestamp):
    # Exemplo de função auxiliar
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
