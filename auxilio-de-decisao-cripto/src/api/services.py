from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

# Configuração para conexão com o banco de dados Postgres
DATABASE_URL = "postgresql://postgres23@postgres:5432/crypto_model_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Função para logar o uso da API
def log_usage(crypto: str):
    # Aqui você poderia salvar os logs de uso em um banco de dados
    print(f"Log: Previsão realizada para {crypto} em {datetime.now()}")
