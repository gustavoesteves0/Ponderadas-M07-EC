from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configurações de conexão com o banco de dados
DATABASE_URL = "postgresql://user:password@localhost/dbname"

# Configuração do engine e da sessão
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Gera uma sessão de banco de dados para uso."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
