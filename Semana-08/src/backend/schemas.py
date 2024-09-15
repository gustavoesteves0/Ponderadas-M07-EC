from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Configuração da conexão com o PostgreSQL
DATABASE_URL = "postgresql://gustavomachado:243677@127.0.0.1/rndr_token_predict"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definição da tabela de logs
class Log(Base):
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    method = Column(String, index=True)
    status_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Criando a tabela no banco de dados
Base.metadata.create_all(bind=engine)
