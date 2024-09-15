from fastapi import FastAPI, Depends, HTTPException
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from pydantic import BaseModel

# Configurações do banco de dados PostgreSQL
DATABASE_URL = "postgresql://gustavomachado:243677@localhost:5432/rndr_token_predict"

# Conexão com o banco de dados usando SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definição das tabelas no banco de dados
class TokenPrediction(Base):
    __tablename__ = "token_predictions"
    id = Column(Integer, primary_key=True, index=True)
    token_name = Column(String, index=True)
    predicted_price = Column(Float)
    prediction_time = Column(DateTime, default=datetime.datetime.utcnow)
    model_version = Column(String, default="ARIMA-5-1-0")
    

# Cria as tabelas no banco de dados (se elas ainda não existirem)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Carrega os dados do arquivo CSV (exemplo)
rndr_df = pd.read_csv('../notebooks/data/RNDR_historical_data.csv')
rndr_df['date'] = pd.to_datetime(rndr_df['date'])
rndr_df.set_index('date', inplace=True)
rndr_df = rndr_df.asfreq('D')

# Variável global para armazenar o modelo treinado
model_fit = None

# Função para treinar o modelo ARIMA
def train_arima_model():
    global model_fit
    model = ARIMA(rndr_df['price'], order=(5, 1, 0))
    model_fit = model.fit()


# Modelo de dados para as previsões via POST
class PredictionCreate(BaseModel):
    token_name: str
    predicted_price: float
    prediction_time: datetime.datetime
    model_version: str = "ARIMA-5-1-0"

# Dependência para obter uma sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint para fazer previsões e salvar no banco de dados
@app.get("/predict")
def predict_price(days: int = 30, db: Session = Depends(get_db)):
    global model_fit
    if model_fit is None:
        raise HTTPException(status_code=400, detail="O modelo ainda não foi treinado. Chame o endpoint /train primeiro.")
    
    forecast = model_fit.forecast(steps=days)

    # Salva as previsões no banco de dados
    for i in range(days):
        predicted_price = float(forecast[i])  # Conversão para float padrão
        prediction = TokenPrediction(
            token_name="RNDR",
            predicted_price=predicted_price,
            prediction_time=datetime.datetime.utcnow() + datetime.timedelta(days=i)
        )
        db.add(prediction)
    
    db.commit()

    return {"forecast": forecast.tolist()}

# Rota POST para treinar o modelo e adicionar previsões ao banco de dados
@app.post("/train-and-predict")
def train_and_add_predictions(days: int = 30, db: Session = Depends(get_db)):
    global model_fit
    # Treina o modelo, se ainda não estiver treinado
    if model_fit is None:
        train_arima_model()

    # Faz previsões para o número de dias especificado
    forecast = model_fit.forecast(steps=days)

    # Salva as previsões no banco de dados
    for i in range(days):
        predicted_price = float(forecast[i])  # Converte np.float64 para float
        prediction_time = datetime.datetime.utcnow() + datetime.timedelta(days=i)
        prediction = TokenPrediction(
            token_name="RNDR",
            predicted_price=predicted_price,
            prediction_time=prediction_time,
            model_version="ARIMA-5-1-0"
        )
        db.add(prediction)
    
    db.commit()  # Confirma a transação e salva no banco de dados
    return {"message": f"{days} previsões adicionadas ao banco de dados."}

# Rota GET para listar todas as previsões no banco de dados
@app.get("/predictions/")
def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(TokenPrediction).all()
    return predictions

# Rota GET para obter uma previsão específica pelo ID
@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.query(TokenPrediction).filter(TokenPrediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Previsão não encontrada")
    return prediction

# Endpoint para treinar o modelo
@app.post("/train")
def train_model():
    train_arima_model()
    return {"message": "Modelo treinado com sucesso"}

if __name__ == "__main__":
    # Treina o modelo assim que a aplicação é iniciada
    train_arima_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
