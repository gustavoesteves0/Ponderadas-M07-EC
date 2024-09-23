# FastAPI backend

from fastapi import FastAPI, Depends, HTTPException
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from pydantic import BaseModel

# Configurações do banco de dados PostgreSQL
DATABASE_URL = "postgresql://gustavomachado:newpassword@postgres:5432/rndr_token_predict"

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
    model_version = Column(String, default="LSTM")

# Cria as tabelas no banco de dados (se elas ainda não existirem)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Carrega os dados do arquivo CSV
rndr_df = pd.read_csv('RNDR_historical_data.csv')
rndr_df['date'] = pd.to_datetime(rndr_df['date'])
rndr_df.set_index('date', inplace=True)
rndr_df = rndr_df.asfreq('D')

# Variável global para armazenar os modelos treinados
lstm_model = None
gru_model = None
scaler = MinMaxScaler(feature_range=(0, 1))

# Função para preparar os dados (escalamento e criação de sequências)
def prepare_data_for_model(df, n_steps=60):
    scaled_data = scaler.fit_transform(df[['price']])
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to be [samples, time steps, features] as exigido pelos modelos LSTM/GRU
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Função para treinar o modelo LSTM
def train_lstm_model(df):
    global lstm_model
    X, y = prepare_data_for_model(df)

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=5, batch_size=32)

# Função para treinar o modelo GRU
def train_gru_model(df):
    global gru_model
    X, y = prepare_data_for_model(df)

    gru_model = Sequential()
    gru_model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    gru_model.add(GRU(units=50))
    gru_model.add(Dense(1))

    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X, y, epochs=5, batch_size=32)

# Função para prever com LSTM ou GRU
def predict_lstm_gru(model, days):
    last_60_days = rndr_df['price'][-60:].values
    scaled_data = scaler.transform(last_60_days.reshape(-1, 1))
    
    X_input = np.array(scaled_data).reshape(1, 60, 1)
    predictions = []
    
    for _ in range(days):
        pred_price = model.predict(X_input)[0][0]
        predictions.append(pred_price)
        
        # Atualiza o input com a nova previsão
        X_input = np.append(X_input[:, 1:, :], [[pred_price]], axis=1)
    
    # Retorna os valores escalados para o intervalo original
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Modelo de dados para as previsões via POST
class PredictionCreate(BaseModel):
    token_name: str
    predicted_price: float
    prediction_time: datetime.datetime
    model_version: str

# Dependência para obter uma sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint para fazer previsões e salvar no banco de dados
@app.get("/predict")
def predict_price(days: int, model_type: str, db: Session = Depends(get_db)):
    if model_type == "LSTM":
        forecast = predict_lstm_gru(lstm_model, days)
        model_version = "LSTM"
    elif model_type == "GRU":
        forecast = predict_lstm_gru(gru_model, days)
        model_version = "GRU"
    else:
        raise HTTPException(status_code=400, detail="Invalid model type.")

    # Salva as previsões no banco de dados
    for i in range(days):
        predicted_price = float(forecast[i])  # Converte para float padrão
        prediction = TokenPrediction(
            token_name="RNDR",
            predicted_price=predicted_price,
            prediction_time=datetime.datetime.utcnow() + datetime.timedelta(days=i),
            model_version=model_version
        )
        db.add(prediction)
    
    db.commit()

    return {"forecast": forecast.tolist()}

# Rota POST para treinar o modelo
@app.post("/train")
def train_model(model_type: str):
    if model_type == "LSTM":
        train_lstm_model(rndr_df)
    elif model_type == "GRU":
        train_gru_model(rndr_df)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type.")
    
    return {"message": f"Modelo {model_type} treinado com sucesso."}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
