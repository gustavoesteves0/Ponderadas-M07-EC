from fastapi import FastAPI, Depends, HTTPException
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from pydantic import BaseModel
import logging
import os

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

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Função para obter uma sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Carrega os dados do arquivo CSV
rndr_df = pd.read_csv('RNDR_historical_data.csv')
rndr_df['date'] = pd.to_datetime(rndr_df['date'])
rndr_df.set_index('date', inplace=True)
rndr_df = rndr_df.asfreq('D')

# Variável global para armazenar os modelos treinados
scaler = MinMaxScaler(feature_range=(0, 1))

# Função para preparar os dados 
def reshape_input_data(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])  # Appending timesteps of data as a sequence
        y.append(data[i + timesteps])    # Appending the next value as the target
    
    X = np.array(X).reshape(-1, timesteps, 1)  # Adding 1 for the feature dimension
    y = np.array(y)
    
    return X, y

def prepare_data_for_model(df, timesteps=60):
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    X, y = reshape_input_data(scaled_data, timesteps)
    return X, y

# Train the LSTM model and save it
def train_lstm_model():
    X_train, y_train = prepare_data_for_model(rndr_df)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(100, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)
    
    # Save the trained LSTM model
    lstm_model.save('lstm_model.h5')
    return lstm_model

# Train the GRU model and save it
def train_gru_model():
    X_train, y_train = prepare_data_for_model(rndr_df)
    
    gru_model = Sequential()
    gru_model.add(GRU(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    gru_model.add(Dropout(0.2))
    gru_model.add(GRU(100))
    gru_model.add(Dropout(0.2))
    gru_model.add(Dense(50))
    gru_model.add(Dense(1))
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    
    gru_model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)
    
    # Save the trained GRU model
    gru_model.save('gru_model.h5')
    return gru_model

# Predict future prices
def predict_future_prices(model, last_sequence, future_days):
    predictions = []
    current_sequence = last_sequence

    for _ in range(future_days):
        next_price = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1))
        predictions.append(next_price[0][0])

        current_sequence = np.append(current_sequence[1:], next_price)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Endpoint for prediction
@app.get("/predict")
def predict_price(days: int, model_type: str, db: Session = Depends(get_db)):
    try:
        # Load the respective model
        if model_type == "LSTM":
            if not os.path.exists('lstm_model.h5'):
                raise HTTPException(status_code=400, detail="LSTM model not trained.")
            model = load_model('lstm_model.h5')
        elif model_type == "GRU":
            if not os.path.exists('gru_model.h5'):
                raise HTTPException(status_code=400, detail="GRU model not trained.")
            model = load_model('gru_model.h5')
        else:
            raise HTTPException(status_code=400, detail="Invalid model type.")

        # Prepare the last sequence of data for prediction
        scaled_data = scaler.fit_transform(rndr_df['price'].values.reshape(-1, 1))
        last_sequence = scaled_data[-60:]

        # Predict future prices
        forecast = predict_future_prices(model, last_sequence, days)

        # Save predictions to the database
        for i, predicted_price in enumerate(forecast):
            prediction = TokenPrediction(
                token_name="RNDR",
                predicted_price=float(predicted_price),
                prediction_time=datetime.datetime.utcnow() + datetime.timedelta(days=i),
                model_version=model_type
            )
            db.add(prediction)

        db.commit()
        return {"forecast": forecast.tolist()}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Train model endpoint
@app.post("/train")
def train_model(model_type: str):
    try:
        if model_type == "LSTM":
            train_lstm_model()
        elif model_type == "GRU":
            train_gru_model()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type.")
        return {"message": f"Modelo {model_type} treinado com sucesso."}
    except Exception as e:
        logging.error(f"Error training {model_type} model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
