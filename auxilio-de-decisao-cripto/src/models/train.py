import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(X_train, y_train, time_steps=60):
    # Definir o modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(time_steps, 1)))  # Usar time_steps e 1 feature
    model.add(Dense(1))  # Saída única para prever preço futuro

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    return model  # Retornar o modelo treinado
