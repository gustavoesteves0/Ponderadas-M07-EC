from models.model_manager import ModelManager
import pandas as pd
from data_loader import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def predict_price(model, X_test, scaler):
    # Fazer previsões com o modelo treinado
    predictions = model.predict(X_test)
    
    # Reverter a normalização dos preços previstos
    predictions = scaler.inverse_transform(predictions)
    
    return predictions


def train_model(X_train, y_train, time_steps=10):
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(time_steps, 1)))  # Expecting only 1 feature
    model.add(Dense(1))  # Single output for regression
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    save_model(model)

    
    return model