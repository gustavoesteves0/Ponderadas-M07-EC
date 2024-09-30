from models.model_manager import ModelManager
import pandas as pd
from data_loader import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def predict_price(model, features):
    # Make sure model is not None
    if model is None:
        raise ValueError("The model has not been trained or loaded.")
    
    # Use model to predict with provided features
    return model.predict(features)

def train_model(X_train, y_train, time_steps=1):
    # Convert X_train and y_train to numpy arrays if they're not
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Debugging: print the shape of X_train before reshaping
    print(f"Original X_train shape: {X_train.shape}")
    
    # If X_train has 2 dimensions, add a time step dimension
    if len(X_train.shape) == 2:
        # Reshape X_train to (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], time_steps, X_train.shape[1]))

    # Debugging: print the shape of X_train after reshaping
    print(f"Reshaped X_train shape: {X_train.shape}")
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Single output for regression

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    return model  # Return the trained model
