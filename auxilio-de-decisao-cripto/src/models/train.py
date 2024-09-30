import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_model(X_train, y_train, time_steps=10, features=1):
    # Reshape X_train for LSTM [samples, time steps, features]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape X_train to match new input shape (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], time_steps, features))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(time_steps, features)))
    model.add(Dropout(0.2))  # Optional: add dropout to prevent overfitting
    model.add(Dense(1))  # Single output for regression

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    return model