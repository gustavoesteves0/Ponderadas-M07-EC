import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_data(data, time_steps=10, test_size=0.2):
    # Select the relevant column and remove missing values
    data = data[['price_brl']].dropna()

    # Normalize the data (use a scaler if necessary)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences of data
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps, 0])  # Features: sequences of past prices
            y.append(data[i + time_steps, 0])    # Labels: future price (next step)
        return np.array(X), np.array(y)

    # Create sequences for training
    X, y = create_sequences(scaled_data, time_steps)

    # Reshape to [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 1 feature as expected

    # Split the data into train and test sets
    split = int(X.shape[0] * (1 - test_size))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    return X_train, X_test, y_train, y_test, scaler