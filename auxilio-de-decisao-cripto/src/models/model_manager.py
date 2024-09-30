import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import load_model
from models.evaluate import evaluate_model

class ModelManager:
    def __init__(self, model_name="crypto_model.pkl"):
        self.model = None
        self.model_path = f"src/models/{model_name}"

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
        
        # Build LSTM model with Input layer
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Single output for regression

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        return model  # Return the trained model
    
    def predict_price(self, features):
        # Load the model if not in memory
        if self.model is None:
            self.load_model()
        
        # Make prediction
        return self.model.predict(features)
    
    def evaluate(self, X_test, y_test):
        # Avalia o modelo
        return evaluate_model(self.model, X_test, y_test)
