from models.preprocess import prepare_data
from models.train import train_model
from models.evaluate import evaluate_model
from models.storage import save_model, load_model
import joblib
import pandas as pd
import os

class ModelManager:
    def __init__(self, model_name="crypto_model.pkl"):
        self.model = None
        self.model_path = os.path.join("src/models", model_name)

    def train(self, crypto: str, start_date: str, end_date: str):
        # Define the path to the data (use config/env variable if necessary)
        data_path = os.path.join("../../data/processed/crypto_prices", f"{crypto}_weekly_brl.csv")
        
        # Load and filter the data
        data = pd.read_csv(data_path)
        # Filter data based on start_date and end_date (assuming 'date' column exists)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # Preprocess the data
        X_train, y_train = prepare_data(data)

        # Train the model
        self.model = train_model(X_train, y_train)

        # Save the trained model
        save_model(self.model, self.model_path)

    def predict(self, features):
        # Load the model if not already in memory
        if self.model is None:
            self.model = load_model(self.model_path)
        
        # Make predictions using the model
        return self.model.predict(features)

    def evaluate(self, X_test, y_test):
        # Evaluate the model
        return evaluate_model(self.model, X_test, y_test)

    def save(self):
        # Save the model using joblib for consistency with `load_model()`
        joblib.dump(self.model, self.model_path)
    
    def load(self):
        # Load the model using joblib
        if self.model is None:
            self.model = joblib.load(self.model_path)