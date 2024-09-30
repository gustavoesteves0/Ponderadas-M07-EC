import pandas as pd
import os
import joblib


MODEL_PATH = os.getenv('MODEL_PATH', "/app/src/models/crypto_model.pkl")

def load_data():
    # Use an environment variable or configuration file for path
    file_path = os.getenv('DATA_PATH', "/app/data/processed/crypto_prices/render_token_weekly_brl.csv")
    
    # Load data
    data = pd.read_csv(file_path)
    return data


def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)