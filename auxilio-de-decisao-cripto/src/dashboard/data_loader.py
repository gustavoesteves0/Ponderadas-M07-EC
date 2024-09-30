import pandas as pd
import os

def load_data():
    # Use an environment variable or configuration file for path
    file_path = os.getenv('DATA_PATH', "/app/data/processed/crypto_prices/render_token_weekly_brl.csv")
    
    # Load data
    data = pd.read_csv(file_path)
    return data
