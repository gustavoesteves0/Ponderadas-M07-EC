import pandas as pd

def load_data():
    # Use absolute path inside the container
    data = pd.read_csv("/app/data/processed/crypto_prices/render_token_weekly_brl.csv")
    return data