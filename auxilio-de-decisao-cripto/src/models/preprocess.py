import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data):
    # Convert 'open_time' to datetime if not already
    data['open_time'] = pd.to_datetime(data['open_time'])

    # Filter only numerical columns (e.g., 'price_brl' as the target variable)
    features = data.select_dtypes(include=['float64', 'int64']).drop(columns=['price_brl'])

    # Target variable
    target = data['price_brl']

    return features, target
