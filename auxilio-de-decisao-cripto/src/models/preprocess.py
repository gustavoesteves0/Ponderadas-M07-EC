import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data, test_size=0.2, random_state=42):
    data['open_time'] = pd.to_datetime(data['open_time'])
    features = data.select_dtypes(include=['float64', 'int64']).drop(columns=['price_brl'])
    target = data['price_brl']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
