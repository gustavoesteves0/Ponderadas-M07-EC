import pandas as pd

def normalize_data(df: pd.DataFrame, column: str):
    """Normaliza uma coluna de um DataFrame entre 0 e 1."""
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop"):
    """Lida com valores ausentes em um DataFrame."""
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill_zero":
        return df.fillna(0)
    # Adicione outras estratégias conforme necessário
