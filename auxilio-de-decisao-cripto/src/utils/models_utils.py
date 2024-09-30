import joblib

def save_model(model, path: str):
    """Salva um modelo treinado em um arquivo."""
    joblib.dump(model, path)

def load_model(path: str):
    """Carrega um modelo salvo de um arquivo."""
    return joblib.load(path)

def evaluate_model(model, X, y, metrics: list):
    """Avalia o modelo com base nas métricas fornecidas."""
    results = {}
    for metric in metrics:
        results[metric.__name__] = metric(y, model.predict(X))
    return results
