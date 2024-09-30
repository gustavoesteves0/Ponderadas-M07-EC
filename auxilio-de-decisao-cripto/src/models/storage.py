import joblib

def save_model(model, model_path):
    # Salva o modelo no caminho especificado
    joblib.dump(model, model_path)

def load_model(model_path):
    # Carrega o modelo do caminho especificado
    return joblib.load(model_path)
