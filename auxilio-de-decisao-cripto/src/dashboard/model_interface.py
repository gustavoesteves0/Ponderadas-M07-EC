from models.model_manager import ModelManager
import pandas as pd

def predict_price():
    # Cria uma instância do gerenciador de modelo
    model_manager = ModelManager()
    
    df = pd.read_csv('data/processed/crypto_prices/render_token_weekly_brl.csv')

    # Define os recursos de entrada para a previsão 
    features = [df["price_brl"]]
    
    # Faz a previsão
    prediction = model_manager.predict(features)
    
    return prediction  # Retorna o preço previsto

def train_model(X_train, y_train):
    # Cria uma instância do gerenciador de modelo
    model_manager = ModelManager()
    
    # Treina o modelo com os dados fornecidos
    model_manager.train(X_train, y_train)
    
    # Retorna uma mensagem de sucesso
    return "Modelo treinado e salvo com sucesso!"