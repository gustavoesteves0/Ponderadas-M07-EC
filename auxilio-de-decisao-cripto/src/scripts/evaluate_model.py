import pandas as pd
from src.models.model_manager import ModelManager

def evaluate_crypto_model(crypto_data_path: str):
    # Carregar dados de teste
    data = pd.read_csv(crypto_data_path)
    
    # Separar em variáveis de entrada e alvo
    X_test = data.drop("price_brl", axis=1)  
    y_test = data["price_brl"]  
    
    # Instanciar ModelManager e carregar o modelo
    model_manager = ModelManager()
    
    # Avaliar o modelo
    evaluation_metrics = model_manager.evaluate(X_test, y_test)
    print(f"Métricas de avaliação: {evaluation_metrics}")

if __name__ == "__main__":
    # Exemplo de uso
    evaluate_crypto_model("../../data/processed/crypto_prices/render_token_weekly_brl.csv")
