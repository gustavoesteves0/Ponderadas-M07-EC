import pandas as pd
from src.models.model_manager import ModelManager

def retrain_model(new_data_path: str):
    # Carregar novos dados
    new_data = pd.read_csv(new_data_path)

    # Instanciar ModelManager
    model_manager = ModelManager()

    # Treinar o modelo com os novos dados
    model_manager.train(new_data)
    print(f"Modelo re-treinado com sucesso.")

if __name__ == "__main__":
    # Exemplo de uso
    retrain_model("../../data/processed/crypto_prices/render_token_weekly_brl.csv")
