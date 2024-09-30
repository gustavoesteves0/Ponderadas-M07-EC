import pandas as pd
from src.models.model_manager import ModelManager

def train_crypto_model(crypto_data_path: str):
    # Carregar dados processados
    data = pd.read_csv(crypto_data_path)

    # Instanciar ModelManager
    model_manager = ModelManager()

    # Treinar o modelo com os dados
    model_manager.train(data)
    print(f"Modelo treinado e salvo com sucesso.")

if __name__ == "__main__":
    # Exemplo de uso
    train_crypto_model("../../data/processed/crypto_prices/render_token_weekly_brl.csv")
