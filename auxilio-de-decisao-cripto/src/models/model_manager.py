from models.preprocess import prepare_data
from models.train import train_model
from models.evaluate import evaluate_model
from models.storage import save_model, load_model
from models.preprocess import prepare_data
from models.train import train_model
import joblib

class ModelManager:
    def __init__(self, model_name="crypto_model.pkl"):
        self.model = None
        self.model_path = f"src/models/{model_name}"


    def train(self, start_date: str, end_date: str):
            save_model(self.model, self.model_path)
            # Aqui você coleta e processa os dados para treinamento
            data = "../../data/processed/crypto_prices/render_token_weekly_brl.csv"
            start_date="2023-09-29"
            end_date="2024-10-06"
            # Preprocess the data
            X_train, y_train = prepare_data(data)

            # Train the model
            self.model = train_model(X_train, y_train)

            # Save the trained model
            # Treinamento do modelo (exemplo usando scikit-learn)
            self.model = self.train_model(data)
            # Salva o modelo treinado
            joblib.dump(self.model, self.model_path)


    def predict(self, features):
        # Carrega o modelo se não estiver em memória
        if self.model is None:
            self.model = load_model(self.model_path)
        
        # Faz a previsão com os recursos fornecidos
        return self.model.predict(features)

    def evaluate(self, X_test, y_test):
        # Avalia o modelo
        return evaluate_model(self.model, X_test, y_test)
