from fastapi import FastAPI, HTTPException
from .models import ModelManager
from .schemas import TrainRequest, PredictRequest, PredictResponse
from .services import get_db, log_usage

app = FastAPI()

# Instancia o gerenciador do modelo
model_manager = ModelManager()

# Endpoint para treinar o modelo
@app.post("/train")
def train_model(request: TrainRequest):
    # Extrai parâmetros do corpo da requisição
    crypto = request.crypto
    start_date = request.start_date
    end_date = request.end_date
    
    # Treina o modelo com os parâmetros fornecidos
    try:
        model_manager.train(crypto, start_date, end_date)
        return {"message": f"Modelo treinado para {crypto} de {start_date} até {end_date}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para realizar previsões
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Extrai parâmetros do corpo da requisição
    crypto = request.crypto

    try:
        # Faz a previsão
        prediction = model_manager.predict(crypto)
        # Loga a utilização da API
        log_usage(crypto)
        return {"crypto": crypto, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Healthcheck básico
@app.get("/healthcheck")
def healthcheck():
    return {"status": "API is running."}
