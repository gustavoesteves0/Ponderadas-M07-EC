from fastapi import FastAPI, HTTPException
from .models import ModelManager
from .schemas import TrainRequest, PredictRequest, PredictResponse
from .services import get_db, log_usage

app = FastAPI()

# Instantiate the model manager
model_manager = ModelManager()

# Endpoint to train the model
@app.post("/train")
def train_model(request: TrainRequest):
    start_date = request.start_date
    end_date = request.end_date
    
    # Train the model with the provided parameters
    try:
        model_manager.train(start_date, end_date)
        # Ideally save the trained model to a file
        model_manager.save(f"../models/render_token_weekly_brl.pkl")
        return {"message": f"Modelo treinado para Render de {start_date} até {end_date}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao treinar o modelo: {e}")

# Endpoint to make predictions
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    crypto = request.crypto

    try:
        # Load the trained model before prediction
        model_manager.load(f"../models/{crypto}_model.pkl")
        prediction = model_manager.predict(crypto)
        # Log the API usage
        log_usage(crypto)
        return {"crypto": crypto, "prediction": prediction}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo não encontrado. Treine o modelo antes de realizar previsões.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar previsão: {e}")

# Basic healthcheck
@app.get("/healthcheck")
def healthcheck():
    return {"status": "API is running."}
