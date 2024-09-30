from pydantic import BaseModel
from typing import Optional

# Esquema para a requisição de treinamento
class TrainRequest(BaseModel):
    crypto: str
    start_date: str
    end_date: str

# Esquema para a requisição de previsão
class PredictRequest(BaseModel):
    crypto: str

# Esquema para a resposta da previsão
class PredictResponse(BaseModel):
    crypto: str
    prediction: Optional[float]