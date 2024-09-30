# model_interface.py

Contém funções que interagem com o modelo para treinamento e previsões.

- **`train_model(X_train, y_train, time_steps=10)`**:
  - Constrói e treina um modelo LSTM usando os dados de treinamento fornecidos.
  - Configura o modelo com a forma de entrada e parâmetros apropriados.
- **`predict_price(model, X_test, scaler)`**:
  - Usa o modelo treinado e os dados de teste para fazer previsões.
  - Reverte a normalização dos preços previstos usando `scaler`.

