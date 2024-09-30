# components.py


Contém funções auxiliares para exibir previsões e métricas no dashboard `Streamlit`.

- **`display_predictions(predictions)`**:
  - Recebe um `DataFrame` contendo `open_time` e `predicted_price`.
  - Exibe um gráfico interativo de linha usando `Altair` para mostrar os preços previstos ao longo do tempo.
- **`display_metrics(model, X_test, y_test)`**:
  - Exibe métricas de performance do modelo (e.g., MAE, RMSE) com base nos dados de teste.
