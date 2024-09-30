from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    # Faz previsões nos dados de teste
    predictions = model.predict(X_test)

    # Calcula métricas de avaliação
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    return {"MAE": mae, "RMSE": rmse}
