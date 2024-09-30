import streamlit as st
from components import display_metrics, display_predictions
from data_loader import load_data, load_model
from models.preprocess import prepare_data
from model_interface import predict_price, train_model
from utils.models_utils import save_model
import matplotlib.pyplot as plt
from styles import set_custom_styles
import pandas as pd
import numpy as np

st.set_page_config(page_title="Crypto Investment Dashboard", layout="wide")

set_custom_styles()

# Define time_steps as a constant
time_steps = 10

# Title and description
st.title("Sistema de Auxílio para Investimento em Cripto Ativos")
st.markdown("Este dashboard permite analisar e prever preços de criptomoedas, fornecendo insights para auxiliar decisões de compra e venda.")

# Load and display historical data
data = load_data()
print("Loaded data sample:", data.head())
st.subheader(f"Dados Históricos de Render Token")
st.line_chart(data.set_index("open_time")["price_brl"])

# Prepare data
X_train, X_test, y_train, y_test, scaler = prepare_data(data, time_steps=time_steps)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Train the model if button is clicked
if st.button("Treinar Modelo"):
    st.info("Treinando o modelo, por favor aguarde...")
    try:
        # Train the model
        model = train_model(X_train, y_train, time_steps=time_steps)
        save_model(model, "/app/src/models/crypto_model.pkl")
        st.success("Modelo treinado com sucesso!")
    except Exception as e:
        st.error(f"Ocorreu um erro ao treinar o modelo: {e}")

# Button for prediction
if st.button("Realizar Previsão"):
    model = load_model()
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Make predictions
    prediction = model.predict(X_test)
    
    # Create `open_time` sequence for predictions (replace this with your actual sequence if available)
    open_time = pd.date_range(start='2023-09-29', periods=len(prediction), freq='D')  # Adjust start date and frequency as necessary
    
    # Create DataFrame from predictions
    predictions_df = pd.DataFrame({
        'open_time': open_time,
        'predicted_price': prediction.flatten()  # Ensure it's a 1D array
    })
    
    # Display predictions using Altair chart
    display_predictions(predictions_df)
    st.success(f"Previsão realizada com sucesso para Render Token!")

# Compare predictions with actual values
y_test_inversed = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

st.subheader("Métricas do Modelo")
display_metrics(model, X_test, y_test_inversed)  # Pass model, X_test, and y_test to display metrics
