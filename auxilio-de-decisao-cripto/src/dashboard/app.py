import streamlit as st
from components import display_metrics, display_predictions
from data_loader import load_data
from models.preprocess import prepare_data
from model_interface import predict_price, train_model
from utils.models_utils import load_model, save_model
import matplotlib.pyplot as plt
from styles import set_custom_styles

st.set_page_config(page_title="Crypto Investment Dashboard", layout="wide")

set_custom_styles()

# Title and description
st.title("Sistema de Auxílio para Investimento em Cripto Ativos")
st.markdown("Este dashboard permite analisar e prever preços de criptomoedas, fornecendo insights para auxiliar decisões de compra e venda.")

# Load and display historical data
data = load_data()
st.subheader(f"Dados Históricos de Render Token")
st.line_chart(data.set_index("open_time")["price_brl"])

# Split data for training and testing
X_train, X_test, y_train, y_test = prepare_data(data)

# Train the model if button is clicked
model = None
if st.button("Treinar Modelo"):
    st.info("Treinando o modelo, por favor aguarde...")
    try:
        model = train_model(X_train, y_train)  # Use the training data
        save_model(model, "../models/")
        st.success("Modelo treinado com sucesso!")
    except Exception as e:
        st.error(f"Ocorreu um erro ao treinar o modelo: {e}")


# Button for prediction
if st.button("Realizar Previsão"):
    prediction = predict_price(model, X_test)  # Use the model and test data for predictions
    display_predictions(prediction)
    st.success(f"Previsão realizada com sucesso para Render Token!")


model = train_model(X_train, y_train)  # Train the model using training data
# Display model metrics
st.subheader("Métricas do Modelo")
display_metrics(model, X_test, y_test)  # Pass model, X_test, and y_test to display metrics
