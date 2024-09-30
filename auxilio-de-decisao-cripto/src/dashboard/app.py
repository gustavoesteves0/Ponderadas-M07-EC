import streamlit as st
from components import display_metrics, display_predictions
from data_loader import load_data
from models.preprocess import prepare_data
from model_interface import predict_price, train_model
from utils.models_utils import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crypto Investment Dashboard", layout="wide")

# Título e descrição do dashboard
st.title("Sistema de Auxílio para Investimento em Cripto Ativos")
st.markdown("Este dashboard permite analisar e prever preços de criptomoedas, fornecendo insights para auxiliar decisões de compra e venda.")

# Carrega e exibe os dados históricos
data = load_data()
st.subheader(f"Dados Históricos de Render Token")
st.line_chart(data.set_index("open_time")["price_brl"])

# Splitting data for training and evaluation (You need to implement prepare_data or adapt it as per your needs)
X_train, y_train = prepare_data(data)

# Train the model if button is clicked
if st.button("Treinar Modelo"):
    st.info("Treinando o modelo, por favor aguarde...")
    # Trigger model training
    try:
        model = train_model(X_train, y_train)  # Use the training data
        st.success("Modelo treinado com sucesso!")
    except Exception as e:
        st.error(f"Ocorreu um erro ao treinar o modelo: {e}")

model = load_model("../models/crypto_model.pkl")  # Load the trained model

# Botão para realizar previsões
if st.button("Realizar Previsão"):
    prediction = predict_price(model, X_test)  # Use the model and test data to make predictions
    display_predictions(prediction)
    st.success(f"Previsão realizada com sucesso para Render Token!")

# Exibe métricas do modelo
st.subheader("Métricas do Modelo")
display_metrics(model, X_test, y_test)  # Pass model, X_test, and y_test to display metrics
