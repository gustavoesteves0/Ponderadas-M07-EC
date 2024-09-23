import streamlit as st
import requests

# Título do dashboard
st.title("Previsão de Preço do Render Token")

# Seletor de dias de previsão
days = st.slider("Escolha o número de dias para prever:", 1, 90, 30)

# Seletor do modelo de previsão
model_choice = st.selectbox("Escolha o modelo de previsão:", ("LSTM", "GRU"))

# Função para treinar o modelo escolhido
def train_model(model_type):
    response = requests.post(f"http://backend:8000/train?model_type={model_type}")
    return response.json()

# Função para prever o preço
def predict_price(model_type, days):
    response = requests.get(f"http://backend:8000/predict?days={days}&model_type={model_type}")
    return response.json()

# Botão para treinar o modelo
if st.button("Treinar Modelo"):
    train_response = train_model(model_choice)
    st.write(train_response["message"])

# Botão para buscar previsões
if st.button("Prever Preço"):
    data = predict_price(model_choice, days)
    
    # Exibindo previsões
    if 'forecast' in data:
        st.line_chart(data['forecast'])
    else:
        st.write("Erro ao prever preços:", data.get("detail", "Detalhes não disponíveis."))

