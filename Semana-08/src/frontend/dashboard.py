import streamlit as st
import requests

st.title("Previsão de Preço do Render Token")

# Seletor de dias de previsão
days = st.slider("Escolha o número de dias para prever:", 1, 90, 30)

# Seletor do modelo de previsão
model_choice = st.selectbox("Escolha o modelo de previsão:", ("LSTM", "GRU"))

# Função para treinar o modelo escolhido
def train_model(model_type):
    try:
        response = requests.post(f"http://backend:8000/train", params={"model_type": model_type})
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                st.error("O servidor não retornou JSON válido.")
                return None
        else:
            st.error(f"Falha ao treinar o modelo. Código de status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Um erro ocorreu: {e}")
        return None

# Função para prever o preço
def predict_price(model_choice, days):
    try:
        response = requests.get(f"http://backend:8000/predict?days={days}&model_type={model_choice}")
        if response.status_code == 200:
            if response.content:
                return response.json()
            else:
                st.error("Nenhum dado retornado pelo servidor.")
                return None
        else:
            st.error(f"Erro {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Um erro ocorreu: {e}")
        return None

# Botão para treinar o modelo
if st.button("Treinar Modelo"):
    train_response = train_model(model_choice)
    if train_response and "message" in train_response:
        st.write(train_response["message"])
    else:
        st.error("Falha ao treinar o modelo ou sem resposta do servidor.")

# Botão para buscar previsões
if st.button("Prever Preço"):
    data = predict_price(model_choice, days)
    if data and 'forecast' in data:
        st.line_chart(data['forecast'])
    else:
        st.error("Previsão de dados ausente ou inválida.")