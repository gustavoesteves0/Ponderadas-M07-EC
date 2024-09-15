import streamlit as st
import requests

# Título do dashboard
st.title("Previsão de Preço do Render Token")

# Seletor de dias de previsão
days = st.slider("Escolha o número de dias para prever:", 1, 90, 30)

# Botão para buscar previsões
if st.button("Prever Preço"):
    response = requests.get(f"http://localhost:8000/predict?days={days}")
    data = response.json()
    
    # Exibindo previsões
    st.line_chart(data['forecast'])
