# models/preprocess.py

Este arquivo cuida do pré-processamento dos dados brutos de preços de criptomoedas antes de serem usados para treinar o modelo.

- **`prepare_data(data, time_steps=10)`**:
  - Recebe um `DataFrame` bruto dos preços das criptomoedas e o prepara para treinamento de LSTM.
  - Normaliza os valores `price_brl`.
  - Converte os dados em sequências de comprimento `time_steps` para serem usados pela LSTM.
  - Divide os dados em conjuntos de treino e teste.
