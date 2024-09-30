# app.py

Este arquivo é a aplicação principal em Streamlit, responsável por criar a interface de usuário e interagir com o modelo. Ele permite aos usuários treinar o modelo com dados de preços de criptomoedas, fazer previsões e visualizar resultados.

- **Importações**: Importa módulos e funções necessárias de `components`, `data_loader`, `models`, e `utils`.
- **Configuração do Streamlit**: Define a configuração da página do dashboard.
- **Título e Descrição**: Exibe o título e a descrição do dashboard.
- **Carregamento e Exibição de Dados**: 
  - Carrega os dados históricos de preços de criptomoedas usando `load_data()`.
  - Exibe os dados como um gráfico de linha com `open_time` no eixo x e `price_brl` no eixo y.
- **Preparação dos Dados para Treinamento**: Pré-processa os dados para serem usados no treinamento e teste, criando sequências temporais dos preços das criptomoedas.
- **Treinamento do Modelo**: Fornece um botão para treinar um modelo nos dados usando `train_model()`. Exibe mensagens de progresso e sucesso/erro.
- **Fazer Previsões**: Fornece um botão para usar o modelo treinado para fazer previsões de preços com os dados de teste.
  - Converte as previsões em um `DataFrame` para visualização.
  - Exibe as previsões com `open_time` no eixo x usando um gráfico `Altair`.
- **Avaliar Performance do Modelo**: Calcula e exibe métricas de performance do modelo, comparando previsões e preços reais.

