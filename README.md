# README.md

# Sistema de Auxílio para Investimento em Cripto Ativos

## Índice

1. [Sobre o Projeto](#sobre-o-projeto)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Como Iniciar o Projeto](#como-iniciar-o-projeto)
4. [Como Usar o Dashboard](#como-usar-o-dashboard)
5. [Detalhes Técnicos](#detalhes-técnicos)
6. [Como Contribuir](#como-contribuir)
7. [Contato](#contato)

---

## Sobre o Projeto

Este projeto é um **Sistema de Auxílio para Investimento em Cripto Ativos** que permite aos usuários visualizar dados históricos de preços de criptomoedas, treinar modelos de Machine Learning para prever preços futuros e avaliar a performance desses modelos. A aplicação é construída utilizando `Streamlit`, e o modelo preditivo é baseado em uma rede neural LSTM. É interessante observar que o modelo é treinado com dados históricos de preços de criptomoedas, e as previsões são feitas com base nesses dados. Tais dados não representam a realidade e não devem ser utilizados para fins de investimento. Porém, para a finalidade do projeto, eles são suficientes para demonstrar o funcionamento do sistema. Para continuidade do projeto, entraria no roadmap a criação de novas features para o modelo, como a inclusão de novas variáveis, a otimização de hiperparâmetros e a implementação de uma estratégia de investimento baseada nas previsões. Um exemplo de criação de novas features levaria em conta a inclusão de dados de expectativa, visto que o mercado de criptoativos é altamente influenciado por notícias e eventos. Para essa inclusão, pode ser utilizado um modelo de LLDA (Latent Dirichlet Allocation) para classificar notícias e eventos em categorias, e então incluir essas categorias como features no modelo preditivo. Por motivos de tempo, não foi possível implementar essa feature no projeto atual, mas fica estabelecida uma possibilidade de melhoria para o futuro.
Além diss, vale ressaltar que apesar de válida para fins de projeto, a implementação de um datalake se faz desnecessária visto que o volume de dados é baixo e pode ser facilmente manipulado em um ambiente local. Para um volume maior de dados, a implementação de um datalake seria necessária para garantir a escalabilidade do sistema.

## Estrutura do Projeto

A estrutura do projeto é organizada da seguinte forma:
    
```
auxilio-de-decisao-cripto/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│   ├── preprocess.py
│   └── model_interface.py
│
├── notebooks/
│
├── src/
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── data_loader.py
│   │   └── models/
│   │       ├── model_interface.py
│   │       └── preprocess.py
│   │
│   └── utils/
│
└── tests/
```

### Principais Pastas e Arquivos:

- **`/data`**: Contém os dados brutos e processados utilizados no projeto.
- **`/src`**: Contém o código-fonte principal, incluindo modelos, scripts, dashboard e utilitários.
- **`/dashboard`**: Contém o aplicativo `Streamlit` para a interface com o usuário.
- **`/models`**: Inclui scripts para treino, avaliação e armazenamento do modelo preditivo.
- **`/utils`**: Funções auxiliares para operações como manipulação de dados e logging.
- **`README.md`**: Arquivo que fornece a documentação do projeto.

## Como Iniciar o Projeto

### Pré-requisitos

- **Python 3.8+**
- **Poetry** (para gerenciamento de dependências)
- **Docker** (opcional, para deploy)

### Passos para Iniciar

1. **Clone o Repositório**
   ```bash
   git clone https://github.com/gustavoesteves0/Ponderadas-M07-EC
   cd Ponderadas-M07-EC
   ```
3. **Inicie a docker**
```bash
   cd docker
   docker-compose up --build
   ```
4. Acesse o dashboard no navegador usando o link fornecido na saída do terminal.

## Como Usar o Dashboard

1. **Visualize Dados Históricos**: A seção inicial exibe gráficos de preços históricos de criptomoedas.
   
   - Assim que você inicia o dashboard, os dados históricos são carregados e exibidos em um gráfico de linha. Este gráfico apresenta a variação de preços ao longo do tempo, facilitando a análise visual.

2. **Treine o Modelo**: Use o botão "Treinar Modelo" para treinar um novo modelo com os dados carregados.
   
   - Para treinar o modelo, clique no botão "Treinar Modelo". O modelo será treinado com os dados pré-processados, e você verá uma mensagem indicando o sucesso ou falha do processo.

3. **Faça Previsões**: Use o botão "Realizar Previsão" para prever os preços futuros das criptomoedas.
   
   - Após o modelo ser treinado, você pode clicar no botão "Realizar Previsão" para usar o modelo e prever os preços futuros. As previsões serão exibidas em um gráfico de linha, mostrando a tendência esperada para a criptomoeda selecionada.

4. **Visualize Métricas**: Confira as métricas de performance do modelo treinado.
   
   - Após fazer previsões, as métricas de desempenho do modelo serão exibidas. Essas métricas ajudam a avaliar a precisão das previsões e a eficácia do modelo.
