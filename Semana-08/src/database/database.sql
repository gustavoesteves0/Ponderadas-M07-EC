-- Conecta ao banco de dados
\c rndr_token_predict;

-- Criação da tabela para armazenar o histórico de preços de tokens
CREATE TABLE token_prices (
    id SERIAL PRIMARY KEY,
    token_name VARCHAR(50) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Criação da tabela para armazenar previsões geradas pelo modelo
CREATE TABLE token_predictions (
    id SERIAL PRIMARY KEY,
    token_name VARCHAR(50) NOT NULL,
    predicted_price DECIMAL(18, 8) NOT NULL,
    prediction_time TIMESTAMP NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criação da tabela para armazenar os logs de uso do sistema
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(255) NOT NULL,
    performed_by VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Tabela de treinamento do modelo (histórico de treinos)
CREATE TABLE model_training (
    id SERIAL PRIMARY KEY,
    token_name VARCHAR(50) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    accuracy DECIMAL(5, 2),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
