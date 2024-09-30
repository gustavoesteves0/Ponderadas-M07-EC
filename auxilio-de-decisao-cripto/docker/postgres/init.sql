CREATE TABLE IF NOT EXISTS crypto_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    crypto VARCHAR(50),
    price DECIMAL
);

-- Insert any initial data if necessary
INSERT INTO crypto_data (timestamp, crypto, price) VALUES 
('2023-01-01 00:00:00', 'render-token', 16800.34),
('2023-01-01 01:00:00', 'render-token', 16850.12);
