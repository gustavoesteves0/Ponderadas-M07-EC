from src.models.model_manager import ModelManager 
import pandas as pd
import tensorflow as tf

def evaluate_crypto_model(crypto_data_path: str):
    # Load test data
    data = pd.read_csv(crypto_data_path)
    
    # Separate features and target
    X_test = data.drop("price_brl", axis=1)
    y_test = data["price_brl"]
    
    # Convert to numpy and add new dimension for model compatibility
    X_test = X_test.to_numpy()
    X_test = tf.expand_dims(X_test, axis=1)  # Add a new dimension at index 1
    
    # Instantiate ModelManager and load the model
    model_manager = ModelManager()
    
    # Evaluate the model and print evaluation metrics
    evaluation_metrics = model_manager.evaluate(X_test, y_test)
    print(f"Evaluation Metrics: {evaluation_metrics}")

if __name__ == "__main__":
    # Example usage
    evaluate_crypto_model("../../data/processed/crypto_prices/render_token_weekly_brl.csv")
