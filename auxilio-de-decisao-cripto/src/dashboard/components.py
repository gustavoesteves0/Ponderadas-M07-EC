import streamlit as st
from models.evaluate import evaluate_model

def display_metrics(model, X_test, y_test):
    # Call the evaluate_model function to get the evaluation metrics
    metrics = evaluate_model(model, X_test, y_test)
    
    # Display the metrics using Streamlit
    st.write(f"Métricas de avaliação para Render Token:")
    st.write(f"- **MAE**: {metrics['MAE']:.2f}")
    st.write(f"- **RMSE**: {metrics['RMSE']:.2f}")

def display_predictions(prediction: float):
    # Display the prediction result
    st.write(f"Preço previsto: ${prediction.flatten()[0]:.2f}")
