import streamlit as st
from models.evaluate import evaluate_model

def display_metrics(model, X_test, y_test):
    # Get evaluation metrics
    metrics = evaluate_model(model, X_test, y_test)
    
    # Display the metrics using Streamlit
    st.write(f"Métricas de avaliação para Render Token:")
    st.write(f"- **MAE**: {metrics['MAE']:.2f}")
    st.write(f"- **RMSE**: {metrics['RMSE']:.2f}")

def display_predictions(predictions):
    # If predictions contain multiple values, show a line chart
    if len(predictions) > 1:
        st.line_chart(predictions)
    else:
        # Display the prediction result if it's a single value
        st.write(f"Preço previsto: ${predictions[0]:.2f}")
