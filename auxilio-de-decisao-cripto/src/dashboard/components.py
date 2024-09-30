import streamlit as st
from models.evaluate import evaluate_model
import pandas as pd
import altair as alt


def display_metrics(model, X_test, y_test):
    # Get evaluation metrics
    metrics = evaluate_model(model, X_test, y_test)
    
    # Display the metrics using Streamlit
    st.write(f"Métricas de avaliação para Render Token:")
    st.write(f"- **MAE**: {metrics['MAE']:.2f}")
    st.write(f"- **RMSE**: {metrics['RMSE']:.2f}")

def display_predictions(predictions):
    # Ensure 'open_time' is in datetime format
    predictions['open_time'] = pd.to_datetime(predictions['open_time'])
    
    # Create Altair chart for better x-axis control
    chart = alt.Chart(predictions).mark_line().encode(
        x='open_time:T',  # T indicates temporal data
        y='predicted_price:Q'  # Q indicates quantitative data
    ).properties(
        title="Predicted Price Over Time"
    ).interactive()  # Makes the chart interactive (zoom, pan)

    # Display the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)
