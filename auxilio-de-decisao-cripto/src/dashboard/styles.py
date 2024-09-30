import streamlit as st

def set_custom_styles():
    st.markdown(
        """
        <style>
        /* Customiza o título */
        .stTitle {
            font-family: 'Arial Black', sans-serif;
            color: #2E8B57;
        }
        /* Customiza o fundo da sidebar */
        .css-1aumxhk {
            background-color: #f4f4f4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
