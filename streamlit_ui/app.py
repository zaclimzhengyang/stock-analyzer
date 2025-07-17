import streamlit as st
import requests
import pandas as pd

st.title('Quantitative Stock Analyzer')

ticker = st.text_input('Enter ticker', 'AAPL')

if st.button('Analyze'):
    url = f'http://localhost:8000/api/analyze/{ticker}'
    r = requests.get(url)
    st.write("Request object:", r)
    data = r.json()
    st.json(data)