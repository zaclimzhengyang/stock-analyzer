import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go


st.title("📈 Quantitative Stock Analyzer")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

if "action" not in st.session_state:
    st.session_state.action = None

col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze"):
        st.session_state.action = "analyze"
with col2:
    if st.button("Run Backtest"):
        st.session_state.action = "backtest"

if ticker and st.session_state.action == "analyze":
    st.info("Fetching fundamental data...")
    try:
        url = f"http://localhost:8000/api/analyze/{ticker}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        st.subheader("📊 Fundamentals")
        st.json(data)
    except Exception as e:
        st.error(f"Error fetching fundamentals: {e}")

elif ticker and st.session_state.action == "backtest":
    st.info("Running backtest...")
    try:
        url = f"http://localhost:8000/api/backtest/{ticker}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        backtest_data = pd.Series(data)
        st.line_chart(backtest_data)

        st.success(f"Final portfolio value: ${backtest_data.iloc[-1, 0]:,.2f}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=backtest_data.index,
                y=backtest_data.iloc[:, 0],
                name="Portfolio Value",
            )
        )
        fig.update_layout(
            title="Backtest Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Value ($)",
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Backtest failed: {e}")
