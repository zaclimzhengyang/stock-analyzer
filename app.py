import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go

st.title("📈 Quantitative Stock Analyzer")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

if "action" not in st.session_state:
    st.session_state.action = None

col1, col2 = st.columns(2)
col3 = st.columns(1)[0]

with col1:
    if st.button("Analyze"):
        st.session_state.action = "analyze"
with col2:
    if st.button("Run Backtest"):
        st.session_state.action = "backtest"
with col3:
    if col3.button("Monte Carlo Simulation"):
        st.session_state.action = "monte_carlo"

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

elif ticker and st.session_state.action == "monte_carlo":
    st.info("Running Monte Carlo simulation...")
    try:
        url = f"http://localhost:8000/api/monte-carlo-sim/{ticker}"
        r = requests.get(url)
        r.raise_for_status()
        sim_data = np.array(r.json())

        df_sim = pd.DataFrame(sim_data)

        fig = go.Figure()

        for col in df_sim.columns:
            fig.add_trace(go.Scatter(
                x=df_sim.index,
                y=df_sim[col],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            ))

        fig.update_layout(
            title="Monte Carlo Simulation of Portfolio Performance",
            xaxis_title="Days",
            yaxis_title="Simulated Portfolio Value ($)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Simulation failed: {e}")

