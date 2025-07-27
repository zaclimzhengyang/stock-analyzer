import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go

from constants import metrics

def _rename_columns(df: pd.DataFrame) -> None:
    df.rename(index=metrics, inplace=True)

def get_request(url: str) -> requests.Response:
    r = requests.get(url)
    r.raise_for_status()
    return r

st.title("📈 Quantitative Stock Analyzer")

with st.form("ticker_form"):
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")
    submitted = st.form_submit_button("Run All Analysis")

if submitted:
    st.info("Running all analyses...")

    # Create placeholders so content can be shown side-by-side or section-by-section
    fundamentals_container = st.container()
    backtest_container = st.container()
    mc_container = st.container()

    try:
        # === Fundamental Data ===
        fundamentals_container.subheader("📊 Fundamentals")
        r1 = get_request(f"http://localhost:8000/api/analyze/{ticker}")
        fundamentals_data = r1.json()

        # Convert dict to single-column DataFrame
        fundamentals_df = pd.DataFrame.from_dict(fundamentals_data, orient='index', columns=['Value'])
        fundamentals_df.index.name = 'Metric'

        _rename_columns(fundamentals_df)

        fundamentals_container.table(fundamentals_df)

        # === Backtest Data ===
        backtest_container.subheader("📈 Backtest")
        r2 = get_request(f"http://localhost:8000/api/backtest/{ticker}")
        backtest_data = pd.Series(r2.json())
        backtest_container.line_chart(backtest_data)
        backtest_container.success(f"Final portfolio value: ${backtest_data.iloc[-1]:,.2f}")

        # === Monte Carlo Simulation ===
        mc_container.subheader("🎲 Monte Carlo Simulation")
        r3 = get_request(f"http://localhost:8000/api/monte-carlo-sim/{ticker}")
        sim_data = np.array(r3.json())
        df_sim = pd.DataFrame(sim_data)

        fig = go.Figure()
        for col in df_sim.columns:
            fig.add_trace(go.Scatter(x=df_sim.index, y=df_sim[col], mode='lines', line=dict(width=1), showlegend=False))

        fig.update_layout(
            title="Monte Carlo Simulation of Portfolio Performance",
            xaxis_title="Days",
            yaxis_title="Simulated Portfolio Value ($)",
            template="plotly_white"
        )
        mc_container.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"One or more analyses failed: {e}")
