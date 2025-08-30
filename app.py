import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt


def get_request(url: str) -> requests.Response:
    r = requests.get(url)
    r.raise_for_status()
    return r


st.title("📈 Quantitative Stock Analyzer")

with st.form("ticker_form"):
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")
    start_date = st.text_input("Enter start date (e.g., 2025-01-01)", "2025-01-01")
    end_date = st.text_input("Enter end date (e.g., 2025-06-01)", "2025-06-01")
    submitted = st.form_submit_button("Run All Analysis")

if submitted:
    st.info("Running all analyses...")

    # Create placeholders so content can be shown side-by-side or section-by-section
    fundamentals_container = st.container()
    pdf_container = st.container()
    backtrader_container = st.container()
    bsm_container = st.container()
    backtest_container = st.container()
    mc_container = st.container()

    try:
        # === Fundamental Data ===
        fundamentals_container.subheader("📊 Fundamentals")
        r1 = get_request(
            f"http://localhost:8000/api/analyze/{ticker}?start_date={start_date}&end_date={end_date}"
        )
        fundamentals_data = r1.json()

        # Convert dict to single-column DataFrame
        fundamentals_df = pd.DataFrame.from_dict(
            fundamentals_data, orient="index", columns=["Value"]
        )
        fundamentals_df.index.name = "Metric"

        fundamentals_container.table(fundamentals_df)

        # === PDF Analysis ===
        pdf_container.subheader("📊 Probability Density Function Analysis")
        r_pdf = get_request(
            f"http://localhost:8000/api/pdf/{ticker}?start_date={start_date}&end_date={end_date}"
        )
        pdf_data = r_pdf.json()

        # Display stats in table
        pdf_df = pd.DataFrame.from_dict(pdf_data["stats"], orient="index", columns=["Value"])
        pdf_df.index.name = "Metric"
        pdf_container.table(pdf_df)

        # Display plot directly (decode base64 to image)
        import base64
        from io import BytesIO
        from PIL import Image

        if "plot" in pdf_data:
            img_bytes = base64.b64decode(pdf_data["plot"])
            img = Image.open(BytesIO(img_bytes))
            pdf_container.image(img, caption="PDF Plot")

        # === Backtrader Backtest Results ===
        backtrader_container.subheader("📈 Backtrader Backtest (1st jan 2020 - 1st jan 2025)")
        bt = get_request(
            # f"http://localhost:8000/api/backtrader/{ticker}?start_date={start_date}&end_date={end_date}"
            f"http://localhost:8000/api/backtrader/{ticker}?start_date=2020-01-01&end_date=2025-01-01"
        )
        backtest_data = bt.json()

        # Convert nested dictionary to DataFrame for display
        # Example structure: {'AAPL': {'Momentum_FinalValue': ..., 'Momentum_CAGR': ...}}
        bt_df = pd.DataFrame(backtest_data).T  # transpose to have strategies as columns
        bt_df.index.name = "Ticker"
        backtrader_container.table(bt_df)

        # Optionally display final portfolio value and CAGR for Momentum and MeanReversion
        for strategy in ["Momentum", "MeanRev"]:
            final_val = bt_df[f"{strategy}_FinalValue"].iloc[0]
            cagr = bt_df[f"{strategy}_CAGR"].iloc[0]
            backtrader_container.success(
                f"{strategy} Strategy → Final Portfolio Value: ${final_val:,.2f}, CAGR: {cagr:.2%}"
            )

        # === Black Scholes Model ===
        bsm_container.subheader("Black Scholes Model")
        bsm_r = get_request(f"http://localhost:8000/api/black-scholes-model/{ticker}")
        bsm_data = bsm_r.json()

        bsm_df = pd.DataFrame(bsm_data)
        bsm_container.table(bsm_df)

        # === Backtest Data ===
        backtest_container = st.container()
        backtest_container.subheader("📈 Backtest")
        r2 = get_request(
            f"http://localhost:8000/api/backtest/{ticker}?start_date={start_date}&end_date={end_date}"
        )
        backtest_data = pd.Series(r2.json())
        backtest_container.line_chart(backtest_data)
        backtest_container.success(
            f"Final portfolio value: ${backtest_data.iloc[-1]:,.2f}"
        )

        # === Monte Carlo Simulation ===
        mc_container.subheader("🎲 Monte Carlo Simulation")
        r3 = get_request(
            f"http://localhost:8000/api/monte-carlo-sim/{ticker}?start_date={start_date}&end_date={end_date}"
        )
        sim_result = r3.json()
        sim_data = np.array(sim_result["simulations"])

        df_sim = pd.DataFrame(sim_data)

        fig = go.Figure()
        for col in df_sim.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sim.index,
                    y=df_sim[col],
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="Monte Carlo Simulation of Portfolio Performance",
            xaxis_title="Days",
            yaxis_title="Simulated Portfolio Value ($)",
            template="plotly_white",
        )
        mc_container.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
        <div style="font-size:22px; font-weight:600; margin-bottom:4px;">5% VaR</div>
        <div style="font-size:18px; color:crimson;">${sim_result['VaR_5']:,.2f}</div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div style="font-size:22px; font-weight:600; margin-bottom:4px;">5% CVaR</div>
        <div style="font-size:18px; color:crimson;">${sim_result['CVaR_5']:,.2f}</div>
        """,
            unsafe_allow_html=True,
        )

        # === Drawdown ===
        mc_container.subheader("📉 Drawdown")

        r4 = get_request(
            f"http://localhost:8000/api/drawdown/{ticker}?start_date={start_date}&end_date={end_date}"
        )
        drawdown_result = r4.json()

        drawdown_dates = pd.to_datetime(drawdown_result["dates"])
        drawdown_values = np.array(drawdown_result["values"])

        # Matplotlib plot
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(drawdown_dates, drawdown_values, color="red", label="Drawdown")
        ax.fill_between(drawdown_dates, drawdown_values, 0, color="red", alpha=0.3)
        ax.set_title(f"Drawdown for {ticker}")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()

        mc_container.pyplot(fig)

        # Show max drawdown in text
        mc_container.success(f"Max Drawdown: {drawdown_result['max_drawdown']:.2%}")

        # === Survivorship Bias ===
        survivorship_container = st.container()
        survivorship_container.subheader("Survivorship Bias: S&P500 vs SPY")
        survivorship_r = get_request("http://localhost:8000/api/survivorship-bias/sp500")
        survivorship_data = survivorship_r.json()

        # Display summary metrics
        summary = survivorship_data["summary"]

        rows = [
            {
                "Portfolio": summary["Portfolio (Current)"],
                "CAGR": summary["CAGR (Current)"],
                "Vol": summary["Vol (Current)"],
                "Sharpe": summary["Sharpe (Current)"],
                "Max Drawdown": summary["Max Drawdown (Current)"],
            },
            {
                "Portfolio": summary["Portfolio"],
                "CAGR": summary["CAGR"],
                "Vol": summary["Vol"],
                "Sharpe": summary["Sharpe"],
                "Max Drawdown": summary["Max Drawdown"],
            },
        ]

        df = pd.DataFrame(rows)
        df = df.set_index("Portfolio")
        st.table(df)

        # Display plot
        img_bytes = base64.b64decode(survivorship_data["plot"])
        img = Image.open(BytesIO(img_bytes))
        st.image(img, caption="Survivorship Bias Demo: $1 Growth (2000-Present)")

    except Exception as e:
        st.error(f"One or more analyses failed: {e}")

st.title("NASDAQ Buy Recommendations")

if st.button("Load Buy Recommendations"):
    try:
        url = "http://localhost:8000/api/nasdaq-buy-recs"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data:
            df = pd.DataFrame(data)
            df = df.rename(
                columns={
                    "ticker": "Ticker",
                    "marketCap": "Market Cap",
                    "recommendation": "Recommendation",
                }
            )

            st.table(df)
        else:
            st.write("No buy recommendations found.")
    except Exception as e:
        st.error(f"Failed to fetch buy recommendations: {e}")
