import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime

# === Import your analysis functions directly ===
from app.data.downloader import get_price_data, get_fundamentals
from app.factors.momentum import generate_momentum_score, generate_signals
from app.backtest.backtest import get_backtest
from app.back_trader.analyzer import backtrader_analyze
from app.back_trader.models import RunSettings
from app.black_scholes.black_scholes_option_pricer import bsop
from app.monte_carlo.simulation import mc_simulation
from app.prediction.predictor import StockPredictor, scan_top_nasdaq
from app.probability_density_function.pdf import pdf
from app.survivorship_bias.survivorship_bias import survivorship_bias_summary_plot

st.title("📈 Quantitative Stock Analyzer")

with st.form("ticker_form"):
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")
    start_date = st.text_input("Enter start date (e.g., 2025-01-01)", "2025-01-01")
    end_date = st.text_input("Enter end date (e.g., 2025-06-01)", "2025-06-01")
    submitted = st.form_submit_button("Run All Analysis")

if submitted:
    st.info("Running all analyses...")

    # Create containers
    fundamentals_container = st.container()
    pdf_container = st.container()
    backtrader_container = st.container()
    bsm_container = st.container()
    backtest_container = st.container()
    mc_container = st.container()
    survivorship_container = st.container()

    try:
        # === Fundamentals ===
        fundamentals_container.subheader("📊 Fundamentals")
        price_data = get_price_data(ticker, start_date, end_date)
        fundamentals = get_fundamentals(ticker, price_data)
        score = generate_momentum_score(price_data)

        fundamentals_data = {
            "Ticker": ticker,
            "Momentum Score": float(score) if score is not None else None,
            **fundamentals,
        }
        fundamentals_df = pd.DataFrame.from_dict(
            fundamentals_data, orient="index", columns=["Value"]
        )
        fundamentals_df.index.name = "Metric"
        fundamentals_container.table(fundamentals_df)

        # === PDF Analysis ===
        pdf_container.subheader("📊 Probability Density Function Analysis")
        fig_pdf, stats = pdf(ticker, start_date, end_date)
        pdf_df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])
        pdf_df.index.name = "Metric"
        pdf_container.table(pdf_df)
        pdf_container.pyplot(fig_pdf)

        # === Backtrader ===
        backtrader_container.subheader("📈 Backtrader Backtest (2020-01-01 → 2025-01-01)")
        settings = RunSettings(
            tickers=[ticker],
            start="2020-01-01",
            end="2025-01-01",
            cash=100000,
            commission=0.0005,
            slippage=0.0002,
            fast=50,
            slow=200,
            rsi_buy=30,
            rsi_sell=70,
            out="results",
        )
        analysis = backtrader_analyze(settings)
        bt_df = pd.DataFrame(analysis).T
        bt_df.index.name = "Ticker"
        backtrader_container.table(bt_df)
        for strategy in ["Momentum", "MeanRev"]:
            final_val = bt_df[f"{strategy}_FinalValue"].iloc[0]
            cagr = bt_df[f"{strategy}_CAGR"].iloc[0]
            backtrader_container.success(
                f"{strategy} → Final Value: ${final_val:,.2f}, CAGR: {cagr:.2%}"
            )

        # === Black Scholes ===
        bsm_container.subheader("⚖️ Black Scholes Model")
        bsm = bsop(ticker)
        bsm_df = bsm[
            ["contractSymbol", "bsmValuation", "delta", "gamma", "vega", "theta", "rho"]
        ]
        bsm_container.table(bsm_df)

        # === Backtest ===
        backtest_container.subheader("📈 Backtest (Momentum Strategy)")
        close_prices = price_data[("Adj Close", ticker)]
        signals = generate_signals(close_prices)
        backtest_result = get_backtest(close_prices, signals)
        backtest_series = pd.Series(backtest_result)
        backtest_container.line_chart(backtest_series)
        backtest_container.success(
            f"Final portfolio value: ${backtest_series.iloc[-1]:,.2f}"
        )

        # === Monte Carlo Simulation ===
        mc_container.subheader("🎲 Monte Carlo Simulation")
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        sim_result = mc_simulation(ticker, start_date_dt, end_date_dt)
        df_sim = pd.DataFrame(sim_result["simulations"])
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
        mc_container.plotly_chart(fig, use_container_width=True)
        mc_container.success(f"5% VaR: ${sim_result['VaR_5']:,.2f}")
        mc_container.success(f"5% CVaR: ${sim_result['CVaR_5']:,.2f}")

        # === Survivorship Bias ===
        survivorship_container.subheader("🏦 Survivorship Bias: S&P500 vs SPY")
        summary, fig_surv = survivorship_bias_summary_plot()
        df = pd.DataFrame(
            [
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
        ).set_index("Portfolio")
        st.table(df)
        st.pyplot(fig_surv)

    except Exception as e:
        st.error(f"One or more analyses failed: {e}")

# === NASDAQ Buy Recs ===
st.title("NASDAQ Buy Recommendations")
if st.button("Load Buy Recommendations"):
    try:
        data = scan_top_nasdaq().to_dict(orient="records")
        if data:
            df = pd.DataFrame(data).rename(
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
