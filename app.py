import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

from matplotlib import pyplot as plt

# === Import analysis functions ===
from app.data.downloader import get_price_data, get_fundamentals
from app.dca_etf.dca_etf import dca_etf
from app.factors.momentum import generate_momentum_scores, generate_signals
from app.backtest.backtest import get_backtest
from app.back_trader.analyzer import backtrader_analyze
from app.back_trader.models import RunSettings
from app.black_scholes.black_scholes_option_pricer import bsop
from app.monte_carlo.simulation import mc_simulation
from app.prediction.predictor import scan_top_nasdaq
from app.probability_density_function.pdf import pdf
from app.survivorship_bias.survivorship_bias import survivorship_bias_summary_plot
from app.trading_strategies.pair_trading import pair_trading_strategy


# === Import trading strategies ===
def run_pair_trading():
    st.subheader("🔗 Pair Trading Strategy (JKHY vs LDOS)")

    pair_trading_strategy()

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        st.pyplot(fig)


def run_etf_dca():
    st.subheader("💰 Top 10 Performing ETF DCA Backtest (2020-2025)")
    # --- Summary Section ---
    st.markdown("""
        ### 📘 Overview
        **Problem Statement:**  
        Investors often wonder which ETFs provide the best long-term growth if they invest regularly, rather than trying to time the market.  
        This backtest applies a **Dollar-Cost Averaging (DCA)** strategy across a broad set of ETFs to identify the top performers.  

        **Approach:**  
        1. Collect all ETFs from NYSE Arca listings.  
        2. Simulate monthly investments of $100 into each ETF between **2020–2025**.  
        3. Calculate portfolio value growth, returns, and CAGR (Compound Annual Growth Rate).  
        4. Rank ETFs and display the **Top 10 performers**.  

        **Why It Matters:**  
        - DCA removes the risk of timing the market by spreading out investments.  
        - Helps investors discover ETFs with **resilient long-term growth**.  
        - Provides a transparent comparison of **risk-adjusted performance** across ETFs.  
        """)

    dca_etf()

    figs = []
    for fig_num in plt.get_fignums():
        figs.append(plt.figure(fig_num))

    return figs


def fundamentals_analysis(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Run fundamentals + momentum analysis and return as DataFrame."""
    price_data = get_price_data(ticker, start_date, end_date)
    fundamentals = get_fundamentals(ticker, price_data)
    scores = generate_momentum_scores(price_data)

    fundamentals_data = {
        "Ticker": ticker,
        "30-Day Momentum": scores[30] if scores else None,
        "60-Day Momentum": scores[60] if scores else None,
        "90-Day Momentum": scores[90] if scores else None,
        **fundamentals,
    }

    return pd.DataFrame.from_dict(fundamentals_data, orient="index", columns=["Value"])


# === App title ===
st.title("📈 Quantitative Stock Analyzer")

# === Sidebar Inputs ===
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
ticker = ticker.upper().strip()
start_date = st.sidebar.text_input("Start Date", "2025-01-01")
end_date = st.sidebar.text_input("End Date", "2025-06-01")

# === Main Page Visual Menu ===
st.sidebar.header("Select Analysis")
features = [
    "Fundamentals",
    "Top 10 performing ETF DCA Backtest (2020-2025)",
    "Pair Trading",
    # "PDF Analysis",
    # "Backtrader Backtest",
    # "Black Scholes",
    # "Momentum Backtest",
    # "Monte Carlo Simulation",
    # "Survivorship Bias",
    # "NASDAQ Buy Recommendations",
]

# Dictionary to map feature to its button state
feature_selected = None
for feature in features:
    if st.sidebar.button(feature):
        feature_selected = feature

# === Run selected feature ===
if feature_selected:
    st.info(f"Running: {feature_selected}")

    try:
        if feature_selected == "Fundamentals":
            st.subheader("📊 Fundamentals")
            df = fundamentals_analysis(ticker, start_date, end_date)
            st.table(df)

        elif feature_selected == "Top 10 performing ETF DCA Backtest (2020-2025)":
            figs = run_etf_dca()
            for fig in figs:
                st.pyplot(fig)

        elif feature_selected == "PDF Analysis":
            container = st.container()
            container.subheader("📊 Probability Density Function Analysis")
            fig_pdf, stats = pdf(ticker, start_date, end_date)
            container.pyplot(fig_pdf)
            container.table(pd.DataFrame.from_dict(stats, orient="index", columns=["Value"]))

        elif feature_selected == "Backtrader Backtest":
            container = st.container()
            container.subheader("📈 Backtrader Backtest")
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
            container.table(bt_df)

        elif feature_selected == "Black Scholes":
            container = st.container()
            container.subheader("⚖️ Black Scholes Model")
            bsm_df = bsop(ticker)[["contractSymbol", "bsmValuation", "delta", "gamma", "vega", "theta", "rho"]]
            container.table(bsm_df)

        elif feature_selected == "Momentum Backtest":
            container = st.container()
            container.subheader("📈 Momentum Backtest")
            extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=500)).strftime("%Y-%m-%d")
            close_prices = get_price_data(ticker, extended_start_date, end_date)[("Adj Close", ticker)]
            signals = generate_signals(close_prices)
            backtest_result = get_backtest(close_prices, signals)
            container.line_chart(pd.Series(backtest_result))
            container.success(f"Final portfolio value: ${pd.Series(backtest_result).iloc[-1]:,.2f}")

        elif feature_selected == "Monte Carlo Simulation":
            container = st.container()
            container.subheader("🎲 Monte Carlo Simulation")
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            sim_result = mc_simulation(ticker, start_date_dt, end_date_dt)
            df_sim = pd.DataFrame(sim_result["simulations"])
            fig = go.Figure()
            for col in df_sim.columns:
                fig.add_trace(
                    go.Scatter(x=df_sim.index, y=df_sim[col], mode="lines", line=dict(width=1), showlegend=False))
            container.plotly_chart(fig, use_container_width=True)
            container.success(f"5% VaR: ${sim_result['VaR_5']:,.2f}")
            container.success(f"5% CVaR: ${sim_result['CVaR_5']:,.2f}")

        elif feature_selected == "Survivorship Bias":
            container = st.container()
            container.subheader("🏦 Survivorship Bias: S&P500 vs SPY")
            summary, fig_surv = survivorship_bias_summary_plot()
            st.pyplot(fig_surv)
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
            container.table(df)

        elif feature_selected == "NASDAQ Buy Recommendations":
            container = st.container()
            container.subheader("NASDAQ Buy Recommendations")
            df = scan_top_nasdaq()
            container.table(df)

        elif feature_selected == "Pair Trading":
            pair_trading_container = st.container()
            with pair_trading_container:
                run_pair_trading()


    except Exception as e:
        st.error(f"Failed to run {feature_selected}: {e}")
