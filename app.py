import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from matplotlib import pyplot as plt

# === Import analysis functions ===
from app.data.downloader import get_price_data, get_fundamentals
from app.dca_etf.dca_etf import dcf_etf_main
from app.factors.momentum import generate_momentum_scores, generate_signals
from app.backtest.backtest import get_backtest
from app.back_trader.analyzer import backtrader_analyze
from app.back_trader.models import RunSettings
from app.black_scholes.black_scholes_option_pricer import bsop
from app.monte_carlo.gbm_simulation import mc_simulation_gbm
from app.prediction.predictor import scan_top_nasdaq
from app.probability_density_function.pdf import pdf
from app.survivorship_bias.survivorship_bias import survivorship_bias_summary_plot
from app.trading_strategies.pair_trading import pair_trading_strategy


# === Import trading strategies ===
def run_gbm_monte_carlo(ticker: str, start_date: str, end_date: str):
    """
    Helper to run the GBM Monte Carlo simulation and display it in Streamlit.
    """
    st.subheader(f"🎲 GBM Monte Carlo Simulation — {ticker}")

    st.markdown("""
        **What it is:**  
        GBM Monte Carlo simulation estimates the distribution of possible future portfolio values
        by generating many random price paths based on historical mean returns and covariance.

        **Approach:**  
        - Retrieve historical mean returns and covariance matrix for the selected ticker.  
        - Simulate hundreds of random price paths over a given horizon.  
        - Calculate **5% Value at Risk (VaR)** and **5% Conditional VaR (CVaR)** from the simulated outcomes.  

        **Why it matters:**  
        - Quantifies downside risk in dollar terms.  
        - Shows the range of possible portfolio outcomes, helping investors understand tail risk.  
    """)
    # --- Run the simulation (this returns the Plotly figure and metrics) ---
    result = mc_simulation_gbm(ticker, start_date, end_date)

    # --- Display the figure inside Streamlit ---
    st.plotly_chart(result["stock_price_fig"], use_container_width=True)
    st.plotly_chart(result["stock_price_histogram"], use_container_width=True)
    st.plotly_chart(result["portfolio_fig"], use_container_width=True)
    st.plotly_chart(result["portfolio_histogram"], use_container_width=True)

    # --- Show the risk metrics ---
    st.success(f"5% VaR: ${result['VaR_5']:,.2f}")
    st.success(f"5% CVaR: ${result['CVaR_5']:,.2f}")


def run_pair_trading():
    st.subheader("🔗 Pairs Trading Strategy (JKHY vs LDOS)")

    st.markdown("""
        **What it is:**  
        Pairs trading is a market-neutral strategy widely used in quantitative finance.  
        It identifies two stocks that are historically **cointegrated** and takes long/short positions 
        when their spread deviates from the historical equilibrium.

        **Approach:**  
        - Fetch stock data for the sector and identify cointegrated pairs.  
        - Compute the spread and z-score of the ratio.  
        - Generate trading signals based on deviations from the mean.  
        - Simulate trades and backtest the portfolio.

        **Why it matters:**  
        - Provides a market-neutral way to potentially profit from mean-reverting relationships.  
        - Helps understand statistical arbitrage concepts like regression-based hedge ratios and signal generation.  
        - Visualizes the PnL curve and trade markers to show how the strategy would perform over time.
        """)

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

    dcf_etf_main()

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
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
start_date = st.sidebar.text_input("Start Date", "2020-01-01")
end_date = st.sidebar.text_input("End Date", "2025-06-01")

# --- Sidebar menu for ticker-specific analyses ---
st.sidebar.header("Ticker Analyses")
ticker_features = ["Fundamentals", "GBM Monte Carlo Simulation"]

# Initialize session_state if not exists
if "ticker_analysis" not in st.session_state:
    st.session_state.ticker_analysis = None
if "other_analysis" not in st.session_state:
    st.session_state.other_analysis = None


def select_ticker_analysis():
    # When a ticker feature is selected, clear other_analysis
    st.session_state.other_analysis = None


def select_other_analysis():
    # When other feature is selected, clear ticker_analysis
    st.session_state.ticker_analysis = None


selected_ticker_feature = st.sidebar.radio(
    "Choose analysis:",
    [None] + ticker_features,
    index=0,
    format_func=lambda x: "Select..." if x is None else x,
    key="ticker_analysis",
    on_change=select_ticker_analysis
)

# --- Sidebar menu for other analyses ---
st.sidebar.header("Other Analyses")
other_features = ["Top 10 performing ETF DCA Backtest (2020-2025)", "Pairs Trading (JKHY vs LDOS)"]

selected_other_feature = st.sidebar.radio(
    "Choose other analysis:",
    [None] + other_features,
    index=0,
    format_func=lambda x: "Select..." if x is None else x,
    key="other_analysis",
    on_change=select_other_analysis
)

# --- Run ticker-specific analysis ---
if st.session_state.ticker_analysis:
    if st.session_state.ticker_analysis == "Fundamentals":
        st.subheader("📊 Fundamentals")
        df = fundamentals_analysis(ticker, start_date, end_date)
        st.table(df)
    elif st.session_state.ticker_analysis == "GBM Monte Carlo Simulation":
        run_gbm_monte_carlo(ticker, start_date, end_date)

# --- Run other analyses ---
elif st.session_state.other_analysis:
    if st.session_state.other_analysis == "Top 10 performing ETF DCA Backtest (2020-2025)":
        figs = run_etf_dca()
        for fig in figs:
            st.pyplot(fig)
    elif st.session_state.other_analysis == "Pairs Trading (JKHY vs LDOS)":
        pair_trading_container = st.container()
        with pair_trading_container:
            run_pair_trading()
#
# # === Run selected feature ===
# if feature_selected:
#     st.info(f"Running: {feature_selected}")
#
#     try:
#         if feature_selected == "PDF Analysis":
#             container = st.container()
#             container.subheader("📊 Probability Density Function Analysis")
#             fig_pdf, stats = pdf(ticker, start_date, end_date)
#             container.pyplot(fig_pdf)
#             container.table(pd.DataFrame.from_dict(stats, orient="index", columns=["Value"]))
#
#         elif feature_selected == "Backtrader Backtest":
#             container = st.container()
#             container.subheader("📈 Backtrader Backtest")
#             settings = RunSettings(
#                 tickers=[ticker],
#                 start="2020-01-01",
#                 end="2025-01-01",
#                 cash=100000,
#                 commission=0.0005,
#                 slippage=0.0002,
#                 fast=50,
#                 slow=200,
#                 rsi_buy=30,
#                 rsi_sell=70,
#                 out="results",
#             )
#             analysis = backtrader_analyze(settings)
#             bt_df = pd.DataFrame(analysis).T
#             bt_df.index.name = "Ticker"
#             container.table(bt_df)
#
#         elif feature_selected == "Black Scholes":
#             container = st.container()
#             container.subheader("⚖️ Black Scholes Model")
#             bsm_df = bsop(ticker)[["contractSymbol", "bsmValuation", "delta", "gamma", "vega", "theta", "rho"]]
#             container.table(bsm_df)
#
#         elif feature_selected == "Momentum Backtest":
#             container = st.container()
#             container.subheader("📈 Momentum Backtest")
#             extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=500)).strftime("%Y-%m-%d")
#             close_prices = get_price_data(ticker, extended_start_date, end_date)[("Adj Close", ticker)]
#             signals = generate_signals(close_prices)
#             backtest_result = get_backtest(close_prices, signals)
#             container.line_chart(pd.Series(backtest_result))
#             container.success(f"Final portfolio value: ${pd.Series(backtest_result).iloc[-1]:,.2f}")
#
#
#         elif feature_selected == "Survivorship Bias":
#             container = st.container()
#             container.subheader("🏦 Survivorship Bias: S&P500 vs SPY")
#             summary, fig_surv = survivorship_bias_summary_plot()
#             st.pyplot(fig_surv)
#             df = pd.DataFrame(
#                 [
#                     {
#                         "Portfolio": summary["Portfolio (Current)"],
#                         "CAGR": summary["CAGR (Current)"],
#                         "Vol": summary["Vol (Current)"],
#                         "Sharpe": summary["Sharpe (Current)"],
#                         "Max Drawdown": summary["Max Drawdown (Current)"],
#                     },
#                     {
#                         "Portfolio": summary["Portfolio"],
#                         "CAGR": summary["CAGR"],
#                         "Vol": summary["Vol"],
#                         "Sharpe": summary["Sharpe"],
#                         "Max Drawdown": summary["Max Drawdown"],
#                     },
#                 ]
#             ).set_index("Portfolio")
#             container.table(df)
#
#         elif feature_selected == "NASDAQ Buy Recommendations":
#             container = st.container()
#             container.subheader("NASDAQ Buy Recommendations")
#             df = scan_top_nasdaq()
#             container.table(df)
#
#
#     except Exception as e:
#         st.error(f"Failed to run {feature_selected}: {e}")
