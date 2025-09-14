from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


def get_mean_returns_cov_matrix(stocks: Union[list[str], str],
                                start: str,
                                end: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculate mean daily returns and covariance matrix for given stocks.

    Financial Description:
    - Mean returns: Average daily return for each stock.
    - Covariance matrix: Measures how returns of stocks move together.

    example:
    - cov_matrix of 0.000426 is the daily variance of AAPL returns in decimal form.
    - Take its square root to say:
    - “AAPL’s daily returns fluctuate about ±2% (1-σ) around the mean.”
    """
    stock_data = yf.download(tickers=stocks, start=start, end=end, auto_adjust=False)["Adj Close"]
    returns = stock_data.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


def mc_var(returns: pd.Series, alpha=5) -> float:
    """
    Calculate Value at Risk (VaR) at the given percentile.

    Financial Description:
    - VaR: The maximum expected loss at a given confidence level (e.g., 5%).
    """
    return np.percentile(returns, alpha)


def mc_cvar(returns: pd.Series, alpha=5):
    """
    Calculate Conditional Value at Risk (CVaR) at the given percentile.

    Financial Description:
    - CVaR: The expected loss in the worst-case (tail) scenarios beyond the VaR threshold.
    """
    var = mc_var(returns, alpha=alpha)
    return returns[returns <= var].mean()


def mc_simulation_gbm(ticker: str,
                      start_date: str,
                      end_date: str,
                      runs: int = 100,
                      horizon: int = 100,
                      initial_value: float = 1_000.0) -> dict[str, Union[float, list, go.Figure]]:
    """
    Monte-Carlo GBM simulation.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    start_date, end_date : str
        Historical data window in 'YYYY-MM-DD' format.
    runs : int, optional
        Number of Monte-Carlo simulation runs (default 100).
    horizon : int, optional
        Number of discrete time steps in each simulation (default 100).
    initial_value : float, optional
        Starting portfolio value in dollars (default 1000).

    Returns
    -------
    Dict[str, Union[float, List, go.Figure]]
        {
            "simulations": list of portfolio value paths,
            "VaR_5": 5% Value at Risk,
            "CVaR_5": 5% Conditional VaR,
            "figure": Plotly Figure of all simulated paths
        }
    """
    stock_data = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]
    daily_returns = stock_data.pct_change().dropna()

    mu = daily_returns.mean().item()
    sigma = daily_returns.std().item()

    delta_t = 1.0
    S0 = float(stock_data.iloc[-1])

    # --- simulate all runs at once ---
    Z = np.random.normal(size=(horizon - 1, runs))  # 99 x 100

    # Discrete GBM formula
    increments = (mu - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * Z
    log_paths = np.cumsum(increments, axis=0)
    log_paths = np.vstack([np.zeros(runs), log_paths])  # 100 x 100
    prices = S0 * np.exp(log_paths)  # 100 x 100

    # Scale to initial portfolio value
    portfolio_sims = prices / S0 * initial_value

    port_results = pd.Series(portfolio_sims[-1, :])
    var = initial_value - mc_var(port_results, alpha=5)
    cvar = initial_value - mc_cvar(port_results, alpha=5)

    df_sim = pd.DataFrame(portfolio_sims)
    portfolio_fig = go.Figure([
        go.Scatter(x=df_sim.index, y=df_sim[col], mode="lines",
                   line=dict(width=1), showlegend=False)
        for col in df_sim.columns
    ])
    portfolio_fig.update_layout(
        title=f"GBM Monte Carlo Simulation for {ticker} ({runs} runs)",
        xaxis_title="Time Steps",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )

    # Histogram of final simulated portfolio values
    portfolio_hist = go.Figure()
    portfolio_hist.add_trace(
        go.Histogram(
            x=port_results,
            nbinsx=30,  # adjust number of bins
            marker_color='steelblue',
            opacity=0.75
        )
    )
    portfolio_hist.update_layout(
        title=f"Distribution of Final Portfolio Values after {horizon} Steps",
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    # --- Stock price line graph (all paths) ---
    price_df = pd.DataFrame(prices)
    stock_price_fig = go.Figure([
        go.Scatter(x=price_df.index, y=price_df[col], mode="lines",
                   line=dict(width=1), showlegend=False)
        for col in price_df.columns
    ])
    stock_price_fig.update_layout(
        title=f"GBM Monte Carlo Simulation – Stock Price Paths for {ticker}",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price ($)",
        template="plotly_white",
    )

    # --- Histogram of final simulated stock prices ---
    final_stock_prices = pd.Series(prices[-1, :])
    stock_price_hist = go.Figure()
    stock_price_hist.add_trace(
        go.Histogram(
            x=final_stock_prices,
            nbinsx=30,
            marker_color='darkorange',
            opacity=0.75
        )
    )
    stock_price_hist.update_layout(
        title=f"Distribution of Final Stock Prices after {horizon} Steps",
        xaxis_title="Final Stock Price ($)",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    return {
        "simulations": portfolio_sims.tolist(),
        "VaR_5": round(var, 2),
        "CVaR_5": round(cvar, 2),
        "portfolio_fig": portfolio_fig,
        "portfolio_histogram": portfolio_hist,
        "stock_price_fig": stock_price_fig,  # NEW
        "stock_price_histogram": stock_price_hist  # NEW
    }


if __name__ == "__main__":
    ticker = "AAPL"
    end_date = "2025-06-01"
    start_date = "2020-01-01"
    result = mc_simulation_gbm(ticker, start_date, end_date)
    print(f"5% VaR: ${result['VaR_5']:,.2f}")
    print(f"5% CVaR: ${result['CVaR_5']:,.2f}")

    result["stock_price_fig"].show()
    result["stock_price_histogram"].show()
    result["portfolio_fig"].show()
    result["portfolio_histogram"].show()
