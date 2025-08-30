import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.data.downloader import get_sp500_tickers, download_prices


def _equal_weight_monthly(prices: pd.DataFrame):
    px = prices.resample("ME").last()
    rets = px.pct_change()
    ew_rets = rets.mean(axis=1, skipna=True)
    return ew_rets


def _cumulative_equity(returns: pd.Series):
    return (1 + returns.fillna(0)).cumprod()


def _cagr(returns, periods_per_year=12):
    growth = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return growth ** (1 / years) - 1


def _annual_vol(returns, periods_per_year=12):
    return returns.std() * np.sqrt(periods_per_year)


def _sharpe(returns, rf=0, periods_per_year=12):
    excess = returns.mean() * periods_per_year - rf
    vol = _annual_vol(returns, periods_per_year)
    return np.nan if vol == 0 else excess / vol


def _max_drawdown(equity):
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return dd.min()


def survivorship_bias_summary_plot() -> tuple[pd.DataFrame, plt.Figure]:
    # Get tickers
    tickers: list[str] = get_sp500_tickers()

    # Get prices
    prices = download_prices(tickers, start="2000-01-01")

    # Get SPY prices for comparison
    spy = download_prices(["SPY"], start="2000-01-01")["SPY"]

    # Monthly aligned
    ew_rets = _equal_weight_monthly(prices)
    spy_rets = spy.resample("M").last().pct_change()

    # Equity curves
    ew_eq = _cumulative_equity(ew_rets)
    spy_eq = _cumulative_equity(spy_rets)

    # Metrics
    summary = {
        "Portfolio (Current)": "Current S&P500 (Equal Weight)",
        "CAGR (Current)": _cagr(ew_rets),
        "Vol (Current)": _annual_vol(ew_rets),
        "Sharpe (Current)": _sharpe(ew_rets),
        "Max Drawdown (Current)": _max_drawdown(ew_eq),
        "Portfolio": "SPY",
        "CAGR": _cagr(spy_rets),
        "Vol": _annual_vol(spy_rets),
        "Sharpe": _sharpe(spy_rets),
        "Max Drawdown": _max_drawdown(spy_eq)
    }

    print("\n=== Summary Metrics ===")
    print(summary)

    # Get figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ew_eq.index, ew_eq, label="Current S&P500 (EW)")
    ax.plot(spy_eq.index, spy_eq, label="SPY")
    ax.set_title("Survivorship Bias Demo: $1 Growth (2000-Present)")
    ax.set_ylabel("Cumulative Growth")
    ax.legend()

    return summary, fig
