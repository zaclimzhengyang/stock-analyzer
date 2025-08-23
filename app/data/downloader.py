from datetime import datetime
from typing import Union, Optional
import numpy as np
import requests
import yfinance as yf
import pandas as pd

from constants import treasury_fred_series
from pandas_datareader import data as pdr


def get_price_data(ticker, start_date="2025-01-01", end_date="2025-06-01"):
    """
    Download historical price data for a given ticker using yfinance.

    Returns a DataFrame with OHLCV data.
    """
    data = yf.download(tickers=[ticker], start=start_date, end=end_date)
    return data


def get_mean_returns_cov_matrix(
    stocks: Union[list[str], str], start: datetime, end: datetime
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculate mean daily returns and covariance matrix for given stocks.

    Financial Description:
    - Mean returns: Average daily return for each stock.
    - Covariance matrix: Measures how returns of stocks move together.
    """
    stock_data = yf.download(stocks, start, end)["Close"]
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


def get_fundamentals(ticker: str, price_data: pd.DataFrame):
    """
    Retrieve fundamental and risk metrics for a given ticker.

    Returns a dictionary with:
    - Valuation ratios (P/E, P/B)
    - Profitability (ROE)
    - Market capitalization
    - Risk/return metrics (Sharpe ratio, max drawdown, annualized return/volatility)

    Financial Description:
    - Sharpe Ratio: Risk-adjusted return.
    - Max Drawdown: Largest observed loss from peak.
    """
    info = yf.Ticker(ticker).info
    price_to_earning = float(info.get("forwardPE")) if info.get("forwardPE") else None
    price_to_book = float(info.get("priceToBook")) if info.get("priceToBook") else None
    return_on_equity = (
        float(info.get("returnOnEquity")) if info.get("returnOnEquity") else None
    )
    market_capitalization = (
        float(info.get("marketCap")) if info.get("marketCap") else None
    )

    daily_returns = price_data["Close"].pct_change()

    mean_daily_return = daily_returns.mean().iloc[0]
    std_daily_return = daily_returns.std().iloc[0]

    annualized_return = mean_daily_return * 252
    annualized_volatility = std_daily_return * np.sqrt(252)

    risk_free_rate = 0.045
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    rolling_max = price_data["Close"].cummax()
    drawdown = price_data["Close"] / rolling_max - 1
    max_drawdown = float(drawdown.min().min())

    return {
        "Price-to-Earning Ratio": price_to_earning,
        "Price-to-Book Ratio": price_to_book,
        "Return on Equity": return_on_equity,
        "Market Capitalization": market_capitalization,
        "Mean Daily Return": mean_daily_return,
        "Std Dev Daily Return": std_daily_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }


def fetch_nasdaq_companies(limit: int = 100) -> pd.DataFrame:
    """
    Fetch a list of NASDAQ-listed companies and their market capitalizations.

    Returns a DataFrame sorted by market cap, limited to the top N companies.
    """
    url = "https://api.nasdaq.com/api/screener/stocks"
    params = {
        "tableonly": "true",
        "limit": 5000,
        "offset": 0,
        "exchange": "nasdaq",
        "download": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    resp_data = resp.json()["data"]["rows"]

    df = pd.DataFrame(resp_data)
    df["marketCap"] = pd.to_numeric(
        df["marketCap"].str.replace(r"[\$,]", "", regex=True), errors="coerce"
    )
    df = df.dropna(subset=["marketCap"])
    df = df.sort_values("marketCap", ascending=False).head(limit)
    return df


# 3. Download yield data for current and historical dates
def get_treasury_yield_curve(date):
    yields = {}

    for label, code in treasury_fred_series.items():
        try:
            val = pdr.DataReader(code, "fred", date, date).iloc[0, 0]
            yields[label] = val
        except Exception:
            yields[label] = None
    return yields


def get_stock_data(ticker: str):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    if not stock.info:
        raise ValueError(f"Ticker {ticker} not found or no data available.")
    return stock


def download_data(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data
