from datetime import datetime
from typing import Union

import numpy as np
import requests
import yfinance as yf
import pandas as pd


# def get_historical_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
#     prices: pd.DataFrame = yf.download(ticker, start_date, end_date)
#
#     # Daily returns
#     daily_returns = prices['Close'].pct_change()
#
#     # Risk metrics
#     mean_daily_return = daily_returns.mean()
#     std_daily_return = daily_returns.std()
#
#     # Annualized returns & volatility
#     annualized_return = mean_daily_return * 252
#     annualized_volatility = std_daily_return * np.sqrt(252)
#
#     # Sharpe ratio
#     risk_free_rate = 0.045
#     sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
#
#     # Max drawdown
#     rolling_max = prices['Close'].cummax()
#     drawdown = prices['Close'] / rolling_max - 1
#     max_drawdown = drawdown.min()
#
#     metrics = {
#         "Mean Daily Return": mean_daily_return,
#         "Std Dev Daily Return": std_daily_return,
#         "Annualized Return": annualized_return,
#         "Annualized Volatility": annualized_volatility,
#         "Sharpe Ratio": sharpe_ratio,
#         "Max Drawdown": max_drawdown
#     }
#
#     metrics_df = pd.DataFrame(metrics, index=[f'{ticker}'])
#     return metrics_df


# get_historical_prices('AAPL', '2025-01-01', None)


def get_price_data(ticker, start_date="2025-01-01", end_date="2025-06-01"):
    data: pd.DataFrame = yf.download(tickers=[ticker], start=start_date, end=end_date)
    return data


def get_mean_returns_cov_matrix(
    stocks: Union[list[str], str], start: datetime, end: datetime
) -> (pd.Series, pd.DataFrame):
    stock_data: pd.DataFrame = yf.download(stocks, start, end)
    stock_data: pd.DataFrame = stock_data["Close"]
    returns: pd.DataFrame = stock_data.pct_change()
    mean_returns: pd.Series = returns.mean()
    cov_matrix: pd.DataFrame = returns.cov()
    return mean_returns, cov_matrix


def get_fundamentals(ticker: str, price_data: pd.DataFrame):
    info: dict[str, float] = yf.Ticker(ticker).info
    price_to_earning: float = (
        float(info.get("forwardPE")) if info.get("forwardPE") else None
    )
    price_to_book: float = (
        float(info.get("priceToBook")) if info.get("priceToBook") else None
    )
    return_on_equity: float = (
        float(info.get("returnOnEquity")) if info.get("returnOnEquity") else None
    )
    market_capitalization: float = (
        float(info.get("marketCap")) if info.get("marketCap") else None
    )

    # Daily returns
    daily_returns = price_data["Close"].pct_change()

    # Risk metrics
    mean_daily_return = daily_returns.mean()
    mean_daily_return = mean_daily_return.iloc[0]

    std_daily_return = daily_returns.std()
    std_daily_return = std_daily_return.iloc[0]

    # Annualized returns & volatility
    annualized_return = mean_daily_return * 252
    annualized_volatility = std_daily_return * np.sqrt(252)

    # Sharpe ratio
    risk_free_rate = 0.045
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Max drawdown
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
