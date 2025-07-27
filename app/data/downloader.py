from datetime import datetime
from typing import Union

import requests
import yfinance as yf
import pandas as pd


def get_price_data(ticker, period="1y", interval="1d"):
    data: pd.DataFrame = yf.download(ticker, period=period, interval=interval)
    return data

def get_mean_returns_cov_matrix(stocks: Union[list[str], str], start: datetime, end: datetime) -> (pd.Series, pd.DataFrame):
    stock_data: pd.DataFrame = yf.download(stocks, start, end)
    stock_data: pd.DataFrame = stock_data['Close']
    returns: pd.DataFrame = stock_data.pct_change()
    mean_returns: pd.Series = returns.mean()
    cov_matrix: pd.DataFrame = returns.cov()
    return mean_returns, cov_matrix

def get_fundamentals(ticker):
    info: dict[str, float] = yf.Ticker(ticker).info
    return {
        "pe": float(info.get("forwardPE")) if info.get("forwardPE") else None,
        "pb": float(info.get("priceToBook")) if info.get("priceToBook") else None,
        "roe": (
            float(info.get("returnOnEquity")) if info.get("returnOnEquity") else None
        ),
        "market_cap": float(info.get("marketCap")) if info.get("marketCap") else None,
    }


def fetch_nasdaq_companies(limit: int = 100) -> pd.DataFrame:
    url = "https://api.nasdaq.com/api/screener/stocks"
    params = {
        "tableonly": "true",
        "limit": 5000,
        "offset": 0,
        "exchange": "nasdaq",
        "download": "true"
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    resp_data = resp.json()["data"]["rows"]

    df = pd.DataFrame(resp_data)
    df["marketCap"] = pd.to_numeric(df["marketCap"].str.replace(r"[\$,]", "", regex=True), errors="coerce")
    df = df.dropna(subset=["marketCap"])
    df = df.sort_values("marketCap", ascending=False).head(limit)
    return df
