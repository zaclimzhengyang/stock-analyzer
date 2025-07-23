import yfinance as yf
import pandas as pd


def get_price_data(ticker, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    return data


def get_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "pe": float(info.get("forwardPE")) if info.get("forwardPE") else None,
        "pb": float(info.get("priceToBook")) if info.get("priceToBook") else None,
        "roe": (
            float(info.get("returnOnEquity")) if info.get("returnOnEquity") else None
        ),
        "market_cap": float(info.get("marketCap")) if info.get("marketCap") else None,
    }
