import math
from datetime import datetime

import numpy as np
from typing import Literal, Tuple

import pandas as pd
import yfinance as yf

from app.data.downloader import get_treasury_yield_curve, get_stock_data
from constants import dates


# -------------------------
# Helper functions
# -------------------------


def _get_spot_price(stock) -> float:
    """Get the latest spot price of the stock"""
    try:
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        raise ValueError(f"Failed to fetch spot price for {stock.ticker}: {e}")


def _standard_normal_pdf(x: float) -> float:
    """Standard normal PDF"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _standard_normal_cdf(x: float) -> float:
    """Standard normal CDF"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S: float, K: float, t: float, r: float, vol: float) -> float:
    numerator = math.log(S / K) + (r + pow(vol, 2) / 2) * t
    denominator = vol * math.sqrt(t)
    return numerator / denominator


def _d2(S: float, K: float, t: float, r: float, vol: float) -> float:
    return _d1(S, K, t, r, vol) - vol * math.sqrt(t)


# -------------------------
# Black-Scholes Pricing
# -------------------------

def bs_call_price(S: float, K: float, t: float, r: float, vol: float) -> float:
    """Black-Scholes-Merton option price"""
    d1 = _d1(S, K, t, r, vol)
    Nd1 = _standard_normal_cdf(d1)
    d2 = _d2(S, K, t, r, vol)
    Nd2 = _standard_normal_cdf(d2)
    return Nd1 * S - Nd2 * K * pow(math.e, -1 * r * t)


def bs_put_price(S: float, K: float, t: float, r: float, vol: float) -> float:
    """Black-Scholes-Merton option price"""
    d1 = _d1(S, K, t, r, vol)
    Nd1 = _standard_normal_cdf(-1 * d1)
    d2 = _d2(S, K, t, r, vol)
    Nd2 = _standard_normal_cdf(-1 * d2)
    return Nd2 * K * pow(math.e, -1 * r * t) - S * Nd1


# -------------------------
# Greeks
# -------------------------

def delta(S: float, K: float, T: float, r: float, vol: float, option: Literal["call", "put"]) -> float:
    d1 = _d1(S, K, T, r, vol)
    if option == "call":
        return _standard_normal_cdf(d1)
    else:
        return _standard_normal_cdf(d1) - 1


def gamma(S, K, t, r, vol) -> float:
    d1 = _d1(S, K, t, r, vol)
    numerator = _standard_normal_pdf(d1)
    denominator = S * vol * math.sqrt(t)
    return numerator / denominator


def vega(S, K, t, r, vol) -> float:
    d1 = _d1(S, K, t, r, vol)
    return S * _standard_normal_pdf(d1) * math.sqrt(t)


def theta(S, K, t, r, vol, option: Literal["call", "put"]) -> float:
    d1 = _d1(S, K, t, r, vol)
    d2 = _d2(S, K, t, r, vol)
    numerator = S * _standard_normal_pdf(d1) * vol
    denominator = 2 * math.sqrt(t)
    if option == "call":
        return -1 * (numerator / denominator) - r * K * pow(math.e, -1 * r * t) * _standard_normal_cdf(d2)
    else:
        return -1 * (numerator / denominator) + r * K * pow(math.e, -1 * r * t) * _standard_normal_cdf(-1 * d2)


def rho(S, K, t, r, vol, option: Literal["call", "put"]) -> float:
    d2 = _d2(S, K, t, r, vol)
    if option == "call":
        return K * t * math.exp(-r * t) * _standard_normal_cdf(d2)
    else:
        return -K * t * math.exp(-r * t) * _standard_normal_cdf(-d2)


def get_10yr_treasury_rate():
    # 10 year treasury ticker symbol
    treasury_ticker = "^TNX"

    now = datetime.now()
    ten_years_ago = now.replace(year=now.year - 10)

    treasury_data = yf.download(treasury_ticker, start=ten_years_ago, end=now)
    last_yield = treasury_data['Close'].iloc[-1]
    return float(last_yield.iloc[0]) / 100


def get_options_data(stock, expiry: str):
    opt_chain = stock.option_chain(expiry)
    puts = opt_chain.puts
    calls = opt_chain.calls
    print("Calls sample:")
    print(calls.head())
    return calls, puts


def calculate_volatility(ticker: str) -> float:
    today = datetime.now()
    one_year_ago = today.replace(year=today.year - 1)
    data = yf.download(ticker, start=one_year_ago, end=today)

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()

    # std of daily returns
    daily_volatility = data['Daily_Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)

    return annualized_volatility


if __name__ == "__main__":
    ticker = "AAPL"
    stock = get_stock_data(ticker)

    # Spot price
    S = _get_spot_price(stock)

    # Treasury yield curve -> get r for chosen maturity
    curves = {label: get_treasury_yield_curve(date) for label, date in dates.items()}
    yc = pd.DataFrame(curves)

    # Choose expiry from options chain
    expiry = stock.options[-1]  # last available expiry date
    t = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365.0
    r = get_10yr_treasury_rate()  # <-- risk-free from US Treasury
    vol = calculate_volatility(ticker)

    print(f"Spot price: {S}, Volatility: {vol}, Time to expiry: {t}, Risk-free rate: {r}")
    if any(map(lambda x: x is None or np.isnan(x) or x == 0, [S, vol, t, r])):
        raise ValueError("One or more input values are invalid (NaN or zero).")

    calls, puts = get_options_data(stock, expiry)

    # Pick one call option to analyze
    row = calls.iloc[0]
    K = row["strike"]
    market_price = row["lastPrice"]

    # Compute theoretical price
    main_df = calls.copy()
    columns_to_drop = ['lastTradeDate', 'lastPrice', 'volume', 'openInterest', 'contractSize', 'currency']
    main_df.drop(columns=columns_to_drop, inplace=True)
    main_df['bsmValuation'] = main_df.apply(lambda row: bs_call_price(S, row['strike'], t, r, vol), axis=1)
    main_df.head(10)

    greeks_df = main_df.copy()
    greeks_df['delta'] = greeks_df.apply(lambda row: delta(S, row['strike'], t, r, vol, "call"), axis=1)
    greeks_df['gamma'] = greeks_df.apply(lambda row: gamma(S, row['strike'], t, r, vol), axis=1)
    greeks_df['vega'] = greeks_df.apply(lambda row: vega(S, row['strike'], t, r, vol), axis=1)
    greeks_df['theta'] = greeks_df.apply(lambda row: theta(S, row['strike'], t, r, vol, "call"), axis=1)
    greeks_df['rho'] = greeks_df.apply(lambda row: rho(S, row['strike'], t, r, vol, "call"), axis=1)

    print(greeks_df.head(10))
