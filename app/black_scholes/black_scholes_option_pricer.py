import math
from datetime import datetime

import numpy as np
from typing import Literal

import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from scipy.optimize import brentq

from app.data.downloader import get_treasury_yield_curve, get_stock_data
from constants import dates

# from scipy.stats import norm

# Show all rows (or set to None for unlimited)
pd.set_option("display.max_rows", None)

# Show all columns (no truncation)
pd.set_option("display.max_columns", None)

# Set column width (None = unlimited, otherwise put an int)
pd.set_option("display.max_colwidth", None)

# Optional: widen the display to fit your screen
pd.set_option("display.width", 1000)


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
    # return norm.pdf(x) # using scipy
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _standard_normal_cdf(x: float) -> float:
    """Standard normal CDF"""
    # return norm.cdf(x) # using scipy
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


def delta(
        S: float, K: float, T: float, r: float, vol: float, option: Literal["call", "put"]
) -> float:
    """
    DeLta (Δ): Sensitivity Of Option Price To A $1 Change In Underlying Stock.
    Call Δ ∈ [0,1], Put Δ ∈ [-1,0]. Also Represents Hedge Ratio (Shares Per Option).

    Trader view: Equivalent stock exposure (hedge ratio) — e.g. Δ=0.6 ≈ 60 shares per call.
    """
    d1 = _d1(S, K, T, r, vol)
    if option == "call":
        return _standard_normal_cdf(d1)
    else:
        return _standard_normal_cdf(d1) - 1


def gamma(S, K, t, r, vol) -> float:
    """
    Gamma (Γ): Sensitivity of Delta to a $1 change in underlying stock.
    High for ATM options; measures convexity and hedge adjustment rate.

    Trader view: Hedge risk — high Γ means Delta shifts quickly, requiring frequent re-hedging.
    """
    d1 = _d1(S, K, t, r, vol)
    numerator = _standard_normal_pdf(d1)
    denominator = S * vol * math.sqrt(t)
    return numerator / denominator


def vega(S, K, t, r, vol) -> float:
    """
    Vega (ν): Sensitivity of option price to a 1% change in volatility.
    Highest for ATM and longer-dated options; long Vega benefits from higher vol.

    Trader view: Exposure to implied vol — long options = long Vega, profit if vol rises.
    """
    d1 = _d1(S, K, t, r, vol)
    return S * _standard_normal_pdf(d1) * math.sqrt(t)


def theta(S, K, t, r, vol, option: Literal["call", "put"]) -> float:
    """
    Theta (Θ): Sensitivity of option price to 1 day of time decay.
    Usually negative; options lose value as expiration approaches (fastest at ATM).

    Trader view: Option premium erodes with time — short options collect Theta, long options pay Theta.
    """
    d1 = _d1(S, K, t, r, vol)
    d2 = _d2(S, K, t, r, vol)
    numerator = S * _standard_normal_pdf(d1) * vol
    denominator = 2 * math.sqrt(t)
    if option == "call":
        return -1 * (numerator / denominator) - r * K * pow(
            math.e, -1 * r * t
        ) * _standard_normal_cdf(d2)
    else:
        return -1 * (numerator / denominator) + r * K * pow(
            math.e, -1 * r * t
        ) * _standard_normal_cdf(-1 * d2)


def rho(S, K, t, r, vol, option: Literal["call", "put"]) -> float:
    """
    Rho (ρ): Change in option price for a 1% change in interest rates.

    Trader view: Less important than Δ/Γ/ν/Θ — calls gain when rates rise, puts lose.
    """
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

    treasury_data = yf.download(treasury_ticker, start=ten_years_ago, end=now, auto_adjust=False)
    last_yield = treasury_data["Adj Close"].iloc[-1]
    return float(last_yield.iloc[0]) / 100


def get_options_data(stock, expiry: str):
    opt_chain = stock.option_chain(expiry)
    puts = opt_chain.puts
    calls = opt_chain.calls
    print("Calls sample:")
    print(calls.head())
    return calls, puts


def implied_volatility(market_price, S, K, T, r):
    """Solve for implied volatility using Brent's method."""
    try:
        # Root-finding between [1e-6, 5] vol (0.0001% to 500%)
        iv = brentq(
            lambda sigma: bs_call_price(S, K, T, r, sigma) - market_price,
            1e-6, 5.0, maxiter=500
        )
        return iv
    except:
        return np.nan


def calculate_volatility(ticker: str) -> float:
    today = datetime.now()
    one_year_ago = today.replace(year=today.year - 1)
    data = yf.download(ticker, start=one_year_ago, end=today, auto_adjust=False)

    # Calculate daily returns
    data["Daily_Return"] = data["Adj Close"].pct_change()

    # std of daily returns
    daily_volatility = data["Daily_Return"].std()
    annualized_volatility = daily_volatility * np.sqrt(252)

    return annualized_volatility


def bsop(ticker: str) -> pd.DataFrame:
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

    print(
        f"Spot price: {S}, Volatility: {vol}, Time to expiry: {t}, Risk-free rate: {r}"
    )
    if any(map(lambda x: x is None or np.isnan(x) or x == 0, [S, vol, t, r])):
        raise ValueError("One or more input values are invalid (NaN or zero).")

    calls, puts = get_options_data(stock, expiry)

    # Pick one call option to analyze
    row = calls.iloc[0]
    K = row["strike"]
    market_price = row["lastPrice"]

    # Compute theoretical price
    main_df = calls.copy()
    columns_to_drop = [
        "lastTradeDate",
        "volume",
        "openInterest",
        "contractSize",
        "currency",
    ]
    main_df.drop(columns=columns_to_drop, inplace=True)
    main_df["spotPrice"] = S
    main_df["bsmValuation"] = main_df.apply(
        lambda row: bs_call_price(S, row["strike"], t, r, vol), axis=1
    )
    main_df.head(10)

    greeks_df = main_df.copy()
    greeks_df["delta"] = greeks_df.apply(
        lambda row: delta(S, row["strike"], t, r, vol, "call"), axis=1
    )
    greeks_df["gamma"] = greeks_df.apply(
        lambda row: gamma(S, row["strike"], t, r, vol), axis=1
    )
    greeks_df["vega"] = greeks_df.apply(
        lambda row: vega(S, row["strike"], t, r, vol), axis=1
    )
    greeks_df["theta"] = greeks_df.apply(
        lambda row: theta(S, row["strike"], t, r, vol, "call"), axis=1
    )
    greeks_df["rho"] = greeks_df.apply(
        lambda row: rho(S, row["strike"], t, r, vol, "call"), axis=1
    )

    print(greeks_df.head(10))

    return greeks_df

if __name__ == "__main__":
    ticker = "AAPL"
    df = bsop(ticker)
    print(df)
    plt.plot(df["strike"], df["bsmValuation"], label="BSM Theoretical Price")
    plt.scatter(df["strike"], df["lastPrice"], color="red", label="Market Price", alpha=0.5)
    plt.title(f"{ticker} Call Options - BSM Theoretical vs Market Price")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()