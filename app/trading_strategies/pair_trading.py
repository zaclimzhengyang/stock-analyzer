from io import StringIO

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import requests

# === Constants ===
THRESHOLD_P_VALUE = 0.05 # p-value threshold for stationarity and co-integration tests
INDUSTRY = "Information Technology Services"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"


# === Helper Functions ===
def get_sp500_tickers():
    url = "https://www.slickcharts.com/sp500"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    tables = pd.read_html(StringIO(html))
    sp500_df = tables[0]
    return sp500_df['Symbol'].tolist()


def get_industry_tickers(industry: str, limit=100):
    tickers = get_sp500_tickers()
    matched = []

    for t in tickers:
        try:
            info = yf.Ticker(t).info
            if info.get("industry") and info["industry"].lower() in industry.lower():
                matched.append(t)
                if len(matched) == limit:
                    break
        except Exception as _:
            continue

    return matched


def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    return data.dropna()


# === Statistical Analysis Functions ===
def find_cointegrated_pairs(data: pd.DataFrame):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]].dropna()
            S2 = data[keys[j]].dropna()
            # Align indices
            df = pd.concat([S1, S2], axis=1).dropna()

            if len(df) < 50:  # skip if too short
                continue

            result = coint(df.iloc[:, 0], df.iloc[:, 1])
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < THRESHOLD_P_VALUE:
                pairs.append((keys[i], keys[j], pvalue))

    return score_matrix, pvalue_matrix, sorted(pairs, key=lambda x: x[2])


def zscore(series):
    return (series - series.mean()) / np.std(series)


# === Backtest Function ===
def backtest_with_signals(S1, S2, long_signals, short_signals):
    portfolio_value = 1000
    position_S1 = 0
    position_S2 = 0
    pnl_curve = []

    for i in range(len(S1)):
        if long_signals.iloc[i]:
            # Long spread → Buy S1, Sell S2
            position_S1 += 1
            position_S2 -= 1
        elif short_signals.iloc[i]:
            # Short spread → Sell S1, Buy S2
            position_S1 -= 1
            position_S2 += 1
        else:
            # Mark-to-market existing position
            pass

            # Portfolio value = cash + MTM positions
        value = portfolio_value + position_S1 * S1.iloc[i] + position_S2 * S2.iloc[i]
        pnl_curve.append(value)

    return pd.Series(pnl_curve, index=S1.index)


def run_regression_and_get_spread(t1_data, t2_data, t1, t2):
    X = sm.add_constant(t1_data)
    results = sm.OLS(t2_data, X).fit()
    hedge_ratio = results.params[t1]
    spread = t2_data - hedge_ratio * t1_data

    plt.figure(figsize=(12, 6))
    spread.plot()
    plt.axhline(spread.mean(), color='black', linestyle='--')
    plt.title(f"Spread {t2} - {hedge_ratio:.2f} * {t1}")
    plt.show()


def plot_zscore_ratio(t1_data, t2_data, t1, t2):
    ratio = t1_data / t2_data
    z_ratios = zscore(ratio)

    plt.figure(figsize=(12, 6))
    z_ratios.plot()
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.title(f"{t1}/{t2} Z-score of Ratio")
    plt.show()

    return ratio


def generate_signals(ratio, t1, t2):
    # Train/test split
    split_idx = int(len(ratio) * 0.7)
    train, test = ratio[:split_idx], ratio[split_idx:]

    # Moving average/z-score strategy
    mavg5 = train.rolling(5).mean()
    mavg60 = train.rolling(60).mean()
    std60 = train.rolling(60).std()
    zscore_60_5 = (mavg5 - mavg60) / std60

    # Buy/sell signals (long spread when z < -1, short spread when z > +1)
    long_signals = zscore_60_5 < -1
    short_signals = zscore_60_5 > 1

    plt.figure(figsize=(12, 6))
    train[60:].plot()
    train[60:][long_signals[60:]].plot(marker='^', color='g', linestyle='None')
    train[60:][short_signals[60:]].plot(marker='v', color='r', linestyle='None')
    plt.title(f"{t1}/{t2} Trading Signals")
    plt.legend(['Ratio', 'Long Signal', 'Short Signal'])
    plt.show()

    return split_idx, long_signals, short_signals


def map_signals_to_stocks(S1_train, S2_train, long_signals, short_signals, t1, t2):
    # Initialize empty signals
    long_S1 = pd.Series(0.0, index=S1_train.index)
    short_S1 = pd.Series(0.0, index=S1_train.index)
    long_S2 = pd.Series(0.0, index=S2_train.index)
    short_S2 = pd.Series(0.0, index=S2_train.index)

    # Long spread → Buy S1, Sell S2
    long_S1[long_signals] = S1_train[long_signals]
    short_S2[long_signals] = S2_train[long_signals]

    # Short spread → Sell S1, Buy S2
    short_S1[short_signals] = S1_train[short_signals]
    long_S2[short_signals] = S2_train[short_signals]

    # === Plot signals on both stocks ===
    plt.figure(figsize=(12, 6))
    S1_train[60:].plot(color='b', label=t1)
    S2_train[60:].plot(color='c', label=t2)

    # Mark trades
    long_S1[60:][long_S1 != 0].plot(marker='^', color='g', linestyle='None', label=f"Long {t1}")
    short_S1[60:][short_S1 != 0].plot(marker='v', color='r', linestyle='None', label=f"Short {t1}")
    long_S2[60:][long_S2 != 0].plot(marker='^', color='darkgreen', linestyle='None', label=f"Long {t2}")
    short_S2[60:][short_S2 != 0].plot(marker='v', color='darkred', linestyle='None', label=f"Short {t2}")

    plt.title(f"{t1} and {t2} with Pair Trade Signals")
    plt.legend()
    plt.show()


def run_backtest_and_plot(S1_train, S2_train, long_signals, short_signals, t1, t2):
    pnl_curve = backtest_with_signals(S1_train, S2_train, long_signals, short_signals)

    plt.figure(figsize=(12, 6))
    pnl_curve.plot()
    plt.title(f"PnL Curve for {t1}/{t2}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.show()

    print(f"Final PnL for {t1}/{t2}: {pnl_curve.iloc[-1]:.2f}")


def pair_trading_strategy():
    # tickers = get_industry_tickers(INDUSTRY, limit=100)
    # print(f"First {len(tickers)} {INDUSTRY} tickers:", tickers)
    tickers = ["JKHY", "LDOS"]

    data = download_data(tickers, START_DATE, END_DATE)
    _, _, pairs = find_cointegrated_pairs(data)

    print("\nCo-integrated pairs (p < 0.1):")
    for p in pairs:
        print(p)

    for t1, t2, pval in pairs:
        print(f"\n=== Analyzing {t1} vs {t2} (p={pval:.4f}) ===")
        t1_data, t2_data = data[t1], data[t2]

        # Regression for hedge ratio and plot spread
        run_regression_and_get_spread(t1_data, t2_data, t1, t2)

        # Ratio and z-score
        ratio = plot_zscore_ratio(t1_data, t2_data, t1, t2)

        # Generate signals
        split_idx, long_signals, short_signals = generate_signals(ratio, t1, t2)

        # Map to stock trades
        S1_train, S2_train = t1_data[:split_idx], t2_data[:split_idx]
        map_signals_to_stocks(S1_train, S2_train, long_signals, short_signals, t1, t2)

        # Backtest and plot PnL
        run_backtest_and_plot(S1_train, S2_train, long_signals, short_signals, t1, t2)

if __name__ == "__main__":
    pair_trading_strategy()
