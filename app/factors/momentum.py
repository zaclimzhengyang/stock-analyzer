import pandas as pd
from typing import Optional


def generate_momentum_scores(price_df: pd.DataFrame, windows: list[int] = [30, 60, 90]) -> Optional[dict[int, float]]:
    """
    Calculate the momentum score for a stock.

    The momentum score is the percentage change in closing price over the last 60 days.

    Financial Description:
    - Momentum: The tendency of assets with recent high returns to continue performing well.
    """
    """
    Generate trading signals based on price and its 50-day moving average.

    Returns:
    - 1 if price > 50-day MA (buy signal)
    - -1 if price <= 50-day MA (sell signal)
    - 0 otherwise

    Financial Description:
    - Moving average crossover is a classic trend-following strategy.
    - Momentum Score = (Price_today / Price_60_days_ago) - 1
    """
    if "Adj Close" not in price_df.columns or price_df.empty:
        return None

    scores = {}
    for w in windows:
        # If price data has at least 60 rows, it looks 60 days back from the last row.
        # Else, it just uses the very first row (avoids negative index).
        idx = max(0, len(price_df) - w)
        try:
            ret = price_df["Adj Close"].iloc[-1] / price_df["Adj Close"].iloc[idx] - 1
            scores[w] = round(float(ret), 4)
        except (KeyError, ZeroDivisionError, IndexError):
            return None

    return scores


def generate_signals(prices: pd.Series) -> list[int]:
    """
    Generate trading signals based on price and its 50-day moving average.

    Returns:
    - 1 if price > 50-day MA (buy signal)
    - -1 if price <= 50-day MA (sell signal)
    - 0 otherwise

    Financial Description:
    - Moving average crossover is a classic trend-following strategy.
    """
    ma50 = prices.rolling(window=50).mean()
    signals = pd.Series(0, index=prices.index)
    signals[prices > ma50] = 1
    signals[(prices <= ma50) & (~ma50.isna())] = -1
    return signals.tolist()
