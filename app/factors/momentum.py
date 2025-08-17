import pandas as pd
from typing import Optional


def generate_momentum_score(price_df: pd.DataFrame) -> Optional[float]:
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
    """
    if "Close" not in price_df.columns or price_df.empty:
        return None
    idx = max(0, len(price_df) - 60)
    try:
        ret = price_df["Close"].iloc[-1] / price_df["Close"].iloc[idx] - 1
        return round(float(ret), 4)
    except (KeyError, ZeroDivisionError, IndexError):
        return None


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
