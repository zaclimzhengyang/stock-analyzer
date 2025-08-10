from typing import Sequence
import pandas as pd


def get_backtest(
    prices: pd.Series, signals: Sequence[int], initial: float = 100_000
) -> pd.Series:
    """
    Simulate a simple trading strategy based on provided signals.

    For each time step:
    - If signal == 1 and not invested, buy as much as possible.
    - If signal == -1 and invested, sell all holdings.
    - Otherwise, hold current position.

    Returns the portfolio value over time.

    Financial Description:
    - Models an all-in/all-out trading strategy.
    - Useful for evaluating the effectiveness of signal-generating algorithms.
    """
    cash = initial
    portfolio = 0
    portfolio_values = []

    for price, signal in zip(prices, signals):
        price = float(price)
        if signal == 1 and cash > 0:
            portfolio = cash / price
            cash = 0
        elif signal == -1 and portfolio > 0:
            cash = portfolio * price
            portfolio = 0
        total_value = cash + portfolio * price
        portfolio_values.append(total_value)

    return pd.Series(portfolio_values, index=prices.index)
