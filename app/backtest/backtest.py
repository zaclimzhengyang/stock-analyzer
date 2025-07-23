from typing import Sequence
import pandas as pd


def get_backtest(
    prices: pd.Series, signals: Sequence[int], initial: float = 100_000
) -> pd.Series:
    cash = initial
    portfolio = 0
    portfolio_values = []

    for price, signal in zip(prices, signals):
        # Convert price to float explicitly
        try:
            price = float(price)
        except ValueError:
            portfolio_values.append(cash + portfolio * 0)
            continue

        if signal == 1 and cash > 0:
            portfolio = cash / price
            cash = 0
        elif signal == -1 and portfolio > 0:
            cash = portfolio * price
            portfolio = 0

        total_value = cash + portfolio * price
        portfolio_values.append(total_value)

    return pd.Series(portfolio_values, index=prices.index)
