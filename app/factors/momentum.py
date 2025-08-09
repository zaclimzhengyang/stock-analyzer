import pandas as pd
from typing import Optional


def generate_momentum_score(price_df: pd.DataFrame) -> Optional[float]:
    """
    Momentum is the speed or velocity of price changes in a stock, security, or tradable instrument.
    It shows the rate of change in price movement over a period, helping investors determine the strength of a trend.
    Stocks that tend to move with the strength of momentum are called momentum stocks.
    Momentum is used by investors to trade stocks in an uptrend by going long (or buying shares) and going short (or selling shares) in a downtrend.
    In other words, a stock can exhibit bullish momentum, meaning the price is rising, or bearish momentum, where the price is steadily falling.
    Since momentum can be quite powerful and indicate a strong trend, investors need to recognize when they're investing with or against the momentum of a stock or the overall market.

    :param price_df:
    :return:
    """
    try:
        if "Close" not in price_df.columns or price_df.empty:
            return None

        idx = max(0, len(price_df) - 60)
        ret = price_df["Close"].iloc[-1] / price_df["Close"].iloc[idx] - 1
        return float(round(ret, 4))
    except (KeyError, ZeroDivisionError):
        return None


def generate_signals(prices: pd.Series) -> list[int]:
    ma50 = prices.rolling(window=50).mean()
    signals = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        ma = ma50.iloc[i]

        if pd.isna(ma):
            signals.append(0)
        elif (price > ma).any():
            signals.append(1)
        else:
            signals.append(-1)

    return signals
