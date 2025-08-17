import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Download S&P 500 data
ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end=None)
close = data["Close"]

# 3. Compute short and long moving averages
short_window = 50
long_window = 200
data["SMA_short"] = close.rolling(window=short_window).mean()
data["SMA_long"] = close.rolling(window=long_window).mean()

# 4. Generate signals
data["Signal"] = 0
data["Signal"][short_window:] = np.where(
    data["SMA_short"][short_window:] > data["SMA_long"][short_window:], 1, 0
)
data["Position"] = data["Signal"].diff()

# 5. Backtest: equity curve (10% allocation per buy)
initial_capital = 100000
cash = initial_capital
shares = 0
portfolio_values = []

for i in range(len(data)):
    signal = data["Signal"].iloc[i]  # hold state (0 or 1)
    price = close.iloc[i]

    if signal == 1 and shares == 0:  # buy once when not holding
        invest_amount = cash * 0.10
        buy_shares = int(invest_amount / price)
        if buy_shares > 0:
            cost = buy_shares * price
            shares += buy_shares
            cash -= cost

    elif signal == 0 and shares > 0:  # exit when signal turns 0
        cash += shares * price
        shares = 0

    total_value = cash + shares * price
    portfolio_values.append(total_value)

data["Portfolio"] = pd.Series(portfolio_values, index=data.index)

# 6. Plot equity curve
plt.figure(figsize=(12,6))
plt.plot(close, label="Price")
plt.plot(data["SMA_short"], label="50d SMA")
plt.plot(data["SMA_long"], label="200d SMA")
plt.plot(data.loc[data["Signal"]==1].index,
         close[data["Signal"]==1],
         "^", markersize=10, color="g", label="Buy")
plt.plot(data.loc[data["Signal"]==0].index,
         close[data["Signal"]==0],
         "v", markersize=10, color="r", label="Sell")
plt.legend()
plt.show()

# 7. Drawdown calculation
data["Portfolio"] = pd.Series(portfolio_values, index=data.index)
data["Portfolio"] = pd.to_numeric(data["Portfolio"], errors="coerce")

rolling_max = data["Portfolio"].cummax()
drawdown = data["Portfolio"] / rolling_max - 1
max_drawdown = drawdown.min()

print(f"Max Drawdown: {max_drawdown:.2%}")

# 8. Win/loss ratio
trades = data[data["Position"] != 0]
