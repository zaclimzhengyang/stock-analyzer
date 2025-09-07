import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download stock data
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01", auto_adjust=False)

# Ensure Series
if isinstance(data["Adj Close"], pd.DataFrame):
    data = data["Adj Close"][ticker]
else:
    data = data["Adj Close"]

# Compute rolling mean and std
window = 20
rolling_mean = data.rolling(window).mean()
rolling_std = data.rolling(window).std()

# Z-score = (price - mean) / std
z_score = (data - rolling_mean) / rolling_std

# Define trading signals: +1 = long, -1 = short, 0 = flat
signal = pd.Series(0, index=data.index)

signal[z_score < -1] = 1    # Buy signal
signal[z_score > 1] = -1    # Sell signal
signal[(z_score > -0.5) & (z_score < 0.5)] = 0   # Exit to cash when back near mean

# Carry forward last non-zero position until exit
position = signal.replace(to_replace=0, method="ffill").fillna(0)

# But when we explicitly set 0 (exit condition), overwrite back to flat
position[signal == 0] = 0

# Daily returns
returns = data.pct_change()

# Strategy returns
strategy_returns = position.shift(1) * returns
cumulative_returns = (1 + strategy_returns).cumprod()

# Plot performance
plt.figure(figsize=(12,6))
plt.plot((1 + returns).cumprod(), label=f"{ticker} Buy & Hold")
plt.plot(cumulative_returns, label="Mean Reversion Strategy")
plt.legend()
plt.title(f"Mean Reversion Backtest on {ticker} (Exit on Reversion)")
plt.show()
