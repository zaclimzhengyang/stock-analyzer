import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01", auto_adjust=False)

# Ensure we are working with a Series, not a DataFrame
data = data["Adj Close"].squeeze()

# Compute rolling mean and std
window = 20
rolling_mean = data.rolling(window).mean()
rolling_std = data.rolling(window).std()

# Z-score = (price - mean) / std
z_score = ((data - rolling_mean) / rolling_std).squeeze()

# Trading signals: Buy if z < -1, Sell if z > 1
buy_signal = z_score < -1
sell_signal = z_score > 1

# Plot
plt.figure(figsize=(12,6))
plt.plot(data, label="Price")
plt.plot(rolling_mean, label="Rolling Mean (20d)", linestyle="--")

plt.scatter(data.index[buy_signal], data[buy_signal],
            marker="^", color="green", label="Buy Signal")

plt.scatter(data.index[sell_signal], data[sell_signal],
            marker="v", color="red", label="Sell Signal")

plt.legend()
plt.title(f"Mean Reversion Strategy on {ticker}")
plt.show()
