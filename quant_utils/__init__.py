# ...existing code...
# Utility functions for quantitative analysis can be placed here for reuse in notebooks.
# ...existing code...
# ...existing code...
# 1. Import libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 2. Download S&P 500 historical prices
ticker = "^GSPC"
data = yf.download(ticker, start="2010-01-01", end=None)
close = data["Close"]

# 3. Compute daily returns
returns = close.pct_change().dropna()

# 4. Compute annualized return, volatility, Sharpe ratio
mean_daily = returns.mean()
std_daily = returns.std()
annualized_return = mean_daily * 252
annualized_volatility = std_daily * np.sqrt(252)
risk_free_rate = 0.02
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# 5. Plot histogram of returns and normal distribution
plt.figure(figsize=(10,6))
plt.hist(returns, bins=50, density=True, alpha=0.6, label="S&P 500 Returns")
mu, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
plt.plot(x, norm.pdf(x, mu, std), 'r', label="Normal Distribution")
plt.title("S&P 500 Daily Returns Histogram")
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.show()

# 6. Short write-up (markdown cell)
# ...existing code...

