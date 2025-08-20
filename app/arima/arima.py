import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# 1. Load data (example: Apple stock closing prices)
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end=None)
ts = data["Close"]

# 2. Fit ARIMA model
# Order = (p, d, q) -> you can tune these values
model = ARIMA(ts, order=(5,1,1))  # (p=5, d=1 for differencing, q=0)
model_fit = model.fit()

# 3. Forecast future values
forecast_steps = 180  # Number of days to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# 4. Plot results
plt.figure(figsize=(12,6))
plt.plot(ts, label="Historical")
plt.plot(pd.date_range(ts.index[-1], periods=forecast_steps+1, freq="B")[1:], forecast, label="Forecast", color="red")
plt.legend()
plt.title(f"ARIMA Forecast for {ticker}")
plt.show()


## Auto arima
# Fit auto_arima model
model = auto_arima(ts, seasonal=False, trace=True)
forecast_steps = 180
auto_arima_forecast = model.predict(n_periods=forecast_steps)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(ts, label="Historical")
future_dates = pd.date_range(ts.index[-1], periods=forecast_steps+1, freq="B")[1:]
plt.plot(future_dates, forecast, label="Forecast", color="red")
plt.legend()
plt.title(f"arima Forecast for {ticker}")
plt.show()

# Plot results
plt.figure(figsize=(12,6))
plt.plot(ts, label="Historical")
future_dates = pd.date_range(ts.index[-1], periods=forecast_steps+1, freq="B")[1:]
plt.plot(future_dates, auto_arima_forecast, label="Forecast", color="red")
plt.legend()
plt.title(f"auto_arima Forecast for {ticker}")
plt.show()