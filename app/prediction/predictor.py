from typing import Optional

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.data.downloader import fetch_nasdaq_companies


class StockPredictor:
    def __init__(self, ticker: Optional[str] = None, days: int = 180):
        self.ticker = ticker or ""
        self.days = days
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, period=f"{self.days}d", interval="1d")
        df.dropna(inplace=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Return"] = df["Close"].pct_change()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["Volatility"] = df["Close"].rolling(window=10).std()
        df["FutureReturn"] = df["Close"].shift(-5) / df["Close"] - 1
        df["Signal"] = (df["FutureReturn"] > 0.02).astype(
            int
        )  # Buy if expected 5-day return > 2%
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        # df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
        df.dropna(inplace=True)
        return df

    def train(self):
        df = self.engineer_features(self.fetch_data())
        features = df[["Return", "MA10", "MA50", "Volatility"]].values
        labels = df["Signal"].values.ravel()
        X_scaled = self.scaler.fit_transform(features)
        X_train, _, y_train, _ = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

    def predict_latest(self) -> str:
        df = self.engineer_features(self.fetch_data())
        latest_features = df[["Return", "MA10", "MA50", "Volatility"]].iloc[-1:]
        latest_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(latest_scaled)[0]
        return "Buy" if prediction == 1 else "Sell"


def scan_top_nasdaq(limit: int = 100, top_k: int = 10) -> pd.DataFrame:
    df = fetch_nasdaq_companies(limit)
    results = []
    for _, row in df.iterrows():
        ticker = row["symbol"]
        market_cap = row["marketCap"]

        try:
            model = StockPredictor(ticker)
            model.train()
            rec = model.predict_latest()
            if rec == "Buy":
                results.append(
                    {"ticker": ticker, "marketCap": market_cap, "recommendation": rec}
                )
        except Exception as e:
            print(f"Error for {ticker}: {e}")

    return pd.DataFrame(results).head(top_k)
