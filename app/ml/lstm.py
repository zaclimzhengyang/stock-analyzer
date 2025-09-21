import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    # Custom dataset class for handling time series data
    # Converts numpy arrays to PyTorch tensors
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    # LSTM neural network with:
    # - Input layer (size=1 for price data)
    # - Hidden LSTM layers (50 units, 2 layers)
    # - Output layer (size=1 for price prediction)
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def prepare_data(data, lookback):
    # Creates sequences of 'lookback' days (20 days by default)
    # For each sequence, target is the next day's price
    # Example: if lookback=20
    # Input: [day1...day20], Target: day21
    # Input: [day2...day21], Target: day22
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def lstm_forecast(ticker, start_date, end_date, forecast_days=30):
    # Get data
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    ad_close = df['Adj Close']
    values = df['Adj Close'].values
    prices = df['Adj Close'].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    # Prepare training data
    lookback = 20
    X, y = prepare_data(scaled_prices, lookback)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = torch.device('cpu')
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    print("Training model...")
    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        # For each epoch:
        # - Process data in batches
        # - Make predictions
        # - Calculate loss
        # - Update model weights
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    # Generate predictions
    print("Generating forecast...")
    model.eval()
    last_sequence = torch.FloatTensor(scaled_prices[-lookback:]).reshape(1, lookback, 1)
    future_prices = []
    future_dates = pd.date_range(
        start=df.index[-1],
        periods=forecast_days + 1,
        freq='B'
    )[1:]

    with torch.no_grad():
        # Uses last 20 days of actual data
        # Predicts next day
        # Updates sequence with prediction
        # Repeats for forecast_days (30)
        for _ in range(forecast_days):
            # Get prediction
            pred = model(last_sequence)
            future_price = scaler.inverse_transform(pred.numpy())[0, 0]
            future_prices.append(future_price)

            # Update sequence
            last_sequence = torch.roll(last_sequence, -1, dims=1)
            last_sequence[0, -1, 0] = pred[0, 0]

    # Create plot
    plt.close('all')  # Clear any existing plots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    ax.plot(df.index[-100:], df['Adj Close'][-100:], label='Historical', color='blue')

    # Plot forecast
    ax.plot(future_dates, future_prices, 'r--', label='Forecast')

    # Customize plot
    ax.set_title(f'{ticker} Stock Price Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    # Format dates on x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create results dictionary
    results = {
        'future_dates': future_dates,
        'future_prices': future_prices,
        'figure': fig,
        'last_actual_price': df['Adj Close'].iloc[-1],
        'forecast_start_date': future_dates[0],
        'forecast_end_date': future_dates[-1]
    }

    return results

if __name__ == "__main__":
    ticker = "GBTC"
    start_date = "2025-01-01"
    end_date = "2025-09-01"
    forecast = lstm_forecast(ticker, start_date, end_date, forecast_days=30)