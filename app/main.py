from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

from app.backtest.backtest import get_backtest
from app.black_scholes.black_scholes_option_pricer import bsop
from app.data.downloader import get_price_data, get_fundamentals
from app.factors.momentum import generate_signals, generate_momentum_score
from app.monte_carlo.simulation import mc_simulation
from app.prediction.predictor import StockPredictor, scan_top_nasdaq

app = Flask(__name__)
CORS(app)


@app.route("/api/analyze/<ticker>", methods=["GET"])
def analyze(ticker):
    """
    Analyze a given stock ticker.

    Retrieves price data and fundamental metrics for the specified ticker,
    computes a momentum score (rate of price change over the last 60 days),
    and returns a summary including valuation ratios and risk metrics.

    Financial Description:
    - Momentum Score: Measures the percentage change in price over a 60-day window.
    - Fundamental metrics: Includes P/E, P/B, ROE, Market Cap, Sharpe Ratio, Max Drawdown, etc.
    """
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        price_data = get_price_data(ticker, start_date, end_date)
        fundamentals = get_fundamentals(ticker, price_data)
        score = generate_momentum_score(price_data)

        return jsonify(
            {
                "Ticker ": ticker,
                "Momentum Score": float(score) if score is not None else None,
                **fundamentals,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/<ticker>", methods=["GET"])
def backtest(ticker):
    """
    Backtest a simple momentum trading strategy for a given ticker.

    Generates buy/sell signals based on price crossing above/below its 50-day moving average,
    simulates trading with these signals, and returns the portfolio value over time.

    Financial Description:
    - Buy when price > 50-day MA, sell when price <= 50-day MA.
    - Tracks portfolio value assuming all-in/all-out trades.
    """
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        price_data = get_price_data(ticker, start_date, end_date)
        close_prices = price_data[("Close", ticker)]

        signals = generate_signals(close_prices)

        if len(close_prices) != len(signals):
            raise ValueError(
                f"Price data length {len(close_prices)} != signals length {len(signals)}"
            )

        backtest_result = get_backtest(close_prices, signals)
        result_dict = {str(idx): val for idx, val in backtest_result.items()}

        return jsonify(result_dict)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/monte-carlo-sim/<ticker>", methods=["GET"])
def mc_sim(ticker):
    """
    Run a Monte Carlo simulation for the given ticker.

    Simulates multiple possible future price paths using historical mean returns and covariance,
    and computes Value at Risk (VaR) and Conditional Value at Risk (CVaR) at the 5% level.

    Financial Description:
    - Monte Carlo simulation: Randomly generates price paths to estimate risk.
    - VaR: Maximum expected loss at a given confidence level.
    - CVaR: Expected loss in the worst-case (tail) scenarios.
    """
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        result = mc_simulation(ticker, start_date_dt, end_date_dt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/<ticker>", methods=["GET"])
def predict(ticker):
    """
    Predict buy/sell recommendation for the given ticker using a machine learning model.

    Trains a Random Forest classifier on engineered features (returns, moving averages, volatility)
    and predicts whether the stock is likely to rise in the near future.

    Financial Description:
    - Uses technical indicators as features for classification.
    - Output is a binary recommendation: Buy or Sell.
    """
    try:
        predictor = StockPredictor(ticker)
        predictor.train()
        decision = predictor.predict_latest()
        return jsonify({"ticker": ticker, "recommendation": decision})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nasdaq-buy-recs", methods=["GET"])
def nasdaq():
    """
    Scan top NASDAQ companies and return those with a 'Buy' recommendation.

    For each of the largest NASDAQ stocks by market cap, applies the ML-based predictor
    and returns a list of tickers with a positive outlook.

    Financial Description:
    - Applies the same ML model as in /api/predict to a universe of large-cap NASDAQ stocks.
    """
    try:
        buy_rec = scan_top_nasdaq()
        return jsonify(buy_rec.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/drawdown/<ticker>", methods=["GET"])
def drawdown(ticker):
    """
    Calculate and return the drawdown series for a given ticker.

    Drawdown measures the decline from a historical peak in price, indicating risk and volatility.
    Returns the time series of drawdown values and the maximum drawdown observed.

    Financial Description:
    - Drawdown: (Current Price / Historical Max Price) - 1
    - Max Drawdown: Largest observed drop from peak to trough.
    """
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        price_df = get_price_data(ticker, start_date, end_date)
        dates = price_df.index.strftime("%Y-%m-%d").tolist()

        drawdown = price_df["Close"] / price_df["Close"].cummax() - 1
        drawdown_list = drawdown[ticker].tolist()
        max_drawdown = drawdown.min()

        response = {
            "dates": dates,
            "values": drawdown_list,
            "max_drawdown": float(max_drawdown),
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/black-scholes-model/<ticker>", methods=["GET"])
def bsm(ticker):
    try:
        bsm = bsop(ticker)
        result = bsm[["contractSymbol", "bsmValuation", "delta", "gamma", "vega", "theta", "rho"]]

        response = {
            "Contract Symbol": result["contractSymbol"].tolist(),
            "BSM Valuation": result["bsmValuation"].tolist(),
            "Delta": result["delta"].tolist(),
            "Gamma": result["gamma"].tolist(),
            "Vega": result["vega"].tolist(),
            "Theta": result["theta"].tolist(),
            "Rho": result["rho"].tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
