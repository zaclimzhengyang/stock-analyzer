from flask import Flask, jsonify
from data.downloader import get_fundamentals, get_price_data
from factors.momentum import momentum_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8501"])

@app.route('/api/analyze/<ticker>', methods=['GET'])
def analyze(ticker):
    price_data = get_price_data(ticker)
    fundamentals = get_fundamentals(ticker)
    score = momentum_score(price_data)

    return jsonify({
        "ticker": ticker,
        "momentum_score": float(score) if score is not None else None,
        **fundamentals
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
