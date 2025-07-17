def momentum_score(price_df):
    try:
        ret = price_df["Close"].iloc[-1] / price_df["Close"].iloc[-60] - 1
        return float(round(ret, 4))
    except Exception:
        return None